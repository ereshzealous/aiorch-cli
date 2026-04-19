# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Inline Python primitive handler.

The `python:` step primitive runs a block of Python source inside
the executor's own interpreter, not a subprocess. That avoids the
30-100ms per-step startup cost of `run: python3 -c ...` and gives
steps access to the same site-packages the executor was built with
(httpx, pydantic, jinja2, stdlib, etc.) with zero quoting pain.

Contract (matches docs/python-primitive-proposal.md):

  - The body runs in a fresh namespace containing:
      `inputs`     — dict of all in-scope context variables
                     (pipeline inputs, step outputs so far, vars,
                     foreach item if applicable). Deep-copied so
                     mutations in the body don't leak.
      `step_name`  — str, the current step name.
      `result`     — pre-seeded to None; the body assigns a
                     JSON-serializable value to make it the step's
                     `output:` variable.

  - If the body sets `result`, that value (after JSON-serializability
    check) is the step output.
  - If the body does NOT set `result`, captured stdout becomes the
    step output — matching the `run:` shell contract.
  - Any exception is re-raised with the traceback attached; the
    executor's normal step-error path handles retry semantics.
  - `step.timeout` is enforced by running the body in a thread
    executor with `asyncio.wait_for`. On timeout the step fails
    cleanly; the runaway thread continues until it exits on its
    own (Python can't kill threads from outside), but the executor
    is free to continue scheduling other work.

Security model: trusted-code, identical to `run:`. Anyone who can
author a pipeline already runs arbitrary shell via `run:` — adding
arbitrary Python is not a privilege escalation. Workspace secrets
still honor the step's `secrets:` allowlist, filtered upstream by
the same merge_env path that `run:` uses.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import logging
import sys
import threading
import traceback
from typing import Any

from aiorch.core.parser import Step
from aiorch.core.utils import parse_duration

logger = logging.getLogger(__name__)


class _ThreadLocalStdoutProxy:
    """Per-thread stdout redirection.

    `contextlib.redirect_stdout` swaps `sys.stdout` process-wide, so
    two concurrent `python:` steps would interleave their output into
    each other's buffers. This proxy installs ONCE at module import
    and forwards every write to a thread-local buffer when one is set
    (via `install()` / `uninstall()`), otherwise to the real stdout.

    The executor runs `python:` bodies in a thread pool, so the
    thread-local lookup is stable for the duration of one body.
    Parallel DAG branches and parallel foreach iterations each get
    their own thread and their own buffer.
    """

    def __init__(self, real: Any):
        self._real = real
        self._local = threading.local()

    def _active_buf(self) -> io.StringIO | None:
        return getattr(self._local, "buf", None)

    def install(self, buf: io.StringIO) -> None:
        self._local.buf = buf

    def uninstall(self) -> None:
        if hasattr(self._local, "buf"):
            del self._local.buf

    def write(self, s: str) -> int:
        buf = self._active_buf()
        if buf is not None:
            return buf.write(s)
        return self._real.write(s)

    def flush(self) -> None:
        buf = self._active_buf()
        if buf is not None:
            buf.flush()
            return
        self._real.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


_STDOUT_PROXY: _ThreadLocalStdoutProxy | None = None


def _ensure_stdout_proxy_installed() -> None:
    """Lazy-install the thread-local stdout proxy.

    Called from the handler on first dispatch instead of at import
    time so the test suite / other consumers that never hit `python:`
    never see sys.stdout mutated.
    """
    global _STDOUT_PROXY
    if _STDOUT_PROXY is None:
        _STDOUT_PROXY = _ThreadLocalStdoutProxy(sys.stdout)
        sys.stdout = _STDOUT_PROXY  # type: ignore[assignment]


async def python_handler(step: Step, context: dict[str, Any]) -> Any:
    """Run the step's `python:` body and return its output.

    This is the handler registered as the `python` primitive in
    `runtime.registry._register_builtins()`.
    """
    if not step.python:
        raise ValueError(f"Step '{step.name}': python primitive body is empty")

    from aiorch.constants import LOGGER_KEY
    run_logger = context.get(LOGGER_KEY)

    _ensure_stdout_proxy_installed()
    assert _STDOUT_PROXY is not None  # for type-checkers

    if run_logger:
        run_logger.log(step.name, "DEBUG", "Python body about to run", {
            "lines": step.python.count("\n") + 1,
            "source": step.python[:1000],
        })

    # Compile once so syntax errors surface with a clear
    # "<pipeline step foo>" filename in the traceback.
    try:
        code = compile(step.python, f"<pipeline step {step.name}>", "exec")
    except SyntaxError as e:
        raise RuntimeError(
            f"Step '{step.name}': python primitive has a syntax error\n"
            f"  {e.msg} at line {e.lineno}, column {e.offset}\n"
        ) from e

    # Deep-copy the context so bodies can't mutate shared state and
    # confuse downstream steps. The immutable-inputs rule from the
    # design doc (decision #3).
    namespace: dict[str, Any] = {
        "inputs": copy.deepcopy(context),
        "step_name": step.name,
        "__name__": f"aiorch_step_{step.name}",
        "result": None,
    }

    buf = io.StringIO()
    timeout = parse_duration(step.timeout)

    def _run_body() -> Any:
        _STDOUT_PROXY.install(buf)
        try:
            exec(code, namespace)
        finally:
            _STDOUT_PROXY.uninstall()
        return namespace.get("result")

    loop = asyncio.get_running_loop()

    try:
        if timeout is not None:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _run_body),
                timeout=timeout,
            )
        else:
            result = await loop.run_in_executor(None, _run_body)
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"Step '{step.name}': python body exceeded timeout of {timeout}s. "
            f"The runaway thread will continue until it returns — Python "
            f"can't interrupt threads from outside — but the step is "
            f"recorded as failed and the executor continues."
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(
            f"Step '{step.name}': python body raised {type(e).__name__}\n"
            f"{tb}"
        ) from e

    stdout_text = buf.getvalue()

    if run_logger and stdout_text.strip():
        run_logger.log(step.name, "DEBUG", "Python stdout captured", {
            "bytes": len(stdout_text),
            "stdout": stdout_text[:2000],
        })

    # If the body didn't set `result`, fall back to captured stdout —
    # matches the `run:` contract where the step output is whatever
    # the command printed.
    if result is None:
        return stdout_text.strip()

    # JSON-serializability check. Fail loudly instead of letting a
    # file handle or socket leak into the downstream context where
    # it would break the first step that tried to JSON-encode it for
    # the Trace panel.
    try:
        json.dumps(result)
    except TypeError as e:
        raise RuntimeError(
            f"Step '{step.name}': python `result` is not JSON-serializable: {e}\n"
            f"Assign a dict, list, str, int, float, bool, or None to `result`. "
            f"Got: {type(result).__name__}"
        )

    # If the body both printed AND set result, we keep the structured
    # result as the step output but forward the printed text to the
    # executor's own logs so it still shows up in Trace.
    if stdout_text.strip():
        logger.info(
            "python step '%s' stdout:\n%s",
            step.name,
            stdout_text.rstrip(),
        )

    return result
