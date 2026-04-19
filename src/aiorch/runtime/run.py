# Copyright 2026 Eresh Gorantla
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shell execution runtime for the `run` primitive."""

from __future__ import annotations

import asyncio


async def execute_run(
    command: str,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Execute a shell command and return its stdout.

    ``env`` is the full environment dict handed to the subprocess. The
    caller is responsible for merging the per-run env bucket with
    os.environ (use ``runtime.run_env.merge_env(context)``). If ``env``
    is None, the subprocess inherits the parent's os.environ directly.
    """
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise TimeoutError(f"Command timed out after {timeout}s: {command}")

    if proc.returncode != 0:
        err_msg = stderr.decode().strip() if stderr else "Unknown error"
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {command}\n{err_msg}"
        )

    return stdout.decode().strip()
