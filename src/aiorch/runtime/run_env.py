# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Per-run environment helper.

On the Platform, every run has its own set of configs (plain text)
and secrets (decrypted at claim time) sourced from the DB. These
values must NEVER be written to ``os.environ`` — concurrent runs
would clobber each other. Instead they live in
``context[RUN_ENV_KEY]`` and are read through the helpers in this
module.

Since the security audit (Bug 8), the per-run bucket is **split**
into two sub-buckets:

    context[RUN_ENV_KEY] = {
        "configs": {"REGION": "us-west-2", "LOG_LEVEL": "info", ...},
        "secrets": {"DB_URL": "...", "GITHUB_TOKEN": "...", ...},
    }

The split is load-bearing:

- ``get_env(context, key)`` — used by **in-process** Python callers
  (SMTP connector, S3 connector, etc.) that run inside the executor
  process. These callers are trusted and can read both configs and
  secrets without an allowlist. Calling behaviour is unchanged from
  the flat-dict era.

- ``merge_env(context, secrets_allowed=...)`` — used by the ``run:``
  primitive to build the **subprocess** environment. Here the
  allowlist kicks in: configs are always included, but secrets are
  included **only** if the pipeline author explicitly named them in
  the step's ``secrets:`` list. This closes the Bug 8 leak where
  every shell step in every pipeline inherited every workspace
  secret in its ``environ``, regardless of whether it needed them.

Backward compat: if some caller still hands us a flat str→str
dict (no "configs"/"secrets" sub-keys), we treat the whole thing
as configs. Tests and older callers continue to work; new callers
should build the split shape.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

from aiorch.constants import RUN_ENV_KEY


def _unpack_run_env(run_env: Any) -> tuple[dict[str, str], dict[str, str]]:
    """Return (configs, secrets) for any legitimate run_env shape.

    Supports three shapes:
      1. None or empty — returns two empty dicts.
      2. New split shape {"configs": {...}, "secrets": {...}}.
      3. Legacy flat shape {"KEY": "value", ...} — treated as all
         configs (no secrets). This path keeps older callers and
         tests working during the transition.
    """
    if not run_env or not isinstance(run_env, dict):
        return {}, {}
    configs = run_env.get("configs")
    secrets = run_env.get("secrets")
    if isinstance(configs, dict) or isinstance(secrets, dict):
        return (
            dict(configs) if isinstance(configs, dict) else {},
            dict(secrets) if isinstance(secrets, dict) else {},
        )
    # Legacy flat dict — treat everything as configs. Nothing is
    # "secret" from the allowlist's perspective, so every key ends
    # up in subprocess env, same as the pre-split behaviour.
    return {k: v for k, v in run_env.items() if isinstance(v, str)}, {}


def get_env(context: dict[str, Any] | None, key: str, default: str = "") -> str:
    """Read a config/secret value for an **in-process** caller.

    Checks the per-run bucket first (secrets win over configs on
    collision, matching pre-split behaviour), then falls back to
    ``os.environ``. Returns ``default`` if the key is not found.

    This is the only function in-process handlers (SMTP, S3, Kafka,
    webhook, etc.) should use for reading workspace-scoped configs
    and secrets. It does NOT consult any allowlist — in-process
    Python runs in the executor's own trust domain, so it already
    has access to everything. The allowlist only applies to
    subprocess env via ``merge_env``.
    """
    if context is not None:
        configs, secrets = _unpack_run_env(context.get(RUN_ENV_KEY))
        # Secrets take precedence so a secret can override a config
        # of the same key (rare but the pre-split behaviour).
        if key in secrets:
            return secrets[key]
        if key in configs:
            return configs[key]
    return os.environ.get(key, default)


def merge_env(
    context: dict[str, Any] | None,
    *,
    secrets_allowed: Iterable[str] | None = None,
) -> dict[str, str]:
    """Return a merged env dict suitable for ``subprocess.Popen(env=...)``.

    Composition order (later entries win):
      1. ``os.environ`` (infrastructure baseline).
      2. Per-run ``configs`` bucket (workspace configs, always
         included — they're plain text by definition).
      3. Per-run ``secrets`` bucket, but only the keys named in
         ``secrets_allowed``. Keys not in the allowlist are dropped.

    The caller's ``os.environ`` is left untouched. Per-run values
    override ``os.environ`` on key collision; allowed secrets
    override configs on key collision.

    Args:
        context: Execution context (see ``build_execution_context``).
        secrets_allowed: Iterable of secret keys the calling step
            has explicitly declared it needs. Any key not in this
            set is stripped from the returned dict. Pass ``None``
            (the default) for "no secrets" — meaning the subprocess
            runs with configs + os.environ only.
    """
    merged = dict(os.environ)
    if context is None:
        return merged

    configs, secrets = _unpack_run_env(context.get(RUN_ENV_KEY))
    # Configs always included — they're not sensitive by definition.
    merged.update(configs)

    if secrets_allowed:
        allowed = set(secrets_allowed)
        for key, value in secrets.items():
            if key in allowed:
                merged[key] = value

    return merged


_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def expand_env_vars(value: Any, context: dict[str, Any] | None = None) -> Any:
    """Recursively expand ${VAR} references using per-run env + os.environ.

    Used by MCP server specs and similar config blocks that accept
    shell-style ${VAR} interpolation. Unlike os.path.expandvars() this
    checks the per-run bucket first (configs + secrets, via get_env),
    so workspace-scoped values resolve correctly without being written
    to os.environ. Called from in-process Python, not from subprocess
    env composition, so no allowlist applies.
    """
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(
            lambda m: get_env(context, m.group(1), m.group(0)),
            value,
        )
    if isinstance(value, dict):
        return {k: expand_env_vars(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v, context) for v in value]
    return value
