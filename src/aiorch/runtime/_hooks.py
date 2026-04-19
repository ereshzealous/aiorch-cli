# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Platform-overridable hooks for the LLM runtime.

Runtime code (``runtime/llm.py``) used to import directly from
``aiorch.server.spend`` and ``aiorch.server.crypto`` for budget
enforcement and API-key decryption. Those imports are fine in the
monorepo but prevent a clean CLI extraction — the OSS CLI package
can't depend on Platform's FastAPI server code.

This module replaces the direct imports with three registered hooks.
CLI defaults are no-ops (or identity in the case of the decryptor).
Platform registers real implementations at app startup via
``aiorch.server.bootstrap``.

Hooks are process-global. Runtime does NOT cache hook references — the
module-level lookups here are hot-path but the call sites all check
``if hook is not None`` before invoking, so registering after boot
is safe but unusual.
"""

from __future__ import annotations

from typing import Callable

# Budget hooks — both take (provider_id, incoming_usd | cost_usd) and
# may raise (pre-check) or silently record (post-charge). CLI never
# sets provider_id so these hooks are never called in CLI mode; they
# exist so Platform can inject budget enforcement without runtime/
# importing server.spend.
_pre_call_hook: Callable[..., None] | None = None
_post_call_hook: Callable[..., None] | None = None

# Provider API-key decryptor — called by LitellmClient when loading
# a Platform-managed provider's stored ciphertext. CLI providers use
# plaintext env vars so this is only wired on Platform.
_provider_key_decryptor: Callable[[str], str] | None = None


def set_pre_call_hook(fn: Callable[..., None] | None) -> None:
    global _pre_call_hook
    _pre_call_hook = fn


def set_post_call_hook(fn: Callable[..., None] | None) -> None:
    global _post_call_hook
    _post_call_hook = fn


def set_provider_key_decryptor(fn: Callable[[str], str] | None) -> None:
    global _provider_key_decryptor
    _provider_key_decryptor = fn


def pre_call_hook() -> Callable[..., None] | None:
    return _pre_call_hook


def post_call_hook() -> Callable[..., None] | None:
    return _post_call_hook


def provider_key_decryptor() -> Callable[[str], str] | None:
    return _provider_key_decryptor
