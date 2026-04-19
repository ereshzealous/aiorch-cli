# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Platform-overridable hooks for the connector runtime.

Connector dispatch, handlers, and the connector config store used to
import from ``aiorch.server.{audit,crypto,routes.secrets}``. That
worked in the monorepo but couples CLI code to Platform's FastAPI
surface, blocking a clean CLI extraction.

Three hook groups here, each with a CLI-friendly default:

  * **secret resolver** — how to look up a workspace-scoped secret
    at dispatch time. CLI default: ``os.environ.get`` on the key
    name. Platform: workspace_secrets table with pipeline-scoped
    overlays.

  * **audit sink** — where to record connector events. CLI default:
    no-op. Platform: write to ``audit_logs`` table.

  * **secret codec** — how to encrypt / decrypt connector config
    payloads at rest. CLI default: identity (plaintext; CLI doesn't
    persist encrypted secrets). Platform: AES-256-GCM with the
    app's data-encryption key.

Hooks are process-global. Platform's lifespan registers real
implementations during boot; CLI leaves them as defaults.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Protocol


# ---- Secret resolver -----------------------------------------------------

class SecretResolver(Protocol):
    """Batch-lookup secrets for a workspace + (optional) pipeline scope.

    Platform implementation walks workspace_secrets + workspace_configs
    and returns every key/value pair for the scope — matching the
    existing ``aiorch.server.routes.secrets.resolve_secrets`` signature
    that connectors/dispatch.py used to import directly.
    """

    def resolve_all(
        self,
        *,
        workspace_id: str | None,
        pipeline_name: str | None,
    ) -> dict[str, str]: ...


class _EmptySecretResolver:
    """Default CLI resolver — CLI has no workspace, so no workspace-scoped
    secrets. Returns ``{}``; callers fall back to os.environ at resolve
    time through the pipeline's own ``${ENV_VAR}`` interpolation."""

    def resolve_all(self, *, workspace_id, pipeline_name):
        return {}


_resolver: SecretResolver = _EmptySecretResolver()


def set_secret_resolver(r: SecretResolver) -> None:
    global _resolver
    _resolver = r


def resolve_secrets(
    workspace_id: str | None,
    *,
    pipeline_name: str | None = None,
) -> dict[str, str]:
    """Keep the existing call-site signature — same args as the old
    ``aiorch.server.routes.secrets.resolve_secrets`` — so dispatch code
    stays identical."""
    return _resolver.resolve_all(
        workspace_id=workspace_id,
        pipeline_name=pipeline_name,
    )


# ---- Audit sink ----------------------------------------------------------

AuditSink = Callable[..., None]


def _noop_audit(*args: Any, **kwargs: Any) -> None:
    return None


_audit: AuditSink = _noop_audit


def set_audit_sink(fn: AuditSink) -> None:
    global _audit
    _audit = fn


def audit(*args: Any, **kwargs: Any) -> None:
    """Record an audit event — no-op by default (CLI), Platform wires
    a real writer on boot."""
    _audit(*args, **kwargs)


# ---- Secret codec --------------------------------------------------------
#
# Platform encrypts connector config payloads at rest (AES-256-GCM with
# the app's data-encryption key). CLI stores them as plaintext — no
# encryption, no key management. Codec hooks let both modes share the
# same connector-store code path.

_encode: Callable[[str], str] = lambda s: s  # identity
_decode: Callable[[str], str] = lambda s: s  # identity


def set_secret_codec(
    *,
    encode: Callable[[str], str],
    decode: Callable[[str], str],
) -> None:
    global _encode, _decode
    _encode = encode
    _decode = decode


def encode_secret(plaintext: str) -> str:
    return _encode(plaintext)


def decode_secret(stored: str) -> str:
    return _decode(stored)
