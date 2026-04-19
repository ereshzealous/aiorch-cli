# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Connectors — named, managed connections to external systems.

Providers connect to models. Connectors connect to systems.

A connector is a Platform-managed record describing how to talk to
an external system — a Postgres database, an S3 bucket, a Kafka
cluster, a Slack webhook, an SMTP server. Pipelines reference them
by name and never touch credentials directly.

This module exposes:

  - Artifact types: ``Connector`` dataclass and enums
  - Exception hierarchy: ``ConnectorError``, ``ConnectorNotFound``,
    ``ConnectorOperationDenied``, ``ConnectorAuthError``,
    ``ConnectorTimeout``, ``NotSupportedError``
  - The ``get_connector_store()`` singleton factory

Concrete handlers live under ``aiorch.connectors.handlers``. The
central dispatch wrapper lives in ``aiorch.connectors.dispatch``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger("aiorch.connectors")


# ---------------------------------------------------------------------------
# Type taxonomy
# ---------------------------------------------------------------------------

ConnectorType = Literal["database", "object_store", "stream", "webhook", "email"]

# Valid subtypes per type — updated as new backends land.
#
# Note for `stream`: the canonical names are `kafka-producer` and
# `kafka-consumer`. The legacy `kafka` name is kept here for back-compat
# validation of connectors created before the rename — the handler is
# registered under both names so existing rows resolve unchanged. The
# UI metadata endpoint hides `kafka` so new connectors only see the
# new names.
VALID_SUBTYPES: dict[str, set[str]] = {
    "database": {"postgres"},
    "object_store": {"s3", "minio", "r2", "gcs"},
    "stream": {"kafka", "kafka-producer", "kafka-consumer"},
    "webhook": {"generic", "slack", "discord", "teams"},
    "email": {"smtp"},
}

# Subtypes hidden from the UI metadata endpoint — kept in VALID_SUBTYPES
# above only for back-compat with existing connector rows.
LEGACY_HIDDEN_SUBTYPES: dict[str, set[str]] = {
    "stream": {"kafka"},
}

# Operations per type — the canonical shape every handler must support
# (a handler may raise NotSupportedError for any it can't map natively)
CANONICAL_OPERATIONS: dict[str, set[str]] = {
    "database": {"query", "insert", "update", "upsert"},
    "object_store": {"read", "write", "list", "delete"},
    "stream": {"publish"},
    "webhook": {"send"},
    "email": {"send"},
}

# Write operations — used by the read-only gate in the dispatcher
WRITE_OPERATIONS: set[str] = {
    "insert", "update", "upsert",       # database
    "write", "delete",                  # object_store
    "publish",                          # stream
    "send",                             # webhook + email
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class Connector:
    """Runtime representation of a connector row.

    ``config`` holds non-secret values (endpoints, bucket names,
    SASL mechanism, region, etc.). ``secrets`` holds decrypted
    secret values and is ONLY populated by the store's internal
    decrypt path — never stored or transmitted as-is.
    """

    id: str
    org_id: str
    workspace_id: str | None
    pipeline_name: str | None
    name: str
    type: ConnectorType
    subtype: str
    capabilities: list[str]
    config: dict[str, Any] = field(default_factory=dict)
    secrets: dict[str, Any] = field(default_factory=dict)
    is_default: bool = False
    is_read_only: bool = False
    is_active: bool = True
    created_by: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def merged_config(self) -> dict[str, Any]:
        """Decrypted merge of config + secrets for the handler to use.

        Handlers get one flat dict with both non-secret and secret
        fields. The split exists only at the storage and display layer.
        """
        merged = dict(self.config)
        merged.update(self.secrets)
        return merged

    def supports(self, operation: str) -> bool:
        return operation in self.capabilities


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class ConnectorError(Exception):
    """Base class for all connector failures."""


class ConnectorNotFound(ConnectorError):
    """Raised when a connector cannot be resolved from scope + name."""


class ConnectorOperationDenied(ConnectorError):
    """Raised when an operation is blocked by capabilities or read-only.

    This is thrown before the handler runs — no side effects.
    """


class ConnectorAuthError(ConnectorError):
    """Authentication failed against the external system."""


class ConnectorTimeout(ConnectorError):
    """Operation exceeded the configured timeout."""


class NotSupportedError(ConnectorError):
    """The subtype's handler doesn't support this canonical operation.

    Example: an SNS stream handler (hypothetical future) might not
    support the canonical ``metadata`` field of publish.
    """


class ConnectorValidationError(ConnectorError):
    """Config or secrets failed validation at creation time."""


# ---------------------------------------------------------------------------
# Singleton factory (implemented in store.py but exposed here)
# ---------------------------------------------------------------------------


_store: "Any" = None   # ConnectorStore, typed loosely to avoid circular import


def get_connector_store() -> "Any":
    """Return the global ConnectorStore singleton."""
    global _store
    if _store is None:
        from aiorch.connectors.store import ConnectorStore
        _store = ConnectorStore()
    return _store


def set_connector_store(store: "Any") -> None:
    """Replace the singleton (tests)."""
    global _store
    _store = store


__all__ = [
    "Connector",
    "ConnectorType",
    "VALID_SUBTYPES",
    "LEGACY_HIDDEN_SUBTYPES",
    "CANONICAL_OPERATIONS",
    "WRITE_OPERATIONS",
    "ConnectorError",
    "ConnectorNotFound",
    "ConnectorOperationDenied",
    "ConnectorAuthError",
    "ConnectorTimeout",
    "NotSupportedError",
    "ConnectorValidationError",
    "get_connector_store",
    "set_connector_store",
]
