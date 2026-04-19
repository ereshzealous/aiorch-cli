# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Connector handlers — one class per (type, subtype) pair.

Each handler implements ``ConnectorHandler.execute(operation, params,
connector_config, context)`` and returns the canonical result shape
for that operation (see ``aiorch.connectors.CANONICAL_OPERATIONS``).

Handlers are **stateless**. Per Q2 of the connector design doc, we
open connections on-demand and close them after the operation
completes. No pools, no caches, no per-operation state kept between
invocations. A handler instance can be a process-wide singleton
because it holds no mutable state.

Concrete handlers live in sibling modules (``postgres.py``,
``object_store.py``, etc.) and register themselves with the
``HandlerRegistry`` at import time.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from aiorch.connectors import NotSupportedError

logger = logging.getLogger("aiorch.connectors.handlers")


class ConnectorHandler(ABC):
    """Abstract base for every connector handler.

    Subclasses set ``type``, ``subtype``, and ``supported_operations``
    as class attributes and implement ``execute()``.
    """

    type: str = ""
    subtype: str = ""
    supported_operations: set[str] = set()

    @abstractmethod
    async def execute(
        self,
        *,
        operation: str,
        params: dict,
        connector_config: dict,
        context: dict,
    ) -> Any:
        """Run a canonical operation against the external system.

        Args:
            operation: The canonical operation name (query, write, send, ...).
            params: Operation-specific arguments (sql, key, message, ...).
                Already resolved through Jinja2 by the time this runs.
            connector_config: Flattened config + secrets dict from
                ``connector.merged_config()``. Handlers read everything
                they need from here.
            context: The pipeline execution context. Handlers use this
                for logging and access to RUN_ENV_KEY if needed.

        Returns:
            Canonical result for the operation type (see section 6.3
            of the connectors proposal doc).

        Raises:
            NotSupportedError: if the operation isn't in
                ``supported_operations`` or can't be mapped to the
                subtype's native API.
            Handler-native exceptions (asyncpg.*, botocore.*, httpx.*)
            for runtime failures. The dispatch wrapper captures and
            re-raises with context.
        """

    def _check_supported(self, operation: str) -> None:
        if operation not in self.supported_operations:
            raise NotSupportedError(
                f"{self.__class__.__name__} does not support operation "
                f"'{operation}'. Supported: {sorted(self.supported_operations)}"
            )

    @classmethod
    def get_client_info(cls) -> dict | None:
        """Return metadata about the underlying client library.

        Surfaced by the /connectors/meta/types route so the UI can
        render a "Powered by aiokafka 0.11.0 — docs ↗" pill near the
        form. Subclasses override to return:

            {
              "client_library": "aiokafka",
              "client_version": "0.11.0",
              "docs_url": "https://kafka.apache.org/documentation/...",
            }

        Returning None (the default) means the UI shows nothing for
        that handler — useful for placeholders and stubs.
        """
        return None


class HandlerRegistry:
    """Process-wide map of (type, subtype) → handler instance."""

    def __init__(self) -> None:
        self._handlers: dict[tuple[str, str], ConnectorHandler] = {}

    def register(self, handler: ConnectorHandler) -> None:
        key = (handler.type, handler.subtype)
        if key in self._handlers:
            logger.debug("Re-registering handler for %s", key)
        self._handlers[key] = handler

    def get(self, type: str, subtype: str) -> ConnectorHandler:
        key = (type, subtype)
        if key not in self._handlers:
            raise LookupError(
                f"No connector handler registered for ({type}, {subtype}). "
                f"Registered: {sorted(self._handlers.keys())}"
            )
        return self._handlers[key]

    def list(self) -> list[tuple[str, str]]:
        return sorted(self._handlers.keys())


# Process-wide singleton. Handlers register themselves on import.
_registry = HandlerRegistry()


def register_handler(handler: ConnectorHandler) -> None:
    """Register a handler in the process-wide registry."""
    _registry.register(handler)


def get_handler(type: str, subtype: str) -> ConnectorHandler:
    """Look up a handler by (type, subtype)."""
    return _registry.get(type, subtype)


def list_handlers() -> list[tuple[str, str]]:
    """List registered (type, subtype) pairs."""
    return _registry.list()


__all__ = [
    "ConnectorHandler",
    "HandlerRegistry",
    "register_handler",
    "get_handler",
    "list_handlers",
]


# ---------------------------------------------------------------------------
# Side-effect imports — concrete handlers register themselves at import
# time via register_handler(). Importing them here is what populates the
# process-wide registry; without this, the server process starts with an
# empty registry and every dispatch raises LookupError.
#
# Each import is wrapped so a missing optional dependency (asyncpg for
# postgres, boto3 for object_store, aiokafka for stream, aiosmtplib for
# email) doesn't block the server from starting — the unavailable
# handler simply stays unregistered and any pipeline that tries to use
# it gets a clear "No connector handler registered for..." error.
# ---------------------------------------------------------------------------


def _load_builtin_handlers() -> None:
    """Import every shipped handler module so they self-register."""
    import importlib
    for modname in (
        "aiorch.connectors.handlers.postgres",
        "aiorch.connectors.handlers.object_store",
        "aiorch.connectors.handlers.kafka",
        "aiorch.connectors.handlers.webhook",
        "aiorch.connectors.handlers.email",
    ):
        try:
            importlib.import_module(modname)
        except ImportError as exc:
            logger.info(
                "Connector handler %s not loaded (optional dep missing): %s",
                modname, exc,
            )
        except Exception as exc:
            logger.warning(
                "Connector handler %s failed to load: %s", modname, exc,
            )


_load_builtin_handlers()
