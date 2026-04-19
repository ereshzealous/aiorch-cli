# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""ConnectorStore — CRUD + scoped lookup for the connectors table.

Thin wrapper over the existing ``aiorch.storage`` layer. Connectors
are stored in Postgres only (Platform-only feature per Q7); the CLI
path fails gracefully with a clear error if anything calls this module
in CLI mode.

Secrets are encrypted at rest via ``aiorch.server.crypto.encrypt()``
using AIORCH_SECRET_KEY. Same pattern as provider API keys and
workspace secret values.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from aiorch.connectors import (
    CANONICAL_OPERATIONS,
    Connector,
    ConnectorNotFound,
    ConnectorType,
    ConnectorValidationError,
    VALID_SUBTYPES,
)

logger = logging.getLogger("aiorch.connectors.store")


class ConnectorStore:
    """CRUD + scoped resolution for the connectors table."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _db(self):
        from aiorch.storage import get_store
        return get_store()

    def _encrypt_secrets(self, secrets: dict[str, Any] | None) -> str | None:
        if not secrets:
            return None
        from aiorch.connectors._hooks import encode_secret
        return encode_secret(json.dumps(secrets))

    def _decrypt_secrets(self, encrypted: str | None) -> dict[str, Any]:
        if not encrypted:
            return {}
        from aiorch.connectors._hooks import decode_secret
        try:
            return json.loads(decode_secret(encrypted))
        except Exception as e:
            logger.warning("Failed to decrypt connector secrets: %s", e)
            return {}

    def _row_to_connector(self, row: dict, *, include_secrets: bool = False) -> Connector:
        if not row:
            raise ValueError("Cannot build Connector from empty row")

        # capabilities is stored as comma-separated text
        caps_raw = row.get("capabilities") or ""
        capabilities = [c.strip() for c in caps_raw.split(",") if c.strip()]

        # config_json may come back as a dict (asyncpg JSONB) or a string (SQLite)
        cfg_raw = row.get("config_json")
        if isinstance(cfg_raw, str):
            try:
                config = json.loads(cfg_raw)
            except json.JSONDecodeError:
                config = {}
        elif isinstance(cfg_raw, dict):
            config = cfg_raw
        else:
            config = {}

        secrets: dict[str, Any] = {}
        if include_secrets:
            secrets = self._decrypt_secrets(row.get("secrets_enc"))

        return Connector(
            id=row["id"],
            org_id=row["org_id"],
            workspace_id=row.get("workspace_id"),
            pipeline_name=row.get("pipeline_name"),
            name=row["name"],
            type=row["type"],
            subtype=row["subtype"],
            capabilities=capabilities,
            config=config,
            secrets=secrets,
            is_default=bool(row.get("is_default", False)),
            is_read_only=bool(row.get("is_read_only", False)),
            is_active=bool(row.get("is_active", True)),
            created_by=row.get("created_by"),
            created_at=float(row.get("created_at", 0) or 0),
            updated_at=float(row.get("updated_at", 0) or 0),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(
        self,
        *,
        type_: str,
        subtype: str,
        capabilities: list[str],
    ) -> None:
        if type_ not in VALID_SUBTYPES:
            raise ConnectorValidationError(
                f"Unknown connector type: '{type_}'. "
                f"Valid types: {', '.join(sorted(VALID_SUBTYPES))}"
            )
        if subtype not in VALID_SUBTYPES[type_]:
            raise ConnectorValidationError(
                f"Subtype '{subtype}' is not valid for type '{type_}'. "
                f"Valid subtypes: {', '.join(sorted(VALID_SUBTYPES[type_]))}"
            )

        valid_ops = CANONICAL_OPERATIONS[type_]
        for cap in capabilities:
            if cap not in valid_ops:
                raise ConnectorValidationError(
                    f"Capability '{cap}' is not valid for type '{type_}'. "
                    f"Valid capabilities: {', '.join(sorted(valid_ops))}"
                )

        if not capabilities:
            raise ConnectorValidationError(
                f"At least one capability is required for type '{type_}'."
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        org_id: str,
        name: str,
        type: str,
        subtype: str,
        capabilities: list[str],
        config: dict[str, Any] | None = None,
        secrets: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        pipeline_name: str | None = None,
        is_default: bool = False,
        is_read_only: bool = False,
        created_by: str | None = None,
    ) -> Connector:
        """Create a new connector. Raises ConnectorValidationError on bad input."""
        self._validate(type_=type, subtype=subtype, capabilities=capabilities)

        connector_id = str(uuid.uuid4())
        now = time.time()
        config_json = json.dumps(config or {})
        secrets_enc = self._encrypt_secrets(secrets)
        caps_str = ",".join(capabilities)

        self._db().execute_sql(
            "INSERT INTO connectors "
            "(id, org_id, workspace_id, pipeline_name, name, type, subtype, "
            " capabilities, config_json, secrets_enc, is_default, is_read_only, "
            " is_active, created_by, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                connector_id, org_id, workspace_id, pipeline_name, name, type, subtype,
                caps_str, config_json, secrets_enc, is_default, is_read_only,
                True, created_by, now, now,
            ),
        )

        logger.info(
            "Created connector %s (%s/%s) in scope org=%s ws=%s pipeline=%s",
            name, type, subtype, org_id[:8], (workspace_id or "-")[:8], pipeline_name or "-",
        )

        return self.get(connector_id, include_secrets=False)

    def get(self, connector_id: str, *, include_secrets: bool = False) -> Connector:
        """Fetch a connector by ID. Raises ConnectorNotFound if missing."""
        row = self._db().query_one(
            "SELECT * FROM connectors WHERE id = ? AND is_active = TRUE",
            (connector_id,),
        )
        if not row:
            raise ConnectorNotFound(f"Connector not found: {connector_id}")
        return self._row_to_connector(row, include_secrets=include_secrets)

    def update(
        self,
        connector_id: str,
        *,
        name: str | None = None,
        capabilities: list[str] | None = None,
        config: dict[str, Any] | None = None,
        secrets: dict[str, Any] | None = None,
        is_default: bool | None = None,
        is_read_only: bool | None = None,
    ) -> Connector:
        """Partial update. Omitted fields preserve their current value.

        ``secrets`` is special: passing None leaves existing secrets
        untouched. Passing an empty dict clears them. Passing a
        populated dict fully replaces them.
        """
        existing = self.get(connector_id, include_secrets=False)

        if capabilities is not None:
            self._validate(
                type_=existing.type,
                subtype=existing.subtype,
                capabilities=capabilities,
            )

        # Build dynamic UPDATE clause
        fields: list[str] = []
        values: list[Any] = []

        if name is not None:
            fields.append("name = ?")
            values.append(name)
        if capabilities is not None:
            fields.append("capabilities = ?")
            values.append(",".join(capabilities))
        if config is not None:
            fields.append("config_json = ?")
            values.append(json.dumps(config))
        if secrets is not None:
            fields.append("secrets_enc = ?")
            values.append(self._encrypt_secrets(secrets))
        if is_default is not None:
            fields.append("is_default = ?")
            values.append(is_default)
        if is_read_only is not None:
            fields.append("is_read_only = ?")
            values.append(is_read_only)

        if not fields:
            return existing  # no-op

        fields.append("updated_at = ?")
        values.append(time.time())
        values.append(connector_id)

        self._db().execute_sql(
            f"UPDATE connectors SET {', '.join(fields)} WHERE id = ?",
            tuple(values),
        )

        logger.info("Updated connector %s (%d fields)", connector_id[:8], len(fields))

        return self.get(connector_id, include_secrets=False)

    def delete(self, connector_id: str) -> None:
        """Soft delete — sets is_active = FALSE. Secrets retained for audit."""
        self._db().execute_sql(
            "UPDATE connectors SET is_active = FALSE, updated_at = ? WHERE id = ?",
            (time.time(), connector_id),
        )
        logger.info("Soft-deleted connector %s", connector_id[:8])

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list(
        self,
        *,
        org_id: str,
        workspace_id: str | None = None,
        type_filter: str | None = None,
        subtype_filter: str | None = None,
        include_inactive: bool = False,
    ) -> list[Connector]:
        """List connectors in a scope.

        When ``workspace_id`` is given, returns workspace-wide AND
        org-wide rows (the caller sees everything their workspace can
        resolve). When only ``org_id`` is given, returns everything
        org-wide across every workspace.
        """
        conditions: list[str] = ["org_id = ?"]
        params: list[Any] = [org_id]

        if workspace_id is not None:
            conditions.append("(workspace_id = ? OR workspace_id IS NULL)")
            params.append(workspace_id)

        if type_filter:
            conditions.append("type = ?")
            params.append(type_filter)

        if subtype_filter:
            conditions.append("subtype = ?")
            params.append(subtype_filter)

        if not include_inactive:
            conditions.append("is_active = TRUE")

        sql = (
            "SELECT * FROM connectors "
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY created_at DESC"
        )

        rows = self._db().query_all(sql, tuple(params))
        return [self._row_to_connector(r, include_secrets=False) for r in rows]

    # ------------------------------------------------------------------
    # Resolution — the heart of the connector runtime
    # ------------------------------------------------------------------

    def resolve_by_name(
        self,
        *,
        org_id: str,
        name: str,
        workspace_id: str | None = None,
        pipeline_name: str | None = None,
        include_secrets: bool = True,
    ) -> Connector:
        """Resolve a connector by name, walking the scope chain.

        Resolution order:
          1. (org_id, workspace_id, pipeline_name, name)  — pipeline-specific
          2. (org_id, workspace_id, NULL, name)            — workspace-wide
          3. (org_id, NULL, NULL, name)                    — org-wide

        Raises ``ConnectorNotFound`` if nothing matches.
        """
        db = self._db()

        # 1. Pipeline-specific
        if workspace_id and pipeline_name:
            row = db.query_one(
                "SELECT * FROM connectors WHERE org_id = ? AND workspace_id = ? "
                "AND pipeline_name = ? AND name = ? AND is_active = TRUE",
                (org_id, workspace_id, pipeline_name, name),
            )
            if row:
                return self._row_to_connector(row, include_secrets=include_secrets)

        # 2. Workspace-wide
        if workspace_id:
            row = db.query_one(
                "SELECT * FROM connectors WHERE org_id = ? AND workspace_id = ? "
                "AND pipeline_name IS NULL AND name = ? AND is_active = TRUE",
                (org_id, workspace_id, name),
            )
            if row:
                return self._row_to_connector(row, include_secrets=include_secrets)

        # 3. Org-wide
        row = db.query_one(
            "SELECT * FROM connectors WHERE org_id = ? AND workspace_id IS NULL "
            "AND pipeline_name IS NULL AND name = ? AND is_active = TRUE",
            (org_id, name),
        )
        if row:
            return self._row_to_connector(row, include_secrets=include_secrets)

        raise ConnectorNotFound(
            f"Connector '{name}' not found in scope "
            f"(org={org_id[:8]}, workspace={(workspace_id or '-')[:8]}, "
            f"pipeline={pipeline_name or '-'})."
        )

    def resolve_default(
        self,
        *,
        org_id: str,
        type: str,
        subtype: str,
        workspace_id: str | None = None,
        pipeline_name: str | None = None,
        include_secrets: bool = True,
    ) -> Connector | None:
        """Find the default connector of a given type+subtype in scope.

        Used by legacy action shortcuts (``action: slack``) to
        auto-resolve to a default connector without an explicit
        ``uses:`` clause.

        Resolution order:
          1. Pipeline-specific default
          2. Workspace-wide default
          3. Org-wide default

        Returns None if no default exists — caller falls through to
        legacy env-var path.
        """
        db = self._db()

        # 1. Pipeline-specific default
        if workspace_id and pipeline_name:
            row = db.query_one(
                "SELECT * FROM connectors WHERE org_id = ? AND workspace_id = ? "
                "AND pipeline_name = ? AND type = ? AND subtype = ? "
                "AND is_default = TRUE AND is_active = TRUE LIMIT 1",
                (org_id, workspace_id, pipeline_name, type, subtype),
            )
            if row:
                return self._row_to_connector(row, include_secrets=include_secrets)

        # 2. Workspace-wide default
        if workspace_id:
            row = db.query_one(
                "SELECT * FROM connectors WHERE org_id = ? AND workspace_id = ? "
                "AND pipeline_name IS NULL AND type = ? AND subtype = ? "
                "AND is_default = TRUE AND is_active = TRUE LIMIT 1",
                (org_id, workspace_id, type, subtype),
            )
            if row:
                return self._row_to_connector(row, include_secrets=include_secrets)

        # 3. Org-wide default
        row = db.query_one(
            "SELECT * FROM connectors WHERE org_id = ? AND workspace_id IS NULL "
            "AND pipeline_name IS NULL AND type = ? AND subtype = ? "
            "AND is_default = TRUE AND is_active = TRUE LIMIT 1",
            (org_id, type, subtype),
        )
        if row:
            return self._row_to_connector(row, include_secrets=include_secrets)

        return None
