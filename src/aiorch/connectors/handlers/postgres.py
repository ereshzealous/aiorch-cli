# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Postgres database connector handler.

Implements the canonical database operations (query / insert / update /
upsert) via asyncpg. Per Q2 of the connectors design doc, every
operation opens a fresh connection, executes, and closes — no pool.
Rotation is automatic (each call re-reads credentials from the
connector row), secret lifecycle is trivial (no stale connections
held anywhere), and correctness is simple to reason about.

Injection safety:
  - `query` uses asyncpg's $1/$2 placeholder binding. The parser
    already rejects any `sql:` field containing Jinja2 markers, so
    the SQL string is frozen and only parameters flow through user
    input.
  - `insert` / `update` / `upsert` generate SQL from caller-supplied
    table + column names. These are validated against the
    `_IDENTIFIER_RE` regex before interpolation. Anything outside
    `[a-zA-Z_][a-zA-Z0-9_]*` is rejected — no quotes, no whitespace,
    no schema separators, no SQL keywords.

Timeouts:
  - `SET statement_timeout = N` issued on each connection. Gives a
    clean asyncpg.exceptions.QueryCanceledError that we map to
    ConnectorTimeout for the error surface.
  - asyncio-level timeout at connection open via asyncpg's
    ``timeout=`` kwarg as a safety net.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from aiorch.connectors import (
    ConnectorAuthError,
    ConnectorTimeout,
    NotSupportedError,
)
from aiorch.connectors.handlers import ConnectorHandler, register_handler

logger = logging.getLogger("aiorch.connectors.handlers.postgres")


# ---------------------------------------------------------------------------
# Identifier validation
# ---------------------------------------------------------------------------

_SIMPLE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_QUALIFIED_IDENTIFIER_RE = re.compile(
    r"^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$"
)
# Kept as a legacy alias for any external callers importing this name.
_IDENTIFIER_RE = _SIMPLE_IDENTIFIER_RE


def _validate_identifier(name: str, kind: str) -> None:
    """Validate a SQL identifier (table or column name).

    Accepts:
    - Simple: ``users``, ``order_items``, ``_tmp``
    - Schema-qualified: ``public.users``, ``app_schema.events``

    Rejects quoted identifiers, three-part names, whitespace, and
    anything that could be an injection attempt. Column names should
    always be simple (no dots).
    """
    if not isinstance(name, str):
        raise ValueError(
            f"Invalid {kind} name: {name!r}. Must be a string."
        )
    if _SIMPLE_IDENTIFIER_RE.match(name):
        return
    # Only tables may be schema-qualified; columns must be simple.
    if kind == "table" and _QUALIFIED_IDENTIFIER_RE.match(name):
        return
    raise ValueError(
        f"Invalid {kind} name: {name!r}. "
        f"Tables accept simple (users) or schema-qualified (public.users) identifiers. "
        f"Columns must be simple. No quotes, whitespace, or three-part names."
    )


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _map_asyncpg_error(exc: Exception) -> Exception:
    """Translate asyncpg exceptions into friendly messages.

    Returns a new exception with a cleaner message; callers can still
    catch the original type via `except Exception` in the dispatcher.
    We preserve the original class where it adds value (auth, timeout)
    and fall through for everything else.
    """
    try:
        import asyncpg.exceptions as pgx  # type: ignore
    except ImportError:
        return exc

    if isinstance(exc, pgx.InvalidAuthorizationSpecificationError):
        return ConnectorAuthError(
            "Postgres authentication failed. Check the connector's user + password."
        )
    if isinstance(exc, pgx.InvalidCatalogNameError):
        return ValueError(
            f"Postgres database does not exist: {exc}"
        )
    if isinstance(exc, pgx.QueryCanceledError):
        return ConnectorTimeout(
            f"Postgres query exceeded statement_timeout: {exc}"
        )
    if isinstance(exc, pgx.UndefinedTableError):
        return ValueError(f"Postgres table does not exist: {exc}")
    if isinstance(exc, pgx.UndefinedColumnError):
        return ValueError(f"Postgres column does not exist: {exc}")
    if isinstance(exc, pgx.InsufficientPrivilegeError):
        return PermissionError(
            f"Postgres user does not have permission for this operation: {exc}"
        )

    return exc


# ---------------------------------------------------------------------------
# Connection builder
# ---------------------------------------------------------------------------


async def _open_connection(connector_config: dict) -> Any:
    """Open a fresh asyncpg connection using the connector config.

    Raises ConnectorAuthError / ConnectorTimeout on known failures;
    other errors fall through unchanged so the dispatch wrapper
    captures them verbatim.
    """
    import asyncpg  # type: ignore

    # DSN takes priority — managed Postgres providers (Heroku, Render,
    # Supabase, Railway, Neon, RDS) all give a single connection string.
    # When set, individual host/port/database/user/password fields are
    # ignored. SSL is still applied from ssl_mode + ssl_root_cert.
    dsn = connector_config.get("dsn") or connector_config.get("connection_string")
    if dsn:
        connect_kwargs: dict = {"dsn": dsn}
    else:
        connect_kwargs = {
            "host": connector_config.get("host", "localhost"),
            "port": int(connector_config.get("port", 5432)),
            "database": connector_config.get("database") or connector_config.get("db"),
            "user": connector_config.get("user") or connector_config.get("username"),
            "password": connector_config.get("password"),
        }

    # SSL mode: explicit opt-in only. 'disable' / unset / 'prefer' all
    # fall through to asyncpg's default (no SSL forced), which works
    # against local docker-compose Postgres and behind private networks.
    # 'require' uses TLS without cert verification.
    # 'verify-ca' / 'verify-full' build a real SSLContext that actually
    # validates the server certificate chain.
    ssl_mode = connector_config.get("ssl_mode", "disable")
    if ssl_mode == "require":
        connect_kwargs["ssl"] = True
    elif ssl_mode in ("verify-ca", "verify-full"):
        import ssl as _ssl

        ctx = _ssl.create_default_context()
        ssl_root_cert = connector_config.get("ssl_root_cert")
        if ssl_root_cert:
            # Loaded from the secrets bucket as a PEM string. asyncpg's
            # ssl context builder doesn't take a PEM blob directly, so
            # we materialize it to a tempfile, load it, and unlink.
            import os
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pem", delete=False
            ) as f:
                f.write(ssl_root_cert)
                ca_path = f.name
            try:
                ctx.load_verify_locations(ca_path)
            finally:
                os.unlink(ca_path)
        else:
            ctx.load_default_certs()
        if ssl_mode == "verify-ca":
            # Verify chain but skip hostname matching
            ctx.check_hostname = False
        connect_kwargs["ssl"] = ctx

    app_name = connector_config.get("application_name", "aiorch-connector")

    connect_timeout = int(connector_config.get("connect_timeout_seconds", 10))

    try:
        conn = await asyncpg.connect(
            **connect_kwargs,
            server_settings={"application_name": app_name},
            timeout=connect_timeout,
        )
    except Exception as exc:
        raise _map_asyncpg_error(exc)

    # Apply statement_timeout so long queries get a clean error
    # rather than hanging the pipeline
    statement_timeout_ms = int(connector_config.get("statement_timeout_ms", 30000))
    await conn.execute(f"SET statement_timeout = {statement_timeout_ms}")

    return conn


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class PostgresHandler(ConnectorHandler):
    """Canonical database handler for Postgres via asyncpg.

    All operations open a fresh connection, execute, and close. No
    pool. Secret rotation is automatic — the next operation reads
    the current ``connector_config``.
    """

    type = "database"
    subtype = "postgres"
    supported_operations = {"query", "insert", "update", "upsert"}

    @classmethod
    def get_client_info(cls) -> dict | None:
        try:
            import asyncpg  # type: ignore
            version = getattr(asyncpg, "__version__", "unknown")
        except ImportError:
            return None
        return {
            "client_library": "asyncpg",
            "client_version": version,
            "docs_url": "https://magicstack.github.io/asyncpg/current/api/index.html",
        }

    async def execute(
        self,
        *,
        operation: str,
        params: dict,
        connector_config: dict,
        context: dict,
    ) -> Any:
        self._check_supported(operation)

        conn = await _open_connection(connector_config)
        try:
            if operation == "query":
                return await self._query(conn, params, connector_config)
            if operation == "insert":
                return await self._insert(conn, params)
            if operation == "update":
                return await self._update(conn, params)
            if operation == "upsert":
                return await self._upsert(conn, params)
            raise NotSupportedError(f"PostgresHandler has no implementation for '{operation}'")
        except Exception as exc:
            mapped = _map_asyncpg_error(exc)
            if mapped is exc:
                raise
            raise mapped from exc
        finally:
            try:
                await conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # query — SELECT with parameterized placeholders
    # ------------------------------------------------------------------

    async def _query(self, conn, params: dict, connector_config: dict) -> list[dict]:
        sql = params.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("query operation requires a non-empty 'sql' string")

        query_params = params.get("params") or []
        if not isinstance(query_params, list):
            raise ValueError("'params' must be a list of values for $1/$2 placeholders")

        max_rows = int(
            params.get("max_rows")
            or connector_config.get("max_rows_default", 1000)
        )

        rows = await conn.fetch(sql, *query_params)
        result = [dict(r) for r in rows]
        if len(result) > max_rows:
            logger.info(
                "Postgres query returned %d rows; truncating to max_rows=%d",
                len(result), max_rows,
            )
            result = result[:max_rows]
        return result

    # ------------------------------------------------------------------
    # insert — rows is a list of dicts with uniform keys
    # ------------------------------------------------------------------

    async def _insert(self, conn, params: dict) -> dict:
        table = params.get("table")
        rows = params.get("rows") or []

        if not table or not isinstance(table, str):
            raise ValueError("insert operation requires a 'table' string")
        if not isinstance(rows, list) or not rows:
            raise ValueError("insert operation requires a non-empty 'rows' list")

        _validate_identifier(table, "table")

        # All rows must have the same column set (use first row's keys)
        columns = list(rows[0].keys())
        for col in columns:
            _validate_identifier(col, "column")

        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        cols_sql = ", ".join(columns)
        sql = f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders})"

        rows_affected = 0
        async with conn.transaction():
            for row in rows:
                values = [row.get(col) for col in columns]
                await conn.execute(sql, *values)
                rows_affected += 1

        return {
            "rows_affected": rows_affected,
            "table": table,
        }

    # ------------------------------------------------------------------
    # update — SET fields from `set:`, WHERE clause from `where:`
    # ------------------------------------------------------------------

    async def _update(self, conn, params: dict) -> dict:
        table = params.get("table")
        set_fields = params.get("set") or {}
        where_fields = params.get("where") or {}

        if not table or not isinstance(table, str):
            raise ValueError("update operation requires a 'table' string")
        if not isinstance(set_fields, dict) or not set_fields:
            raise ValueError("update operation requires a non-empty 'set' dict")
        if not isinstance(where_fields, dict) or not where_fields:
            raise ValueError(
                "update operation requires a non-empty 'where' dict — "
                "refusing to issue an unbounded UPDATE"
            )

        _validate_identifier(table, "table")
        for col in set_fields:
            _validate_identifier(col, "column")
        for col in where_fields:
            _validate_identifier(col, "column")

        set_clause_parts = []
        where_clause_parts = []
        values: list[Any] = []
        idx = 1

        for col, val in set_fields.items():
            set_clause_parts.append(f"{col} = ${idx}")
            values.append(val)
            idx += 1

        for col, val in where_fields.items():
            where_clause_parts.append(f"{col} = ${idx}")
            values.append(val)
            idx += 1

        sql = (
            f"UPDATE {table} SET {', '.join(set_clause_parts)} "
            f"WHERE {' AND '.join(where_clause_parts)}"
        )

        status = await conn.execute(sql, *values)
        # status looks like 'UPDATE 5'
        rows_affected = int(status.split()[-1]) if status.startswith("UPDATE") else 0

        return {
            "rows_affected": rows_affected,
            "table": table,
        }

    # ------------------------------------------------------------------
    # upsert — INSERT ... ON CONFLICT (keys) DO UPDATE SET ...
    # ------------------------------------------------------------------

    async def _upsert(self, conn, params: dict) -> dict:
        table = params.get("table")
        rows = params.get("rows") or []
        conflict_key = params.get("conflict_key") or []

        if not table or not isinstance(table, str):
            raise ValueError("upsert operation requires a 'table' string")
        if not isinstance(rows, list) or not rows:
            raise ValueError("upsert operation requires a non-empty 'rows' list")
        if not isinstance(conflict_key, list) or not conflict_key:
            raise ValueError(
                "upsert operation requires a non-empty 'conflict_key' list "
                "identifying the columns that form the uniqueness constraint"
            )

        _validate_identifier(table, "table")
        columns = list(rows[0].keys())
        for col in columns:
            _validate_identifier(col, "column")
        for col in conflict_key:
            _validate_identifier(col, "column")
            if col not in columns:
                raise ValueError(
                    f"conflict_key column '{col}' must also be in the row data"
                )

        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        cols_sql = ", ".join(columns)
        conflict_sql = ", ".join(conflict_key)

        # Update every non-key column on conflict
        update_cols = [c for c in columns if c not in conflict_key]
        if update_cols:
            update_clause = ", ".join(
                f"{c} = EXCLUDED.{c}" for c in update_cols
            )
            sql = (
                f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_sql}) DO UPDATE SET {update_clause}"
            )
        else:
            # All columns are part of the conflict key — no-op on conflict
            sql = (
                f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_sql}) DO NOTHING"
            )

        rows_affected = 0
        async with conn.transaction():
            for row in rows:
                values = [row.get(col) for col in columns]
                await conn.execute(sql, *values)
                rows_affected += 1

        return {
            "rows_affected": rows_affected,
            "table": table,
        }


# Register on import so `get_handler("database", "postgres")` finds it
register_handler(PostgresHandler())
