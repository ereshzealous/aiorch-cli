# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""SQLite storage backend for CLI mode — lightweight, zero-infra persistence.

All CLI run history, step traces, cost tracking, and cache persist to
~/.aiorch/history.db. No Postgres, no Docker, no setup.

Used automatically when DATABASE_URL is not set.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from aiorch.storage import Store

logger = logging.getLogger("aiorch.storage.sqlite")

DEFAULT_DB_PATH = Path.home() / ".aiorch" / "history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    file TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    started_at REAL NOT NULL,
    finished_at REAL,
    total_cost REAL DEFAULT 0,
    org_id TEXT,
    workspace_id TEXT,
    pipeline_version_id TEXT,
    triggered_by TEXT,
    log_level TEXT,
    heartbeat_at REAL,
    claim_token TEXT
);

CREATE TABLE IF NOT EXISTS step_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    step_name TEXT NOT NULL,
    primitive TEXT,
    status TEXT NOT NULL,
    started_at REAL,
    finished_at REAL,
    duration_ms REAL DEFAULT 0,
    cost REAL DEFAULT 0,
    output_preview TEXT,
    error TEXT,
    error_type TEXT,
    traceback TEXT,
    model TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    provider_name TEXT,
    step_output_json TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT,
    response_json TEXT,
    cost REAL DEFAULT 0,
    created_at REAL,
    hit_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_step_runs_run_id ON step_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);

-- Artifacts (CLI mode): content-addressed file store backed by
-- ~/.aiorch/artifacts/<sha[:2]>/<sha>. Metadata lives here, bytes live
-- on disk. Inputs dedup by sha256; outputs never dedup.
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    sha256 TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'input' CHECK (role IN ('input', 'output')),
    pipeline_name TEXT,
    run_id INTEGER,
    metadata_json TEXT,
    created_at REAL NOT NULL,
    deleted_at REAL
);

-- Dedup inputs by sha256. Outputs are unique per row regardless of
-- content, so they are excluded via the partial index.
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_sha256_input
    ON artifacts(sha256) WHERE role = 'input' AND deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_artifacts_created_at ON artifacts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_artifacts_role ON artifacts(role);

-- Run ↔ artifact lineage. One row per (run, binding_name, role).
CREATE TABLE IF NOT EXISTS run_artifacts (
    run_id INTEGER NOT NULL,
    artifact_id TEXT NOT NULL,
    binding_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('input', 'output')),
    created_at REAL NOT NULL,
    PRIMARY KEY (run_id, binding_name, role),
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (artifact_id) REFERENCES artifacts(id)
);

CREATE INDEX IF NOT EXISTS idx_run_artifacts_artifact ON run_artifacts(artifact_id, role);

-- Performance indexes (mirrors Postgres migration q2r3s4t5u6v7)
CREATE INDEX IF NOT EXISTS idx_runs_ws_started_desc
    ON runs(workspace_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_ws_status_started_desc
    ON runs(workspace_id, status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_pending_started_partial
    ON runs(started_at ASC) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_runs_running_heartbeat_partial
    ON runs(heartbeat_at) WHERE status = 'running' AND finished_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_run_artifacts_artifact_created_desc
    ON run_artifacts(artifact_id, created_at DESC);
"""


class SQLiteStore(Store):
    """SQLite-backed store for CLI mode. Persists to a single .db file."""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else DEFAULT_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("SQLite store: %s", self._path)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Generic query interface (for compatibility) ---

    def query_one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        sql = sql.replace("?", "?")  # SQLite uses ? natively
        try:
            row = self._conn.execute(sql, params).fetchone()
            return dict(row) if row else None
        except sqlite3.OperationalError:
            return None

    def query_all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        try:
            rows = self._conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def execute_sql(self, sql: str, params: tuple = ()) -> None:
        try:
            self._conn.execute(sql, params)
            self._conn.commit()
        except sqlite3.OperationalError:
            pass

    def execute_many_transactional(self, statements: list[tuple[str, tuple]]) -> None:
        """Execute multiple statements in a single SQLite transaction."""
        try:
            self._conn.execute("BEGIN")
            for sql, params in statements:
                self._conn.execute(sql, params)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # --- Runs ---

    def start_run(self, name: str, file: str | None = None, **kwargs) -> int:
        # Honor the caller's `status` argument so SQLite matches the
        # Postgres store's contract — previously the status column was
        # hard-coded to "running" regardless of what the caller passed,
        # which drifted from platform semantics (Postgres defaults to
        # "pending" so the executor can claim rows via the normal
        # pending → running transition). The default stays "running"
        # here because SQLite is the CLI-mode store and `aiorch run`
        # executes pipelines in-process without a separate executor —
        # there's nothing to claim a pending row, so "running" is the
        # right default for the CLI's direct-execution shape.
        status = kwargs.get("status", "running")
        cur = self._conn.execute(
            "INSERT INTO runs (name, file, status, started_at, org_id, workspace_id, "
            "pipeline_version_id, triggered_by, log_level) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, file, status, time.time(),
             kwargs.get("org_id"), kwargs.get("workspace_id"),
             kwargs.get("pipeline_version_id"), kwargs.get("triggered_by"),
             kwargs.get("log_level")),
        )
        self._conn.commit()
        return cur.lastrowid

    def finish_run(
        self, run_id: int, status: str = "success", total_cost: float = 0,
        *, claim_token: str | None = None,
    ) -> None:
        if claim_token is None:
            self._conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, total_cost = ? WHERE id = ?",
                (status, time.time(), total_cost, run_id),
            )
        else:
            # Token guard — see executor CRITICAL #2. Protects against
            # stale-executor writes landing after a peer has reclaimed.
            self._conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, total_cost = ? "
                "WHERE id = ? AND claim_token = ?",
                (status, time.time(), total_cost, run_id, claim_token),
            )
        self._conn.commit()

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def claim_pending_run(self) -> dict[str, Any] | None:
        # SQLite is single-process (CLI mode). No inter-process contention,
        # no SKIP LOCKED support. Serial claim inside a single transaction
        # is sufficient. We still stamp a claim_token on the row so the
        # Store interface contract stays the same across backends —
        # reclaim-vs-finish protection is genuinely only needed on
        # multi-process Postgres, but shipping the field on SQLite too
        # avoids divergent code paths in the executor.
        import uuid
        row = self._conn.execute(
            "SELECT * FROM runs WHERE status = 'pending' "
            "ORDER BY started_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        run = dict(row)
        now = time.time()
        token = str(uuid.uuid4())
        self._conn.execute(
            "UPDATE runs SET status = 'running', heartbeat_at = ?, claim_token = ? WHERE id = ?",
            (now, token, run["id"]),
        )
        self._conn.commit()
        run["status"] = "running"
        run["heartbeat_at"] = now
        run["claim_token"] = token
        return run

    def update_heartbeat(self, run_id: int, claim_token: str | None = None) -> None:
        if claim_token is None:
            self._conn.execute(
                "UPDATE runs SET heartbeat_at = ? WHERE id = ? AND status = 'running'",
                (time.time(), run_id),
            )
        else:
            self._conn.execute(
                "UPDATE runs SET heartbeat_at = ? "
                "WHERE id = ? AND status = 'running' AND claim_token = ?",
                (time.time(), run_id, claim_token),
            )
        self._conn.commit()

    # --- Steps ---

    def log_step(self, run_id: int, step_name: str, primitive: str, status: str,
                 started_at: float, finished_at: float, cost: float = 0,
                 output_preview: str | None = None, error: str | None = None,
                 error_type: str | None = None, model: str | None = None,
                 prompt_tokens: int = 0, completion_tokens: int = 0,
                 provider_name: str | None = None,
                 traceback: str | None = None) -> None:
        duration_ms = (finished_at - started_at) * 1000
        self._conn.execute(
            "INSERT INTO step_runs (run_id, step_name, primitive, status, started_at, "
            "finished_at, duration_ms, cost, output_preview, error, error_type, traceback, "
            "model, prompt_tokens, completion_tokens, provider_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, step_name, primitive, status, started_at, finished_at,
             duration_ms, cost, output_preview, error, error_type, traceback,
             model, prompt_tokens, completion_tokens, provider_name),
        )
        self._conn.commit()

    def get_run_steps(self, run_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM step_runs WHERE run_id = ? ORDER BY started_at",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_step_cost(self, run_id: int, step_name: str, cost: float) -> None:
        self._conn.execute(
            "UPDATE step_runs SET cost = ? WHERE run_id = ? AND step_name = ?",
            (cost, run_id, step_name),
        )
        self._conn.commit()

    def save_step_output(self, run_id: int, step_name: str, output_json: str) -> None:
        self._conn.execute(
            "UPDATE step_runs SET step_output_json = ? WHERE run_id = ? AND step_name = ?",
            (output_json, run_id, step_name),
        )
        self._conn.commit()

    def get_step_outputs(self, run_id: int) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT step_name, step_output_json FROM step_runs "
            "WHERE run_id = ? AND status = 'success' AND step_output_json IS NOT NULL",
            (run_id,),
        ).fetchall()
        result = {}
        for r in rows:
            try:
                result[r["step_name"]] = json.loads(r["step_output_json"])
            except Exception:
                pass
        return result

    # --- Cache ---

    def cache_get(self, key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT response_json, cost FROM llm_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if not row:
            return None
        self._conn.execute(
            "UPDATE llm_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
            (key,),
        )
        self._conn.commit()
        return {"response": json.loads(row["response_json"]), "cost": row["cost"]}

    def cache_put(self, key: str, model: str, response: Any, cost: float = 0) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, model, response_json, cost, created_at, hit_count) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (key, model, json.dumps(response), cost, time.time()),
        )
        self._conn.commit()

    def cache_stats(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT COUNT(*) as entries, COALESCE(SUM(hit_count), 0) as total_hits, "
            "COALESCE(SUM(cost), 0) as saved_cost FROM llm_cache"
        ).fetchone()
        return dict(row) if row else {"entries": 0, "total_hits": 0, "saved_cost": 0}

    # --- Dashboard ---

    def get_dashboard_stats(self) -> dict[str, Any]:
        now = time.time()
        day_ago = now - 86400
        week_ago = now - 604800

        total = self._conn.execute("SELECT COUNT(*) as c, COALESCE(SUM(total_cost), 0) as cost FROM runs").fetchone()
        today = self._conn.execute("SELECT COUNT(*) as c, COALESCE(SUM(total_cost), 0) as cost FROM runs WHERE started_at > ?", (day_ago,)).fetchone()
        week = self._conn.execute("SELECT COUNT(*) as c, COALESCE(SUM(total_cost), 0) as cost FROM runs WHERE started_at > ?", (week_ago,)).fetchone()

        success = self._conn.execute("SELECT COUNT(*) as c FROM runs WHERE status = 'success'").fetchone()
        total_count = total["c"] if total else 0
        success_count = success["c"] if success else 0
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0

        return {
            "today_runs": today["c"] if today else 0,
            "today_cost": today["cost"] if today else 0,
            "week_runs": week["c"] if week else 0,
            "week_cost": week["cost"] if week else 0,
            "total_runs": total_count,
            "total_cost": total["cost"] if total else 0,
            "success_rate": round(success_rate, 1),
            "top_pipelines": [],
            "recent_failures": [],
        }
