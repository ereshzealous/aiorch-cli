# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Storage layer — abstract interface with PostgreSQL implementation.

The Store interface defines all storage operations. PostgresStore is the
sole backend. DATABASE_URL env var (loaded from .env) configures the connection.

Usage:
    from aiorch.storage import init_storage, start_run, log_step, get_runs
    init_storage()  # Reads DATABASE_URL from environment
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("aiorch.storage")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    id: int = 0
    name: str = ""
    file: str | None = None
    status: str = "running"
    started_at: float = 0
    finished_at: float | None = None
    total_cost: float = 0
    steps: list[StepRecord] = field(default_factory=list)


@dataclass
class StepRecord:
    step_name: str = ""
    primitive: str = ""
    status: str = "running"
    started_at: float = 0
    finished_at: float | None = None
    duration_ms: float = 0
    cost: float = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Abstract Store interface
# ---------------------------------------------------------------------------

class RunStore(ABC):
    """Run lifecycle tracking."""

    @abstractmethod
    def start_run(self, name: str, file: str | None = None) -> int: ...

    @abstractmethod
    def finish_run(
        self, run_id: int, status: str = "success", total_cost: float = 0,
        *, claim_token: str | None = None,
    ) -> None:
        """Mark a run terminal. When ``claim_token`` is provided, the update
        only lands if the row's stored token still matches — protection
        against the reclaim-vs-finish race where a stuck executor resumes
        after its run has been reassigned to a peer. Callers that aren't
        the claiming executor (legacy paths, admin overrides) pass None
        and skip the guard."""

    @abstractmethod
    def get_run(self, run_id: int) -> dict[str, Any] | None: ...

    @abstractmethod
    def get_runs(self, limit: int = 20) -> list[dict[str, Any]]: ...

    def claim_pending_run(self) -> dict[str, Any] | None:
        """Atomically claim the oldest pending run and flip it to 'running'.

        Backends that support it (Postgres) must use FOR UPDATE SKIP LOCKED
        so multiple executors can claim lock-free in parallel. Single-process
        backends (SQLite, in-memory) may use a simpler serial claim.

        Must generate a fresh ``claim_token`` on the row and return it in
        the result dict so the executor can pass it to subsequent heartbeat
        and finish_run calls. Returns the claimed run row, or None if the
        queue is empty.
        """
        raise NotImplementedError

    def update_heartbeat(self, run_id: int, claim_token: str | None = None) -> None:
        """Refresh the heartbeat timestamp for a running run.

        Called periodically by the executor to signal that the run is still
        alive. The stale-run reclaimer checks heartbeat_at instead of
        started_at so long-running pipelines are not wrongly reset. When
        ``claim_token`` is provided, the update only lands if the row's
        stored token still matches — so a reclaimed run's original executor
        cannot silently keep the heartbeat alive for a peer's claim.
        """
        raise NotImplementedError


class StepStore(ABC):
    """Step execution logging."""

    @abstractmethod
    def log_step(
        self, run_id: int, step_name: str, primitive: str, status: str,
        started_at: float, finished_at: float, cost: float = 0,
        output_preview: str | None = None, error: str | None = None,
        error_type: str | None = None, model: str | None = None,
        prompt_tokens: int = 0, completion_tokens: int = 0,
        provider_name: str | None = None,
        traceback: str | None = None,
    ) -> None: ...

    @abstractmethod
    def get_run_steps(self, run_id: int) -> list[dict[str, Any]]: ...

    @abstractmethod
    def update_step_cost(self, run_id: int, step_name: str, cost: float) -> None: ...

    def save_step_output(self, run_id: int, step_name: str, output_json: str) -> None:
        pass

    def get_step_outputs(self, run_id: int) -> dict[str, Any]:
        return {}


class CacheStore(ABC):
    """LLM response caching."""

    @abstractmethod
    def cache_get(self, key: str) -> dict | None: ...

    @abstractmethod
    def cache_put(self, key: str, model: str, response: Any, cost: float = 0) -> None: ...

    @abstractmethod
    def cache_stats(self) -> dict[str, Any]: ...


class DashboardStore(ABC):
    """Aggregate statistics for the dashboard."""

    @abstractmethod
    def get_dashboard_stats(self) -> dict[str, Any]: ...


class Store(RunStore, StepStore, CacheStore, DashboardStore):
    """Full storage interface — combines all store protocols."""

    def query_one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        raise NotImplementedError

    def query_all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        raise NotImplementedError

    def execute_sql(self, sql: str, params: tuple = ()) -> None:
        raise NotImplementedError

    def execute_many_transactional(self, statements: list[tuple[str, tuple]]) -> None:
        """Execute multiple SQL statements inside a single transaction.

        All statements succeed or all are rolled back. Used by multi-step
        onboarding flows (registration, invitation accept, OAuth user creation)
        to prevent orphaned records on mid-flow failures.

        Default implementation falls back to sequential execute_sql calls
        (no atomicity guarantee — backends should override).
        """
        for sql, params in statements:
            self.execute_sql(sql, params)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_store: Store | None = None


def get_store() -> Store:
    """Get the current store instance. Requires init_storage() to have been called."""
    if _store is None:
        raise RuntimeError(
            "Storage not initialized. Call init_storage() first. "
            "Ensure DATABASE_URL is set in .env"
        )
    return _store


def set_store(store: Store) -> None:
    """Replace the store singleton (used by tests)."""
    global _store
    _store = store


def init_storage() -> None:
    """Initialize storage backend.

    Resolution:
      1. DATABASE_URL env var set → PostgresStore (Platform mode)
      2. No DATABASE_URL           → SQLiteStore at ~/.aiorch/history.db (CLI mode)
    """
    global _store
    if _store is not None:
        return

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        from aiorch.core.config import get_config
        cfg = get_config()
        try:
            from aiorch.storage.postgres import PostgresStore
        except ImportError as e:
            raise RuntimeError(
                "DATABASE_URL is set but the Postgres backend "
                "(aiorch.storage.postgres) is not available. Install "
                "aiorch-platform alongside aiorch-cli, or unset "
                "DATABASE_URL to run in CLI-only (SQLite) mode."
            ) from e
        _store = PostgresStore(
            url=database_url,
            pool_size=cfg.storage.pool_size,
        )
        logger.info("Storage: PostgreSQL (%s...)", database_url[:40])
    else:
        from aiorch.storage.sqlite import SQLiteStore
        _store = SQLiteStore()
        logger.info("Storage: SQLite (~/.aiorch/history.db)")


# ---------------------------------------------------------------------------
# Convenience functions — delegate to singleton
# ---------------------------------------------------------------------------

def start_run(name: str, file: str | None = None, **kwargs) -> int:
    return get_store().start_run(name, file, **kwargs)

def finish_run(
    run_id: int, status: str = "success", total_cost: float = 0,
    *, claim_token: str | None = None,
) -> None:
    get_store().finish_run(run_id, status, total_cost, claim_token=claim_token)

def log_step(run_id: int, step_name: str, primitive: str, status: str,
             started_at: float, finished_at: float, cost: float = 0,
             output_preview: str | None = None, error: str | None = None,
             error_type: str | None = None, model: str | None = None,
             prompt_tokens: int = 0, completion_tokens: int = 0,
             provider_name: str | None = None,
             traceback: str | None = None) -> None:
    get_store().log_step(run_id, step_name, primitive, status, started_at, finished_at,
                         cost, output_preview, error, error_type, model, prompt_tokens, completion_tokens,
                         provider_name, traceback=traceback)

def get_run(run_id: int) -> dict[str, Any] | None:
    return get_store().get_run(run_id)

def get_runs(limit: int = 20) -> list[dict[str, Any]]:
    return get_store().get_runs(limit)

def get_run_steps(run_id: int) -> list[dict[str, Any]]:
    return get_store().get_run_steps(run_id)

def cache_get(key: str) -> dict | None:
    return get_store().cache_get(key)

def cache_put(key: str, model: str, response: Any, cost: float = 0) -> None:
    get_store().cache_put(key, model, response, cost)

def cache_stats() -> dict[str, Any]:
    return get_store().cache_stats()

def get_dashboard_stats() -> dict[str, Any]:
    return get_store().get_dashboard_stats()

def save_step_output(run_id: int, step_name: str, output_json: str) -> None:
    get_store().save_step_output(run_id, step_name, output_json)

def get_step_outputs(run_id: int) -> dict[str, Any]:
    return get_store().get_step_outputs(run_id)


def cache_key(prompt: str, model: str, system: str | None = None, temperature: float | None = None) -> str:
    """Compute a deterministic cache key from prompt parameters."""
    parts = f"{model}|{system or ''}|{temperature or ''}|{prompt}"
    return hashlib.sha256(parts.encode()).hexdigest()
