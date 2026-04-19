# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""In-memory storage backend for CLI mode — no database, no persistence.

All data lives in Python dicts for the duration of the process.
Used when DATABASE_URL is not set (CLI run, validate, plan, etc.).
"""

from __future__ import annotations

import json
import time
from typing import Any

from aiorch.storage import Store


class MemoryStore(Store):
    """Lightweight in-memory store for CLI mode."""

    def __init__(self):
        self._runs: list[dict] = []
        self._steps: list[dict] = []
        self._cache: dict[str, dict] = {}
        # Start IDs high to avoid colliding with existing log files from server mode
        self._next_run_id = int(time.time()) % 1_000_000

    # --- Runs ---

    def start_run(self, name: str, file: str | None = None, **kwargs) -> int:
        # Honor the caller's `status` argument to match Postgres store
        # semantics — the previous implementation hard-coded "running"
        # regardless of what was passed, which silently broke tests that
        # expect a newly-created run to land as "pending" and transition
        # via claim. Default stays "running" for mock/CLI callers that
        # don't have a separate executor loop.
        self._next_run_id += 1
        self._runs.append({
            "id": self._next_run_id,
            "name": name,
            "file": file,
            "status": kwargs.get("status", "running"),
            "started_at": time.time(),
            "finished_at": None,
            "total_cost": 0,
            "org_id": kwargs.get("org_id"),
            "workspace_id": kwargs.get("workspace_id"),
            "pipeline_version_id": kwargs.get("pipeline_version_id"),
            "triggered_by": kwargs.get("triggered_by"),
            "log_level": kwargs.get("log_level"),
        })
        return self._next_run_id

    def finish_run(
        self, run_id: int, status: str = "success", total_cost: float = 0,
        *, claim_token: str | None = None,
    ) -> None:
        for r in self._runs:
            if r["id"] == run_id:
                if claim_token is not None and r.get("claim_token") != claim_token:
                    # Token mismatch — the row was reclaimed since we
                    # acquired it. Silently drop (same semantics as the
                    # SQL backends' zero-row UPDATE).
                    return
                r["status"] = status
                r["finished_at"] = time.time()
                r["total_cost"] = total_cost
                break

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        for r in self._runs:
            if r["id"] == run_id:
                return dict(r)
        return None

    def get_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        runs = sorted(self._runs, key=lambda r: r.get("started_at", 0), reverse=True)
        return [dict(r) for r in runs[:limit]]

    def claim_pending_run(self) -> dict[str, Any] | None:
        import uuid
        pending = [r for r in self._runs if r.get("status") == "pending"]
        if not pending:
            return None
        pending.sort(key=lambda r: r.get("started_at", 0))
        run = pending[0]
        run["status"] = "running"
        run["claim_token"] = str(uuid.uuid4())
        return dict(run)

    def update_heartbeat(self, run_id: int, claim_token: str | None = None) -> None:
        for r in self._runs:
            if r["id"] == run_id and r.get("status") == "running":
                if claim_token is not None and r.get("claim_token") != claim_token:
                    return
                r["heartbeat_at"] = time.time()
                return

    # --- Steps ---

    def log_step(self, run_id: int, step_name: str, primitive: str, status: str,
                 started_at: float, finished_at: float, cost: float = 0,
                 output_preview: str | None = None, error: str | None = None,
                 error_type: str | None = None, model: str | None = None,
                 prompt_tokens: int = 0, completion_tokens: int = 0,
                 provider_name: str | None = None,
                 traceback: str | None = None) -> None:
        self._steps.append({
            "run_id": run_id, "step_name": step_name, "primitive": primitive,
            "status": status, "started_at": started_at, "finished_at": finished_at,
            "duration_ms": (finished_at - started_at) * 1000,
            "cost": cost, "output_preview": output_preview, "error": error,
            "error_type": error_type, "traceback": traceback,
            "model": model, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    def get_run_steps(self, run_id: int) -> list[dict[str, Any]]:
        return [dict(s) for s in self._steps if s["run_id"] == run_id]

    def update_step_cost(self, run_id: int, step_name: str, cost: float) -> None:
        for s in self._steps:
            if s["run_id"] == run_id and s["step_name"] == step_name:
                s["cost"] = cost
                break

    def save_step_output(self, run_id: int, step_name: str, output_json: str) -> None:
        for s in self._steps:
            if s["run_id"] == run_id and s["step_name"] == step_name:
                s["step_output_json"] = output_json
                return
        # Step not logged yet — store for later
        self._steps.append({
            "run_id": run_id, "step_name": step_name,
            "status": "success", "step_output_json": output_json,
        })

    def get_step_outputs(self, run_id: int) -> dict[str, Any]:
        result = {}
        for s in self._steps:
            if s["run_id"] == run_id and s.get("status") == "success" and s.get("step_output_json"):
                try:
                    result[s["step_name"]] = json.loads(s["step_output_json"])
                except Exception:
                    pass
        return result

    # --- Cache ---

    def cache_get(self, key: str) -> dict | None:
        return self._cache.get(key)

    def cache_put(self, key: str, model: str, response: Any, cost: float = 0) -> None:
        self._cache[key] = {"response": response, "cost": cost}

    def cache_stats(self) -> dict[str, Any]:
        return {"entries": len(self._cache), "total_hits": 0, "saved_cost": 0}

    # --- Dashboard ---

    def get_dashboard_stats(self) -> dict[str, Any]:
        return {
            "today_runs": len(self._runs), "today_cost": 0,
            "week_runs": len(self._runs), "week_cost": 0,
            "total_runs": len(self._runs), "total_cost": 0,
            "success_rate": 0, "top_pipelines": [], "recent_failures": [],
        }
