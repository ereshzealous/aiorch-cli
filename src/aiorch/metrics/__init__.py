# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Metrics module — Prometheus + optional OTEL export.

Opt-in via environment variables:
    AIORCH_METRICS_ENABLED=true   → enables Prometheus /metrics endpoint
    OTEL_EXPORTER_OTLP_ENDPOINT   → additionally pushes to OTEL collector
    OTEL_TRACES_ENABLED=true      → enables distributed tracing via OTEL

When disabled, all metric operations are no-ops (zero overhead).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("aiorch.metrics")

# ─── Global state ───
_enabled = False
_otel_tracing = False
_registry = None  # prometheus_client.CollectorRegistry

# Prometheus metric objects (None when disabled)
_http_request_duration = None
_http_requests_total = None
_run_total = None
_run_duration = None
_run_active = None
_step_duration = None
_step_errors_total = None
_llm_request_total = None
_llm_request_duration = None
_llm_tokens_total = None
_llm_cost_total = None
_llm_cache_hits_total = None
_llm_errors_total = None
_schedule_triggers_total = None
_schedule_missed_total = None
_db_query_duration = None

# Executor concurrency metrics (Phase 3)
_executor_in_flight = None
_executor_max_concurrent = None
_executor_sem_wait = None
_executor_claim_attempts_total = None
_executor_drain_total = None

# Artifact store metrics (input-artifacts feature)
_artifact_upload_total = None
_artifact_upload_bytes_total = None
_artifact_dedup_hits_total = None
_artifact_download_total = None
_artifact_quota_rejections_total = None
_artifact_storage_bytes = None

# Connector metrics (connectors feature — Phase F)
_connector_ops_total = None
_connector_op_duration = None
_connector_rows_total = None
_connector_bytes_total = None
_connector_errors_total = None

# Coordination-layer metrics (review-driven hardening — Phase C)
# Surface admission, reconciler, and Lua-reload state so operators
# can see distributed coordination health without SSH + redis-cli.
_admission_attempts_total = None
_runlease_set_size = None
_reconcile_runs_total = None
_reconcile_orphans_dropped_total = None
_lua_noscript_reloads_total = None

# OTEL tracer (None when disabled)
_tracer = None


def is_enabled() -> bool:
    return _enabled


def init_metrics() -> None:
    """Initialize metrics collection if enabled via environment.

    Call this once during application startup (lifespan).
    Safe to call multiple times — idempotent.
    """
    global _enabled, _otel_tracing, _registry, _tracer
    global _http_request_duration, _http_requests_total
    global _run_total, _run_duration, _run_active
    global _step_duration, _step_errors_total
    global _llm_request_total, _llm_request_duration, _llm_tokens_total
    global _llm_cost_total, _llm_cache_hits_total, _llm_errors_total
    global _schedule_triggers_total, _schedule_missed_total
    global _db_query_duration

    if _enabled:
        return  # already initialized

    metrics_enabled = os.environ.get("AIORCH_METRICS_ENABLED", "").lower() in ("true", "1", "yes")
    if not metrics_enabled:
        logger.info("Metrics disabled (set AIORCH_METRICS_ENABLED=true to enable)")
        return

    try:
        from prometheus_client import (
            CollectorRegistry, Counter, Histogram, Gauge,
        )
    except ImportError:
        logger.warning("prometheus_client not installed — metrics disabled. "
                       "Install with: pip install prometheus-client")
        return

    _registry = CollectorRegistry()
    _enabled = True

    # ─── HTTP metrics ───
    _http_request_duration = Histogram(
        "aiorch_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "path", "status"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        registry=_registry,
    )
    _http_requests_total = Counter(
        "aiorch_http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
        registry=_registry,
    )

    # ─── Pipeline run metrics ───
    _run_total = Counter(
        "aiorch_run_total",
        "Total pipeline runs",
        ["pipeline", "status", "trigger_type"],
        registry=_registry,
    )
    _run_duration = Histogram(
        "aiorch_run_duration_seconds",
        "Pipeline run duration",
        ["pipeline"],
        buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600],
        registry=_registry,
    )
    _run_active = Gauge(
        "aiorch_run_active",
        "Currently executing runs",
        registry=_registry,
    )

    # ─── Step metrics ───
    _step_duration = Histogram(
        "aiorch_step_duration_seconds",
        "Pipeline step duration",
        ["step_type"],
        buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60],
        registry=_registry,
    )
    _step_errors_total = Counter(
        "aiorch_step_errors_total",
        "Total step errors",
        ["step_type", "error_type"],
        registry=_registry,
    )

    # ─── LLM metrics ───
    _llm_request_total = Counter(
        "aiorch_llm_request_total",
        "Total LLM requests",
        ["provider", "model"],
        registry=_registry,
    )
    _llm_request_duration = Histogram(
        "aiorch_llm_request_duration_seconds",
        "LLM request latency",
        ["provider", "model"],
        buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60],
        registry=_registry,
    )
    _llm_tokens_total = Counter(
        "aiorch_llm_tokens_total",
        "Total LLM tokens processed",
        ["provider", "model", "direction"],
        registry=_registry,
    )
    _llm_cost_total = Counter(
        "aiorch_llm_cost_dollars",
        "Total LLM cost in dollars",
        ["provider", "model"],
        registry=_registry,
    )
    _llm_cache_hits_total = Counter(
        "aiorch_llm_cache_hits_total",
        "LLM cache hits",
        ["provider", "model"],
        registry=_registry,
    )
    _llm_errors_total = Counter(
        "aiorch_llm_errors_total",
        "Total LLM errors",
        ["provider", "model", "error_type"],
        registry=_registry,
    )

    # ─── Scheduler metrics ───
    _schedule_triggers_total = Counter(
        "aiorch_schedule_triggers_total",
        "Total scheduled triggers",
        ["pipeline", "trigger_type"],
        registry=_registry,
    )
    _schedule_missed_total = Counter(
        "aiorch_schedule_missed_total",
        "Missed schedule triggers",
        registry=_registry,
    )

    # ─── DB metrics ───
    _db_query_duration = Histogram(
        "aiorch_db_query_duration_seconds",
        "Database query latency",
        ["operation"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        registry=_registry,
    )

    # ─── Executor concurrency metrics ───
    global _executor_in_flight, _executor_max_concurrent, _executor_sem_wait
    global _executor_claim_attempts_total, _executor_drain_total
    _executor_in_flight = Gauge(
        "aiorch_executor_in_flight_runs",
        "Runs currently executing in this executor process",
        registry=_registry,
    )
    _executor_max_concurrent = Gauge(
        "aiorch_executor_max_concurrent",
        "Configured MAX_CONCURRENT_RUNS cap for this executor process",
        registry=_registry,
    )
    _executor_sem_wait = Histogram(
        "aiorch_executor_semaphore_wait_seconds",
        "Time a claimed run waited for a free semaphore slot before starting",
        buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 30.0, 120.0, 300.0],
        registry=_registry,
    )
    _executor_claim_attempts_total = Counter(
        "aiorch_executor_claim_attempts_total",
        "Total claim attempts, labeled by outcome",
        ["outcome"],  # "claimed" | "empty"
        registry=_registry,
    )
    _executor_drain_total = Counter(
        "aiorch_executor_drain_total",
        "Shutdown drain events, labeled by outcome",
        ["outcome"],  # "clean" | "timeout"
        registry=_registry,
    )

    # ─── Artifact store metrics ───
    global _artifact_upload_total, _artifact_upload_bytes_total, _artifact_dedup_hits_total
    global _artifact_download_total, _artifact_quota_rejections_total, _artifact_storage_bytes
    _artifact_upload_total = Counter(
        "aiorch_artifact_upload_total",
        "Artifacts successfully uploaded, labeled by role and content-type category",
        ["workspace_id", "role", "content_type_category"],
        registry=_registry,
    )
    _artifact_upload_bytes_total = Counter(
        "aiorch_artifact_upload_bytes_total",
        "Total bytes uploaded to the artifact store",
        ["workspace_id", "role"],
        registry=_registry,
    )
    _artifact_dedup_hits_total = Counter(
        "aiorch_artifact_dedup_hits_total",
        "Uploads that matched an existing artifact by sha256 (deduped, not stored twice)",
        ["workspace_id"],
        registry=_registry,
    )
    _artifact_download_total = Counter(
        "aiorch_artifact_download_total",
        "Artifact download URL requests, labeled by role and content-type category",
        ["workspace_id", "role", "content_type_category"],
        registry=_registry,
    )
    _artifact_quota_rejections_total = Counter(
        "aiorch_artifact_quota_rejections_total",
        "Uploads rejected due to size limits or workspace quota, labeled by reason",
        ["workspace_id", "reason"],  # reason: too_large | quota_exceeded
        registry=_registry,
    )
    _artifact_storage_bytes = Gauge(
        "aiorch_artifact_storage_bytes",
        "Current total bytes stored per workspace",
        ["workspace_id"],
        registry=_registry,
    )

    # ─── Connector metrics ───
    # Cardinality note: we deliberately DO NOT include connector_id in
    # labels — a workspace can have thousands of connectors and tagging
    # every series by id would blow up the Prometheus cardinality budget.
    # Type + subtype + operation is enough to spot hotspots, and the
    # audit log + connector_usage table hold the per-id trail.
    global _connector_ops_total, _connector_op_duration
    global _connector_rows_total, _connector_bytes_total, _connector_errors_total
    _connector_ops_total = Counter(
        "aiorch_connector_operations_total",
        "Connector operations dispatched, labeled by type/subtype/operation/outcome",
        ["type", "subtype", "operation", "outcome"],
        registry=_registry,
    )
    _connector_op_duration = Histogram(
        "aiorch_connector_operation_duration_seconds",
        "Connector operation latency end-to-end (handler execute + I/O)",
        ["type", "subtype", "operation"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        registry=_registry,
    )
    _connector_rows_total = Counter(
        "aiorch_connector_rows_total",
        "Rows returned / affected by connector operations (database + object_store list)",
        ["type", "subtype", "operation"],
        registry=_registry,
    )
    _connector_bytes_total = Counter(
        "aiorch_connector_bytes_total",
        "Bytes transferred by connector operations (object_store read/write mostly)",
        ["type", "subtype", "operation"],
        registry=_registry,
    )
    _connector_errors_total = Counter(
        "aiorch_connector_errors_total",
        "Connector operation failures, labeled by type/subtype and error class",
        ["type", "subtype", "error_class"],
        registry=_registry,
    )

    # ─── Coordination-layer observability (Phase C) ───
    # Fill the visibility gaps the code review called out — admission
    # attempts by outcome, runlease set cardinality per workspace,
    # reconciler runs, orphan-drop counter, and Lua script reloads.
    global _admission_attempts_total, _runlease_set_size
    global _reconcile_runs_total, _reconcile_orphans_dropped_total
    global _lua_noscript_reloads_total
    _admission_attempts_total = Counter(
        "aiorch_admission_attempts_total",
        "Outcomes of the admit_run Lua script",
        # outcome: admitted | rejected | duplicate | redis_unavailable
        ["workspace_id", "outcome"],
        registry=_registry,
    )
    _runlease_set_size = Gauge(
        "aiorch_runlease_set_size",
        "Cardinality of the per-workspace admission lease set",
        ["workspace_id"],
        registry=_registry,
    )
    _reconcile_runs_total = Counter(
        "aiorch_reconcile_runs_total",
        "Reconcile job completions from the scheduler's maintenance loop",
        ["kind", "outcome"],  # kind: admission|spend; outcome: ok|error
        registry=_registry,
    )
    _reconcile_orphans_dropped_total = Counter(
        "aiorch_reconcile_orphans_dropped_total",
        "Orphaned lease members removed by the admission reconciler",
        registry=_registry,
    )
    _lua_noscript_reloads_total = Counter(
        "aiorch_lua_noscript_reloads_total",
        "Times a Redis Lua script was re-loaded after a NOSCRIPT error "
        "(cache flushed, server restarted, or connected to a fresh replica)",
        ["script_name"],
        registry=_registry,
    )

    logger.info("Prometheus metrics enabled")

    # ─── Optional OTEL tracing ───
    otel_traces = os.environ.get("OTEL_TRACES_ENABLED", "").lower() in ("true", "1", "yes")
    otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    if otel_traces and otel_endpoint:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": "aiorch"})
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(endpoint=otel_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            _tracer = trace.get_tracer("aiorch")
            _otel_tracing = True
            logger.info("OTEL tracing enabled → %s", otel_endpoint)
        except ImportError:
            logger.warning("OTEL tracing requested but opentelemetry packages not installed")
        except Exception as e:
            logger.warning("OTEL tracing init failed: %s", e)

    # ─── Optional OTEL metrics export ───
    if otel_endpoint:
        try:
            from opentelemetry.exporter.prometheus import PrometheusMetricReader
            logger.info("OTEL metrics export available alongside Prometheus scrape")
        except ImportError:
            pass  # Fine — Prometheus scrape is the primary path


def get_registry():
    """Return the Prometheus CollectorRegistry (for /metrics endpoint)."""
    return _registry


# ═══════════════════════════════════════════════════════════
# PUBLIC INSTRUMENTATION API
# All functions are safe to call when metrics are disabled (no-op).
# ═══════════════════════════════════════════════════════════

# ─── HTTP ───

def observe_http_request(method: str, path: str, status: int, duration: float) -> None:
    if not _enabled:
        return
    # Normalize path to reduce cardinality: /api/runs/123 → /api/runs/{id}
    normalized = _normalize_path(path)
    _http_request_duration.labels(method=method, path=normalized, status=str(status)).observe(duration)
    _http_requests_total.labels(method=method, path=normalized, status=str(status)).inc()


def _normalize_path(path: str) -> str:
    """Reduce path cardinality by replacing IDs with {id}."""
    import re
    # Replace UUIDs
    path = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '{id}', path)
    # Replace numeric IDs
    path = re.sub(r'/(\d+)(?=/|$)', '/{id}', path)
    return path


# ─── Runs ───

def inc_run_started(pipeline: str, trigger_type: str = "manual") -> None:
    if not _enabled:
        return
    _run_active.inc()


def observe_run_completed(pipeline: str, status: str, duration: float, trigger_type: str = "manual") -> None:
    if not _enabled:
        return
    _run_total.labels(pipeline=pipeline, status=status, trigger_type=trigger_type).inc()
    _run_duration.labels(pipeline=pipeline).observe(duration)
    _run_active.dec()


# ─── Steps ───

def observe_step(step_type: str, duration: float) -> None:
    if not _enabled:
        return
    _step_duration.labels(step_type=step_type).observe(duration)


def inc_step_error(step_type: str, error_type: str = "unknown") -> None:
    if not _enabled:
        return
    _step_errors_total.labels(step_type=step_type, error_type=error_type).inc()


# ─── LLM ───

def observe_llm_request(
    provider: str, model: str, duration: float,
    prompt_tokens: int = 0, completion_tokens: int = 0,
    cost: float = 0.0, cache_hit: bool = False,
) -> None:
    if not _enabled:
        return
    _llm_request_total.labels(provider=provider, model=model).inc()
    _llm_request_duration.labels(provider=provider, model=model).observe(duration)
    if prompt_tokens:
        _llm_tokens_total.labels(provider=provider, model=model, direction="prompt").inc(prompt_tokens)
    if completion_tokens:
        _llm_tokens_total.labels(provider=provider, model=model, direction="completion").inc(completion_tokens)
    if cost:
        _llm_cost_total.labels(provider=provider, model=model).inc(cost)
    if cache_hit:
        _llm_cache_hits_total.labels(provider=provider, model=model).inc()


def inc_llm_error(provider: str, model: str, error_type: str = "unknown") -> None:
    if not _enabled:
        return
    _llm_errors_total.labels(provider=provider, model=model, error_type=error_type).inc()


# ─── Scheduler ───

def inc_schedule_trigger(pipeline: str, trigger_type: str = "cron") -> None:
    if not _enabled:
        return
    _schedule_triggers_total.labels(pipeline=pipeline, trigger_type=trigger_type).inc()


def inc_schedule_missed() -> None:
    if not _enabled:
        return
    _schedule_missed_total.inc()


# ─── DB ───

def observe_db_query(operation: str, duration: float) -> None:
    if not _enabled:
        return
    _db_query_duration.labels(operation=operation).observe(duration)


# ─── Executor concurrency ───

def set_executor_max_concurrent(cap: int) -> None:
    """Record the configured in-process concurrency cap. Called once on startup."""
    if not _enabled:
        return
    _executor_max_concurrent.set(cap)


def inc_executor_in_flight() -> None:
    """A run just entered the semaphore critical section."""
    if not _enabled:
        return
    _executor_in_flight.inc()


def dec_executor_in_flight() -> None:
    """A run just released its semaphore slot (completed or crashed)."""
    if not _enabled:
        return
    _executor_in_flight.dec()


def observe_executor_sem_wait(duration: float) -> None:
    """Record how long a claimed run waited for a free semaphore slot."""
    if not _enabled:
        return
    _executor_sem_wait.observe(duration)


def inc_executor_claim_attempt(outcome: str) -> None:
    """Record a claim attempt. outcome: 'claimed' or 'empty'."""
    if not _enabled:
        return
    _executor_claim_attempts_total.labels(outcome=outcome).inc()


def inc_executor_drain(outcome: str) -> None:
    """Record a shutdown drain outcome. outcome: 'clean' or 'timeout'."""
    if not _enabled:
        return
    _executor_drain_total.labels(outcome=outcome).inc()


# ─── Artifact store ───


def _content_type_category(content_type: str | None) -> str:
    """Collapse content types into a small set of label values.

    Keeps the cardinality of the Prometheus labels bounded. We don't
    want one label value per long-tail MIME type.
    """
    if not content_type:
        return "other"
    prefix = content_type.split("/", 1)[0]
    if prefix in ("text", "image", "video", "audio", "application"):
        return prefix
    return "other"


def inc_artifact_upload(
    workspace_id: str,
    role: str,
    content_type: str | None,
    size_bytes: int,
) -> None:
    """Record a successful artifact upload (post-dedup)."""
    if not _enabled:
        return
    cat = _content_type_category(content_type)
    _artifact_upload_total.labels(
        workspace_id=workspace_id, role=role, content_type_category=cat,
    ).inc()
    _artifact_upload_bytes_total.labels(
        workspace_id=workspace_id, role=role,
    ).inc(size_bytes)


def inc_artifact_dedup_hit(workspace_id: str) -> None:
    """Record that an upload matched an existing sha256 and was deduped."""
    if not _enabled:
        return
    _artifact_dedup_hits_total.labels(workspace_id=workspace_id).inc()


def inc_artifact_download(
    workspace_id: str | None,
    role: str,
    content_type: str | None,
) -> None:
    """Record a signed-URL download request."""
    if not _enabled:
        return
    cat = _content_type_category(content_type)
    _artifact_download_total.labels(
        workspace_id=workspace_id or "unknown",
        role=role,
        content_type_category=cat,
    ).inc()


def inc_artifact_quota_rejection(workspace_id: str, reason: str) -> None:
    """Record an upload rejected for size or quota reasons.

    reason: 'too_large' | 'quota_exceeded'
    """
    if not _enabled:
        return
    _artifact_quota_rejections_total.labels(
        workspace_id=workspace_id, reason=reason,
    ).inc()


def set_artifact_storage_bytes(workspace_id: str, total_bytes: int) -> None:
    """Update the workspace's current storage usage gauge."""
    if not _enabled:
        return
    _artifact_storage_bytes.labels(workspace_id=workspace_id).set(total_bytes)


# ─── Connectors ───

def inc_connector_operation(
    *,
    connector_id: str,
    type: str,
    subtype: str,
    operation: str,
    outcome: str,
) -> None:
    """Record one connector operation dispatch.

    connector_id is accepted for signature compatibility with the
    dispatcher but deliberately NOT used as a label — see the
    cardinality note in init_metrics().
    """
    if not _enabled:
        return
    _connector_ops_total.labels(
        type=type, subtype=subtype, operation=operation, outcome=outcome,
    ).inc()


def observe_connector_duration(
    *,
    type: str,
    subtype: str,
    operation: str,
    seconds: float,
) -> None:
    """Record the handler-to-dispatch round-trip duration in seconds."""
    if not _enabled:
        return
    _connector_op_duration.labels(
        type=type, subtype=subtype, operation=operation,
    ).observe(seconds)


def inc_connector_rows(
    *,
    type: str,
    subtype: str,
    operation: str,
    count: int,
) -> None:
    """Record rows returned / affected (databases and object store list)."""
    if not _enabled or count is None or count <= 0:
        return
    _connector_rows_total.labels(
        type=type, subtype=subtype, operation=operation,
    ).inc(count)


def inc_connector_bytes(
    *,
    type: str,
    subtype: str,
    operation: str,
    count: int,
) -> None:
    """Record bytes transferred (object store read/write)."""
    if not _enabled or count is None or count <= 0:
        return
    _connector_bytes_total.labels(
        type=type, subtype=subtype, operation=operation,
    ).inc(count)


def inc_connector_error(
    *,
    type: str,
    subtype: str,
    error_class: str,
) -> None:
    """Record a connector operation failure by error class."""
    if not _enabled:
        return
    _connector_errors_total.labels(
        type=type, subtype=subtype, error_class=error_class,
    ).inc()


# ─── Coordination-layer instrumentation (Phase C) ───

def inc_admission_attempt(workspace_id: str, outcome: str) -> None:
    """Record an admit_run.lua outcome.

    `outcome` is one of: admitted | rejected | duplicate | redis_unavailable.
    Called from server/concurrency.py at every branch of the admission
    path including the Redis-unavailable fallback.
    """
    if not _enabled:
        return
    _admission_attempts_total.labels(
        workspace_id=workspace_id, outcome=outcome,
    ).inc()


def set_runlease_set_size(workspace_id: str, size: int) -> None:
    """Update the per-workspace admission set cardinality gauge.

    Called from the scheduler's admission reconciler, which already
    iterates every runlease set on its 30s tick. Free piggyback.
    """
    if not _enabled:
        return
    _runlease_set_size.labels(workspace_id=workspace_id).set(max(0, int(size)))


def inc_reconcile_run(kind: str, outcome: str = "ok") -> None:
    """Record a reconcile-loop completion. kind=admission|spend,
    outcome=ok|error."""
    if not _enabled:
        return
    _reconcile_runs_total.labels(kind=kind, outcome=outcome).inc()


def inc_reconcile_orphans_dropped(count: int = 1) -> None:
    """Increment the cumulative orphan-drops counter by `count`."""
    if not _enabled or count <= 0:
        return
    _reconcile_orphans_dropped_total.inc(count)


def inc_lua_noscript_reload(script_name: str) -> None:
    """Record that a Lua script was re-loaded after a NOSCRIPT error.
    Frequent NOSCRIPT reloads signal Redis restarts or script cache
    evictions — worth alerting on in production."""
    if not _enabled:
        return
    _lua_noscript_reloads_total.labels(script_name=script_name).inc()


# ─── OTEL Tracing ───

@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """Create an OTEL trace span. No-op when tracing is disabled."""
    if not _otel_tracing or not _tracer:
        yield None
        return
    with _tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        yield span
