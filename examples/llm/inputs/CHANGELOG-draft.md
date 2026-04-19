# v0.5.0 — 2026-04-20

## Features

- Webhooks now include delivery history, test-firing, and per-trigger rate limits.
- Improved observability with admission metrics and reconcile state for the coordination layer.
- The UI now tracks dirty forms, shows inline field errors, and features accessible modals.
- Expanded MCP Registry to include Opsgenie, MongoDB, Jira, Confluence, Sentry, ClickHouse, PagerDuty, and Grafana.
- CLI now supports `--from <step>` to resume runs after failures.

## Fixes

- Stopped silencing `aiorch` loggers after boot-time Alembic execution.
- API documentation now hides internal routes and accurately describes response shapes.
- Executor heartbeat and `finish_run` now properly guard token claims.
- Executor offloads synchronous store and Redis calls using `asyncio.to_thread` for better performance.
- Eliminated SSE bleed across different run IDs by suppressing events on run ID changes.

## Performance

- Batching `pg_stat_activity` reads on the health endpoint to reduce database load.

## Security

- Authentication is now secure-by-default with a boot guard.
- Removed `.env.example` disclosure vulnerability.

## Refactoring

- Unified Redis access through a shared cache client.
- Explicitly registered startup tasks with timing information for better server management.
- Extracted admission lease helpers into a dedicated concurrency module for cleaner storage logic.

## Docs

- Updated CLAUDE_HANDOFF documentation with new webhook trigger redesign details.
