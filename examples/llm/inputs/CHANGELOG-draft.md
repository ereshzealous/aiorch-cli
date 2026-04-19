# v0.5.0 — 2026-04-19

## Features

- Added webhook delivery history, test-fire functionality, and per-trigger rate limits.
- Improved health observability with admission metrics and reconcile state.
- Enhanced UI with dirty-form tracking, inline field errors, and accessible modals.
- Expanded MCP registry to include Opsgenie, MongoDB, Jira, Confluence, Sentry, ClickHouse, PagerDuty, and Grafana.
- Command-line interface now supports resuming runs from a specific step after failure.

## Fixes

- API documentation now hides internal routes and accurately describes response shapes.
- Executor now properly guards claim tokens during heartbeat and run finalization.
- Executor offloads synchronous store and Redis calls for improved responsiveness.
- Scheduler correctly rolls back claims on pipeline-resolve failures.
- Suppressed SSE on run ID change to prevent event bleed between runs.
- Pinned pip version in CI to avoid regressions.

## Performance

- Health endpoint now batches pg_stat_activity reads for better performance.

## Security

- Authentication is now secure-by-default with boot guard.
- Resolved an issue with `.env.example` disclosure.

## Refactoring

- Unified Redis access through a shared cache client for consistency.
- Explicit startup task registry with timing for better server management.
- Extracted admission lease helpers into a concurrency module.
- Updated CLAUDE_HANDOFF documentation for webhook trigger redesign.

## Docs

- Updated CLAUDE_HANDOFF documentation to reflect Phase-11 webhook trigger redesign.
