# Changelog

Generated from `/Users/ereshgorantla/Documents/Dev/oss/aiorch-cli/examples/core/inputs/sample-commits.txt` (25 commits)

## Features

- parallel step dispatch with bounded worker pool
- cron-free schedule table backed by postgres
- webhooks management page skeleton
- connectors pipeline polish pass
- inbound webhook triggers with hmac verification
- surface condition-skipped steps in trace

## Bug fixes

- drain in-flight tasks on shutdown
- webhook header form cleared on save
- reject empty hmac secret at registration
- avoid double-fire when schedule row updated mid-tick

## Performance

- remove per-step asyncio.sleep(0) yield

## Refactors

- drop apscheduler dep
- pull status enum into shared module

## Docs

- note postgres-coordination model
- trace semantics for skipped
- connector auth picker walkthrough

## Tests

- webhook integration suite
- cancellation racer regression test

## Build

- multi-stage image with uv pip install

## Chores

- upgrade tailwind to 3.5
- bump httpx 0.27 -> 0.28
- pin ruff 0.5.7

## Style

- black + ruff sweep

## Other

- Merge pull request #412 from priyan/executor-pool
- Random commit without a conventional prefix
