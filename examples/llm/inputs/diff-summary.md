# Diff summary

**Stats:** 2 files changed · +54 / -0

## What changed
- Webhook triggers now support configurable rate limiting, allowing users to specify a maximum number of deliveries per minute.
- A new `webhook_deliveries` table has been added to record detailed logs of each webhook invocation, including its outcome, even if it fails.
- The webhook firing mechanism now performs a rate limit check before processing, returning a `429 Too Many Requests` error if the limit is exceeded.
- Delivery records are created for all webhook invocations, including those that are rate-limited, providing better observability into webhook activity.

## Risk assessment
- **Database Schema Change Impact:** The additions of `max_per_minute` to `webhook_triggers` and the new `webhook_deliveries` table require careful review of the Alembic migration script to ensure it handles existing data correctly and does not cause downtime or data loss during deployment.
- **Rate Limiting Accuracy and Performance:** The `_check_rate_limit` function's reliance on Redis and potential degradation on outage should be verified for consistency and performance under high load. The Redis keyspace management and expiration could also be a concern.
- **Error Handling in `_record_delivery`:** The `try...except Exception` block in `_record_delivery` suppresses all errors during delivery recording. While intended to prevent breaking the fire path, it could mask critical issues with database writes to the `webhook_deliveries` table, potentially leading to incomplete logging without notification. Consider logging the exception or adding specific error handling.
- **Completeness of `_record_delivery` arguments:** The `INSERT INTO webhook_deliveries (...) VALUES (...)` line is a placeholder. The actual columns being inserted and the corresponding values (`...`) need to be thoroughly checked to ensure all necessary delivery information is being captured correctly and aligns with the new table's schema.
