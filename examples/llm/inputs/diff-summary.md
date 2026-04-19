# Diff summary

**Stats:** 2 files changed · +54 / -0

## What changed
- Webhook triggers now support an optional `max_per_minute` rate limit, preventing excessive invocations.
- A new `webhook_deliveries` table has been added to record each attempt to fire a webhook, including details like status and outcome.
- If a webhook trigger exceeds its rate limit, a `429 Too Many Requests` HTTP response is returned, and this event is logged in the new `webhook_deliveries` table.
- The API for creating and updating webhook triggers has been extended to include the `max_per_minute` setting.

## Risk assessment
- The `_record_delivery` function catches all exceptions, silently failing to record delivery. While intended to prevent fire path breakage, this could mask critical database issues with delivery logging.
- The rate limiting logic relies on `trigger.get("max_per_minute", 60)`. Ensure this default matches expectations, especially if some triggers might not have this column yet during a mixed-version deployment.
- The `INSERT INTO webhook_deliveries` SQL is incomplete (`(...) VALUES (...)`). This needs to be filled in correctly for the delivery logging to function at all.
- Review the `_check_rate_limit` implementation. While it states "degrades open on outage," ensure this behavior is acceptable under Redis failures.
