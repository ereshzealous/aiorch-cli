# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Webhook connector handler — generic + Slack / Discord / Teams.

One handler class per subtype. All share the same HTTP transport
(httpx.AsyncClient) and the same canonical ``send`` operation; the
only difference is how they format the payload for the destination
service.

Per Q2 of the connectors design doc, every send opens a fresh
httpx.AsyncClient context, POSTs, and closes. No shared client
across operations — simpler mental model, automatic secret
rotation, no pool lifecycle to manage.

Retry policy: transient 5xx responses are retried up to
max_retries with exponential backoff (1s, 2s, 4s). 4xx errors
are not retried.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiorch.connectors import ConnectorAuthError, NotSupportedError
from aiorch.connectors.handlers import ConnectorHandler, register_handler

logger = logging.getLogger("aiorch.connectors.handlers.webhook")


DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Payload formatters — one per subtype
# ---------------------------------------------------------------------------


def _format_generic(params: dict) -> dict:
    """Generic webhook — pass the raw body through as JSON or bytes."""
    body = params.get("body") or params.get("payload")
    if body is None:
        # Fall back to `message` so step configs written with either
        # `message:` or `body:` work uniformly.
        body = params.get("message")

    if body is None:
        raise ValueError("webhook send requires 'body', 'payload', or 'message'")

    if isinstance(body, (dict, list)):
        return {"json_body": body}
    if isinstance(body, str):
        # Try JSON-parse first so strings that happen to be JSON
        # go over the wire as JSON. Otherwise send as plain text.
        import json as _json
        try:
            parsed = _json.loads(body)
            if isinstance(parsed, (dict, list)):
                return {"json_body": parsed}
        except (_json.JSONDecodeError, ValueError):
            pass
        return {"text_body": body}
    if isinstance(body, bytes):
        return {"text_body": body.decode("utf-8", errors="replace")}

    raise ValueError(f"webhook body must be dict/list/str/bytes, got {type(body).__name__}")


def _format_slack(params: dict) -> dict:
    """Slack Incoming Webhook payload.

    https://api.slack.com/messaging/webhooks#posting_with_webhooks
    """
    message = params.get("message") or params.get("text")
    if not message:
        raise ValueError("Slack webhook send requires 'message'")

    payload: dict[str, Any] = {"text": message}

    channel = params.get("channel")
    if channel:
        payload["channel"] = channel

    attachments = params.get("attachments")
    if attachments:
        payload["attachments"] = attachments

    blocks = params.get("blocks")
    if blocks:
        payload["blocks"] = blocks

    thread_ts = params.get("thread_ts") or params.get("thread_id")
    if thread_ts:
        payload["thread_ts"] = thread_ts

    username = params.get("username")
    if username:
        payload["username"] = username

    icon_emoji = params.get("icon_emoji")
    if icon_emoji:
        payload["icon_emoji"] = icon_emoji

    return {"json_body": payload}


def _format_discord(params: dict) -> dict:
    """Discord webhook payload.

    https://discord.com/developers/docs/resources/webhook#execute-webhook
    """
    message = params.get("message") or params.get("content")
    if not message:
        raise ValueError("Discord webhook send requires 'message'")

    # Discord has a 2000 char limit on content
    payload: dict[str, Any] = {"content": message[:2000]}

    username = params.get("username")
    if username:
        payload["username"] = username

    avatar_url = params.get("avatar_url")
    if avatar_url:
        payload["avatar_url"] = avatar_url

    embeds = params.get("embeds")
    if embeds:
        payload["embeds"] = embeds

    return {"json_body": payload}


def _format_teams(params: dict) -> dict:
    """Microsoft Teams Adaptive Card payload.

    https://learn.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/connectors-using
    """
    message = params.get("message") or params.get("text")
    if not message:
        raise ValueError("Teams webhook send requires 'message'")

    body_text = params.get("body", "")

    card_body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": message,
            "weight": "Bolder",
            "size": "Medium",
            "wrap": True,
        },
    ]
    if body_text:
        card_body.append({
            "type": "TextBlock",
            "text": str(body_text),
            "wrap": True,
        })

    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": card_body,
                },
            }
        ],
    }
    return {"json_body": payload}


_FORMATTERS = {
    "generic": _format_generic,
    "slack": _format_slack,
    "discord": _format_discord,
    "teams": _format_teams,
}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class WebhookHandler(ConnectorHandler):
    """Outbound HTTP webhook handler.

    One class handles all four subtypes — the subtype switch happens
    inside execute() and drives the payload formatter choice. This
    is different from the object store handler which registers one
    instance per subtype because we need different payload shapes
    per destination service, but the actual transport and auth
    layers are identical.
    """

    type = "webhook"
    subtype = "generic"
    supported_operations = {"send"}

    @classmethod
    def get_client_info(cls) -> dict | None:
        try:
            import httpx
            version = getattr(httpx, "__version__", "unknown")
        except ImportError:
            return None
        return {
            "client_library": "httpx",
            "client_version": version,
            "docs_url": "https://www.python-httpx.org/api/",
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

        webhook_url = connector_config.get("webhook_url")
        if not webhook_url:
            raise ValueError(
                "Webhook connector requires 'webhook_url' in config/secrets"
            )

        # SSRF gate (Round 2 #13 in the audit): the webhook_url is
        # template-rendered at the connector layer and could be
        # pointed at IMDS / loopback / RFC1918 by a malicious
        # connector config. safe_http_url enforces the scheme
        # allowlist and private-host block; AIORCH_ALLOW_PRIVATE_HOSTS=1
        # opts out for operators with internal-service use cases.
        from aiorch.core.http_safety import safe_http_url
        webhook_url = safe_http_url(
            webhook_url,
            purpose=f"webhook connector ({self.subtype})",
        )

        timeout_seconds = int(
            connector_config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        )
        max_retries = int(
            connector_config.get("max_retries", DEFAULT_MAX_RETRIES)
        )

        # HTTP method — only the generic subtype is configurable;
        # slack/discord/teams are POST by protocol spec.
        if self.subtype == "generic":
            method = str(connector_config.get("method", "POST")).upper()
            if method not in ("POST", "PUT", "PATCH"):
                raise ValueError(
                    f"webhook method must be POST, PUT, or PATCH — got {method!r}"
                )
        else:
            method = "POST"

        # Format the payload for this subtype
        formatter = _FORMATTERS.get(self.subtype, _format_generic)
        formatted = formatter(params)

        # Build headers from extra_headers (a dict on disk; the UI's KV
        # row editor converts to/from a list of {k, v} for editing only).
        headers = dict(connector_config.get("extra_headers") or {})
        url_params: dict[str, str] = {}

        # Resolve auth_type with backward-compat inference for connectors
        # created before Phase 6 — those have auth_token / basic_auth_*
        # set but no auth_type field. Existing connectors keep working
        # without a re-save.
        auth_type = connector_config.get("auth_type")
        if not auth_type:
            if connector_config.get("auth_token"):
                auth_type = "bearer"
            elif connector_config.get("basic_auth_username"):
                auth_type = "basic"
            else:
                auth_type = "none"

        if auth_type == "bearer":
            auth_token = connector_config.get("auth_token")
            if auth_token and "Authorization" not in headers:
                headers["Authorization"] = f"Bearer {auth_token}"

        elif auth_type == "basic":
            basic_username = connector_config.get("basic_auth_username")
            if basic_username and "Authorization" not in headers:
                import base64
                basic_password = connector_config.get("basic_auth_password", "") or ""
                token = base64.b64encode(
                    f"{basic_username}:{basic_password}".encode("utf-8")
                ).decode("ascii")
                headers["Authorization"] = f"Basic {token}"

        elif auth_type == "api_key":
            api_key_name = connector_config.get("api_key_name")
            api_key_value = connector_config.get("api_key_value")
            api_key_in = (connector_config.get("api_key_in") or "header").lower()
            if api_key_name and api_key_value is not None:
                if api_key_in == "header":
                    if api_key_name not in headers:
                        headers[api_key_name] = str(api_key_value)
                elif api_key_in == "query":
                    url_params[api_key_name] = str(api_key_value)
                else:
                    raise ValueError(
                        f"api_key_in must be 'header' or 'query', got {api_key_in!r}"
                    )

        # auth_type == "none" → no Authorization header added

        import httpx

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    request_kwargs: dict = {"headers": headers}
                    if url_params:
                        request_kwargs["params"] = url_params
                    if "json_body" in formatted:
                        response = await client.request(
                            method,
                            webhook_url,
                            json=formatted["json_body"],
                            **request_kwargs,
                        )
                    else:
                        request_kwargs["headers"] = {**headers, "Content-Type": "text/plain"}
                        response = await client.request(
                            method,
                            webhook_url,
                            content=formatted["text_body"],
                            **request_kwargs,
                        )

                # 2xx: success, return immediately
                if 200 <= response.status_code < 300:
                    # Auto-parse JSON responses so downstream steps can
                    # template `{{result.body.field}}`; fall back to raw
                    # text on parse failure.
                    content_type = response.headers.get("content-type", "") if hasattr(response, "headers") else ""
                    response_body: Any = response.text
                    if "json" in content_type.lower():
                        try:
                            import json as _json
                            response_body = _json.loads(response.text)
                        except Exception:
                            pass
                    return {
                        "status_code": response.status_code,
                        "body": response_body,
                        "body_preview": response.text[:200],
                        "bytes_transferred": len(response.content),
                        "content_type": content_type,
                    }

                # 4xx: client error, don't retry
                if 400 <= response.status_code < 500:
                    if response.status_code in (401, 403):
                        raise ConnectorAuthError(
                            f"Webhook {self.subtype} rejected credentials "
                            f"({response.status_code}): {response.text[:200]}"
                        )
                    response.raise_for_status()

                # 5xx: retry with backoff
                if attempt < max_retries:
                    backoff = 2 ** attempt   # 1s, 2s, 4s
                    logger.warning(
                        "Webhook %s returned %d, retrying in %ds (attempt %d/%d)",
                        self.subtype, response.status_code, backoff,
                        attempt + 1, max_retries,
                    )
                    await asyncio.sleep(backoff)
                    continue

                response.raise_for_status()

            except httpx.TimeoutException as exc:
                if attempt < max_retries:
                    backoff = 2 ** attempt
                    logger.warning(
                        "Webhook %s timed out, retrying in %ds",
                        self.subtype, backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise TimeoutError(f"Webhook {self.subtype} timed out: {exc}") from exc

        # Should not reach here; all paths either return or raise
        raise RuntimeError(f"Webhook {self.subtype} failed after {max_retries} retries")


# ---------------------------------------------------------------------------
# Registration — one instance per subtype
# ---------------------------------------------------------------------------


def _make_webhook_handler(subtype_name: str) -> WebhookHandler:
    h = WebhookHandler()
    h.subtype = subtype_name
    return h


for _subtype in ("generic", "slack", "discord", "teams"):
    register_handler(_make_webhook_handler(_subtype))
