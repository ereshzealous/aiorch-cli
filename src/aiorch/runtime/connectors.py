# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Connectors — outbound integrations for pipelines.

Each connector is an action handler registered in the action registry.
All connectors share: retry with exponential backoff, template resolution,
and secret injection via ${VAR} syntax.

Action-specific parameters are read from step.config (dict).

Built-in connectors:
  - email         — send email via SMTP
  - s3            — upload/download from S3-compatible storage
  - kafka         — produce messages to Kafka topics
  - teams         — Microsoft Teams incoming webhook
  - discord       — Discord incoming webhook

Usage in YAML:
    steps:
      notify:
        action: email
        config:
          to: "team@company.com"
          subject: "Pipeline {{pipeline_name}} completed"
          body: "{{steps.report.output}}"
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from aiorch.core import template
from aiorch.core.parser import Step
from aiorch.runtime.action import _with_retry, _cfg_resolve

logger = logging.getLogger("aiorch.connectors")


# ═══════════════════════════════════════════════════════════
# CONNECTOR — managed connector dispatch (Platform only)
# ═══════════════════════════════════════════════════════════


def _resolve_config_templates(cfg: dict, context: dict) -> dict:
    """Walk a step.config dict and Jinja-resolve every string value.

    Nested dicts/lists are descended recursively; scalars that aren't
    strings are left untouched. This matches how the legacy action
    handlers call ``template.resolve()`` per field — centralized here
    so the ``action: connector`` path gets the same behavior without
    field-by-field enumeration.
    """
    def _walk(value):
        if isinstance(value, str):
            try:
                return template.resolve(value, context)
            except Exception:
                return value
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        return value
    return _walk(cfg)


async def connector_action(step: Step, context: dict[str, Any]) -> Any:
    """Dispatch a ``action: connector`` step.

    YAML:
        action: connector
        config:
          uses: prod_readonly_pg   # connector name in scope
          operation: query
          sql: "SELECT * FROM orders WHERE id = $1"
          params: [{{order_id}}]

    Optional ``track: true`` on write operations records the written
    data as an output artifact so it appears on the Artifacts page
    with full run lineage::

        action: connector
        config:
          uses: my-bucket
          operation: write
          key: "reports/{{date}}.json"
          content: "{{report_json}}"
          track: true

    All config fields except ``uses``, ``operation``, and ``track``
    are passed to the handler as canonical params. Jinja templates
    are resolved in every string field before dispatch.
    """
    from aiorch.connectors.integration import resolve_and_execute

    cfg = step.config or {}
    if "uses" not in cfg:
        raise ValueError(
            f"Step '{step.name}': action: connector requires "
            f"'uses:' (connector name) in config"
        )
    if "operation" not in cfg:
        raise ValueError(
            f"Step '{step.name}': action: connector requires "
            f"'operation:' in config (e.g. query, read, send, publish)"
        )

    resolved = _resolve_config_templates(cfg, context)

    # Pop track before passing to the handler — it's not a connector param
    track = resolved.pop("track", False)

    result = await resolve_and_execute(
        resolved, context, step_name=step.name or "<inline>",
    )

    # If track=true on a write operation, record the written content
    # as an output artifact so it shows up on the Artifacts page.
    if track and cfg.get("operation") == "write":
        _track_connector_write(step, context, resolved, result)

    return result


def _track_connector_write(
    step: Step,
    context: dict[str, Any],
    resolved_cfg: dict,
    result: dict,
) -> None:
    """Record a connector write as a tracked output artifact."""
    try:
        from aiorch.artifacts import get_artifact_store
        from aiorch.constants import RUNTIME_META_KEY

        artifact_store = get_artifact_store()

        content = resolved_cfg.get("content") or resolved_cfg.get("body") or ""
        content_bytes = content.encode("utf-8") if isinstance(content, str) else content
        key = resolved_cfg.get("key", step.name or "connector-output")
        content_type = resolved_cfg.get("content_type", "application/octet-stream")

        meta = context.get(RUNTIME_META_KEY, {}) if isinstance(context, dict) else {}
        pipeline_name = meta.get("pipeline_name") if isinstance(meta, dict) else None
        run_id = context.get("__run_id__")
        workspace_id = context.get("__workspace_id__")
        org_id = context.get("__org_id__")

        if not workspace_id:
            return

        artifact = artifact_store.put(
            name=key,
            content=content_bytes,
            content_type=content_type,
            role="output",
            workspace_id=workspace_id,
            org_id=org_id,
            pipeline_name=pipeline_name,
            run_id=run_id,
        )

        if run_id is not None:
            artifact_store.record_run_binding(
                run_id=int(run_id),
                artifact_id=artifact.id,
                binding_name=step.name or key,
                role="output",
            )

        logger.info(
            "Tracked connector write as artifact %s (%s) for run %s",
            artifact.id[:8], key, run_id,
        )
    except Exception as exc:
        # Best-effort — don't break the pipeline if tracking fails
        logger.warning("Failed to track connector write: %s", exc)


# ═══════════════════════════════════════════════════════════
# EMAIL — SMTP
# ═══════════════════════════════════════════════════════════

async def connector_email(step: Step, context: dict[str, Any]) -> str:
    """Send email via SMTP.

    YAML:
        action: email
        config:
          to: "user@example.com"
          subject: "Pipeline done"
          body: "Email body text"

        # Managed-connector form (Platform only):
        action: email
        config:
          uses: prod_smtp
          to: "ops@company.com"
          subject: "Alert"
          body: "..."

    Env vars:
        SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM, SMTP_USE_TLS
    """
    if step.config.get("uses"):
        from aiorch.connectors.integration import try_legacy_connector_auto_resolve
        resolved_cfg = _resolve_config_templates(step.config, context)
        result = await try_legacy_connector_auto_resolve(
            resolved_cfg, context,
            step_name=step.name or "email",
            type_name="email", subtype_name="smtp",
            operation="send",
        )
        recipients = result.get("recipients", 0) if isinstance(result, dict) else 0
        return f"Email sent to {recipients} recipient(s)"

    to_addr = _cfg_resolve(step, "to", context)
    subject = _cfg_resolve(step, "subject", context) or _cfg_resolve(step, "message", context)
    body = _cfg_resolve(step, "body", context)

    if not to_addr:
        raise ValueError("email action requires 'to' (recipient email) in config")
    if not subject:
        raise ValueError("email action requires 'subject' (email subject) in config")

    from aiorch.runtime.run_env import get_env
    smtp_host = get_env(context, "SMTP_HOST", "localhost")
    smtp_port = int(get_env(context, "SMTP_PORT", "587"))
    smtp_user = get_env(context, "SMTP_USER", "")
    smtp_pass = get_env(context, "SMTP_PASSWORD", "")
    smtp_from = get_env(context, "SMTP_FROM", "aiorch@localhost")
    use_tls = get_env(context, "SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")

    import asyncio

    def _send_sync():
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        msg = MIMEMultipart("alternative")
        msg["From"] = smtp_from
        msg["To"] = to_addr
        msg["Subject"] = subject

        if body.strip().startswith("<") and body.strip().endswith(">"):
            msg.attach(MIMEText(body, "html"))
        else:
            msg.attach(MIMEText(body, "plain"))

        if use_tls:
            server = smtplib.SMTP(smtp_host, smtp_port)
            server.starttls()
        else:
            server = smtplib.SMTP(smtp_host, smtp_port)

        if smtp_user:
            server.login(smtp_user, smtp_pass)

        recipients = [a.strip() for a in to_addr.split(",")]
        server.sendmail(smtp_from, recipients, msg.as_string())
        server.quit()
        return len(recipients)

    async def _send():
        count = await asyncio.get_event_loop().run_in_executor(None, _send_sync)
        return count

    count = await _with_retry(_send, retries=2, timeout=30, label="email")
    logger.info("Email sent to %s: %s", to_addr, subject[:60])
    return f"Email sent to {count} recipient(s): {subject[:80]}"


# ═══════════════════════════════════════════════════════════
# S3 — Upload/Download from S3-compatible storage
# ═══════════════════════════════════════════════════════════

async def connector_s3(step: Step, context: dict[str, Any]) -> str:
    """Upload or download from S3-compatible storage.

    YAML:
        action: s3
        config:
          operation: upload
          url: "s3://my-bucket/reports/{{date}}.json"
          body: "{{steps.generate.output}}"

        # Managed-connector form (Platform only):
        action: s3
        config:
          uses: prod_s3_bucket
          operation: write
          key: "reports/{{date}}.json"
          body: "{{report}}"

    Env vars:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
        S3_ENDPOINT_URL (optional, for MinIO/R2)
    """
    if step.config.get("uses"):
        from aiorch.connectors.integration import try_legacy_connector_auto_resolve
        from aiorch.connectors import get_connector_store
        resolved_cfg = _resolve_config_templates(step.config, context)
        # object_store has multiple subtypes — look up the connector
        # first to learn which subtype it maps to, then dispatch.
        org_id = context.get("__org_id__")
        store = get_connector_store()
        connector = store.resolve_by_name(
            org_id=org_id,
            name=resolved_cfg["uses"],
            workspace_id=context.get("__workspace_id__"),
            pipeline_name=(context.get("_meta") or {}).get("pipeline_name")
                if isinstance(context.get("_meta"), dict) else None,
            include_secrets=True,
        )
        if connector.type != "object_store":
            raise ValueError(
                f"Step '{step.name}': 'uses: {resolved_cfg['uses']}' "
                f"is a {connector.type} connector, not object_store."
            )
        result = await try_legacy_connector_auto_resolve(
            resolved_cfg, context,
            step_name=step.name or "s3",
            type_name="object_store", subtype_name=connector.subtype,
            operation=resolved_cfg.get("operation", "write"),
        )
        return str(result) if result is not None else "ok"

    import asyncio

    operation = _cfg_resolve(step, "operation", context, default="upload").lower()
    s3_url = _cfg_resolve(step, "url", context)

    if not s3_url:
        raise ValueError("s3 action requires 'url' (e.g., s3://bucket/key) in config")

    path = s3_url.replace("s3://", "")
    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    from aiorch.runtime.run_env import get_env
    endpoint_url = get_env(context, "S3_ENDPOINT_URL") or None

    def _do_s3():
        import boto3
        kwargs = {}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        s3 = boto3.client("s3", **kwargs)

        if operation == "upload":
            content = _cfg_resolve(step, "body", context)
            s3.put_object(Bucket=bucket, Key=key, Body=content.encode())
            return f"Uploaded {len(content)} bytes to s3://{bucket}/{key}"

        elif operation == "download":
            resp = s3.get_object(Bucket=bucket, Key=key)
            data = resp["Body"].read().decode()
            return data

        elif operation == "list":
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=100)
            keys = [o["Key"] for o in resp.get("Contents", [])]
            return json.dumps(keys)

        else:
            raise ValueError(f"Unknown s3 operation: {operation}. Use upload, download, or list.")

    result = await asyncio.get_event_loop().run_in_executor(None, _do_s3)
    logger.info("S3 %s: %s/%s", operation, bucket, key[:60])
    return result


# ═══════════════════════════════════════════════════════════
# KAFKA — Produce messages
# ═══════════════════════════════════════════════════════════

async def connector_kafka(step: Step, context: dict[str, Any]) -> str:
    """Produce a message to a Kafka topic.

    YAML:
        action: kafka
        config:
          topic: "my-topic"
          value: '{"event": "done", "data": "{{output}}"}'
          key: "pipeline-run"

        # Managed-connector form (Platform only):
        action: kafka
        config:
          uses: prod_kafka
          topic: "orders"
          value: "{{payload}}"

    Env vars:
        KAFKA_BOOTSTRAP_SERVERS, KAFKA_SECURITY_PROTOCOL, KAFKA_SASL_*
    """
    if step.config.get("uses"):
        from aiorch.connectors.integration import try_legacy_connector_auto_resolve
        resolved_cfg = _resolve_config_templates(step.config, context)
        # Legacy field 'value' maps to canonical 'payload'
        result = await try_legacy_connector_auto_resolve(
            resolved_cfg, context,
            step_name=step.name or "kafka",
            type_name="stream", subtype_name="kafka",
            operation="publish",
            param_map={"value": "payload"},
        )
        if isinstance(result, dict):
            return f"Sent to {result.get('topic')} (partition={result.get('partition')}, offset={result.get('offset')})"
        return str(result) if result is not None else "ok"

    import asyncio

    topic = _cfg_resolve(step, "topic", context) or _cfg_resolve(step, "url", context)
    if not topic:
        raise ValueError("kafka action requires 'topic' in config")

    value = _cfg_resolve(step, "value", context) or _cfg_resolve(step, "body", context) or _cfg_resolve(step, "message", context)
    key = _cfg_resolve(step, "key", context) or None

    from aiorch.runtime.run_env import get_env
    bootstrap = get_env(context, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    def _produce():
        from kafka import KafkaProducer

        config = {"bootstrap_servers": bootstrap.split(",")}

        protocol = get_env(context, "KAFKA_SECURITY_PROTOCOL")
        if protocol:
            config["security_protocol"] = protocol
        mechanism = get_env(context, "KAFKA_SASL_MECHANISM")
        if mechanism:
            config["sasl_mechanism"] = mechanism
        username = get_env(context, "KAFKA_SASL_USERNAME")
        if username:
            config["sasl_plain_username"] = username
        password = get_env(context, "KAFKA_SASL_PASSWORD")
        if password:
            config["sasl_plain_password"] = password

        producer = KafkaProducer(
            value_serializer=lambda v: v.encode(),
            key_serializer=lambda k: k.encode() if k else None,
            **config,
        )
        future = producer.send(topic, value=value, key=key)
        record = future.get(timeout=30)
        producer.flush()
        producer.close()
        return record.partition, record.offset

    partition, offset = await asyncio.get_event_loop().run_in_executor(None, _produce)
    logger.info("Kafka message sent to %s (partition=%d, offset=%d)", topic, partition, offset)
    return f"Sent to {topic} (partition={partition}, offset={offset})"


# ═══════════════════════════════════════════════════════════
# TEAMS — Microsoft Teams incoming webhook
# ═══════════════════════════════════════════════════════════

async def connector_teams(step: Step, context: dict[str, Any]) -> str:
    """Post message to Microsoft Teams via incoming webhook.

    YAML:
        action: teams
        config:
          url: "${TEAMS_WEBHOOK_URL}"
          message: "Pipeline **{{pipeline_name}}** completed"
          body: "Detailed text"

        # Managed-connector form (Platform only):
        action: teams
        config:
          uses: prod_teams
          message: "Alert"

    Env vars:
        TEAMS_WEBHOOK_URL (fallback if url not set)
    """
    if step.config.get("uses"):
        from aiorch.connectors.integration import try_legacy_connector_auto_resolve
        resolved_cfg = _resolve_config_templates(step.config, context)
        await try_legacy_connector_auto_resolve(
            resolved_cfg, context,
            step_name=step.name or "teams",
            type_name="webhook", subtype_name="teams",
            operation="send",
        )
        return f"Posted to Teams: {resolved_cfg.get('message', '')[:80]}"

    from aiorch.runtime.run_env import get_env as _get_env
    webhook_url = _cfg_resolve(step, "url", context) or _get_env(context, "TEAMS_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError("teams action requires 'url' in config or TEAMS_WEBHOOK_URL env var")

    # SSRF gate (Pass 3 #3 in the audit). The pass-2 fix patched
    # the action handler and the managed connector handler but
    # missed the direct connector path that lives here.
    from aiorch.core.http_safety import safe_http_url
    webhook_url = safe_http_url(webhook_url, purpose="teams connector")

    message = _cfg_resolve(step, "message", context)
    body_text = _cfg_resolve(step, "body", context)

    payload = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {"type": "TextBlock", "text": message, "weight": "Bolder", "size": "Medium"},
                    *([{"type": "TextBlock", "text": body_text, "wrap": True}] if body_text else []),
                ],
            },
        }],
    }

    async def _send():
        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()
            return resp.text

    await _with_retry(_send, label="teams")
    logger.info("Teams message sent: %s", message[:60])
    return f"Posted to Teams: {message[:80]}"


# ═══════════════════════════════════════════════════════════
# DISCORD — Discord incoming webhook
# ═══════════════════════════════════════════════════════════

async def connector_discord(step: Step, context: dict[str, Any]) -> str:
    """Post message to Discord via incoming webhook.

    YAML:
        action: discord
        config:
          url: "${DISCORD_WEBHOOK_URL}"
          message: "Pipeline completed: {{status}}"

        # Managed-connector form (Platform only):
        action: discord
        config:
          uses: prod_discord
          message: "Done"

    Env vars:
        DISCORD_WEBHOOK_URL (fallback if url not set)
    """
    if step.config.get("uses"):
        from aiorch.connectors.integration import try_legacy_connector_auto_resolve
        resolved_cfg = _resolve_config_templates(step.config, context)
        await try_legacy_connector_auto_resolve(
            resolved_cfg, context,
            step_name=step.name or "discord",
            type_name="webhook", subtype_name="discord",
            operation="send",
        )
        return f"Posted to Discord: {resolved_cfg.get('message', '')[:80]}"

    from aiorch.runtime.run_env import get_env as _get_env
    webhook_url = _cfg_resolve(step, "url", context) or _get_env(context, "DISCORD_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError("discord action requires 'url' in config or DISCORD_WEBHOOK_URL env var")

    # SSRF gate (Pass 3 #3 in the audit). Same class as Teams.
    from aiorch.core.http_safety import safe_http_url
    webhook_url = safe_http_url(webhook_url, purpose="discord connector")

    message = _cfg_resolve(step, "message", context)
    body_text = _cfg_resolve(step, "body", context)

    content = message
    if body_text:
        content += f"\n{body_text}"

    payload = {"content": content[:2000]}

    async def _send():
        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()
            return resp.status_code

    status = await _with_retry(_send, label="discord")
    logger.info("Discord message sent: %s", message[:60])
    return f"Posted to Discord ({status})"
