# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Action primitive runtime — outbound integrations for aiorch pipelines.

Each action has retry with exponential backoff and timeout.

Action-specific parameters are read from step.config (dict), not flat step fields.

YAML format:
    action: slack
    config:
      channel: "#alerts"
      message: "Pipeline done: {{report}}"

Supported actions:
  - slack         — post message via incoming webhook
  - webhook       — generic HTTP request to any endpoint
  - github-comment — comment on a PR via GitHub API
  - github-issue   — create a GitHub issue via API
  - write-file    — save output to local filesystem
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from aiorch.core import template
from aiorch.core.parser import Step

logger = logging.getLogger("aiorch.action")

# Default retry/timeout for all actions
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 30


def _cfg(step: Step, key: str, default: str = "") -> str:
    """Read a config value from step.config, with template resolution deferred."""
    return step.config.get(key, default) or default


def _cfg_resolve(step: Step, key: str, context: dict[str, Any], default: str = "") -> str:
    """Read a config value from step.config and resolve Jinja2 templates."""
    raw = step.config.get(key, default) or default
    if isinstance(raw, str) and raw:
        return template.resolve(raw, context)
    return raw


# ---------------------------------------------------------------------------
# Retry helper — exponential backoff
# ---------------------------------------------------------------------------

async def _with_retry(fn, retries=DEFAULT_RETRIES, timeout=DEFAULT_TIMEOUT, label="action"):
    """Execute fn with retry + timeout. Returns result or raises last error."""
    last_error = None
    for attempt in range(retries):
        try:
            return await asyncio.wait_for(fn(), timeout=timeout)
        except asyncio.TimeoutError:
            last_error = TimeoutError(f"{label} timed out after {timeout}s (attempt {attempt + 1}/{retries})")
            logger.warning("%s timeout (attempt %d/%d)", label, attempt + 1, retries)
        except Exception as e:
            last_error = e
            logger.warning("%s failed (attempt %d/%d): %s", label, attempt + 1, retries, e)
        if attempt < retries - 1:
            await asyncio.sleep(min(2 ** attempt, 10))  # 1s, 2s, 4s... max 10s
    raise last_error


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

async def execute_action(step: Step, context: dict[str, Any]) -> Any:
    """Execute an action step, dispatching to the correct handler."""
    from aiorch.runtime.registry import get_action, get_registered_actions
    from aiorch.constants import LOGGER_KEY

    action = step.action
    if action is None:
        raise ValueError("Step is not an action primitive (step.action is None)")

    handler = get_action(action)
    if handler is None:
        registered = ", ".join(sorted(get_registered_actions()))
        raise ValueError(
            f"Unknown action '{action}'. "
            f"Available: {registered}."
        )

    run_logger = context.get(LOGGER_KEY)
    if run_logger:
        # Redact secret-looking keys before logging the config preview.
        safe_cfg = {
            k: ("<redacted>" if any(s in k.lower() for s in ("token", "secret", "password", "key", "auth")) else v)
            for k, v in (step.config or {}).items()
        }
        run_logger.log(step.name, "DEBUG", f"Action dispatch: {action}", {
            "action": action, "config": safe_cfg,
        })

    result = await handler(step, context)

    if run_logger:
        result_preview = (
            str(result)[:600] if not isinstance(result, (dict, list)) else result
        )
        run_logger.log(step.name, "DEBUG", f"Action {action} completed", {
            "result_type": type(result).__name__,
            "result": result_preview,
        })

    return result


# ---------------------------------------------------------------------------
# Slack — post message via incoming webhook
# ---------------------------------------------------------------------------

async def _action_slack(step: Step, context: dict[str, Any]) -> str:
    """Post a message to Slack via incoming webhook.

    YAML:
        action: slack
        config:
          channel: "#alerts"
          message: "Alert: {{report}}"
          url: "${SLACK_WEBHOOK_URL}"     # or set env var

        # Managed-connector form (Platform only):
        action: slack
        config:
          uses: prod_slack
          message: "Alert"
    """
    # Auto-resolve: if the step names a managed connector via 'uses:',
    # route through the connector dispatcher so usage, metrics and
    # audit trail are captured. Falls through to legacy env-var path
    # if 'uses' is not set.
    if step.config.get("uses"):
        from aiorch.runtime.connectors import _resolve_config_templates
        from aiorch.connectors.integration import try_legacy_connector_auto_resolve
        resolved_cfg = _resolve_config_templates(step.config, context)
        result = await try_legacy_connector_auto_resolve(
            resolved_cfg, context,
            step_name=step.name or "slack",
            type_name="webhook", subtype_name="slack",
            operation="send",
        )
        target = f"#{resolved_cfg.get('channel', 'default')}"
        return f"Posted to {target}"

    channel = _cfg_resolve(step, "channel", context)
    message = _cfg_resolve(step, "message", context)

    from aiorch.core.http_safety import safe_http_url
    from aiorch.runtime.run_env import get_env
    webhook_url = _cfg_resolve(step, "url", context)
    if not webhook_url:
        webhook_url = get_env(context, "SLACK_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError(
            "Slack webhook URL required. Set 'url' in config or SLACK_WEBHOOK_URL env var. "
            "Create one at: https://api.slack.com/messaging/webhooks"
        )
    # SSRF gate — Slack's real URL is hooks.slack.com, any deviation
    # into private/loopback/IMDS ranges is almost certainly an attack.
    webhook_url = safe_http_url(webhook_url, purpose="slack action")

    payload: dict[str, Any] = {"text": message}
    if channel:
        payload["channel"] = channel

    async def _send():
        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()
            return resp.text

    await _with_retry(_send, label=f"slack({channel or 'default'})")
    target = f"#{channel}" if channel else "Slack"
    logger.info("Slack message sent to %s", target)
    return f"Posted to {target}"


# ---------------------------------------------------------------------------
# Webhook — generic HTTP request
# ---------------------------------------------------------------------------

async def _action_webhook(step: Step, context: dict[str, Any]) -> str:
    """Send an HTTP request to any endpoint.

    YAML:
        action: webhook
        config:
          url: "https://api.example.com/events"
          method: POST
          headers:
            Authorization: "Bearer ${TOKEN}"
            Content-Type: "application/json"
          body: '{"event":"pipeline_done","data":"{{output}}"}'

        # Managed-connector form (Platform only):
        action: webhook
        config:
          uses: prod_generic_webhook
          body: "{{output}}"
    """
    if step.config.get("uses"):
        from aiorch.runtime.connectors import _resolve_config_templates
        from aiorch.connectors.integration import try_legacy_connector_auto_resolve
        resolved_cfg = _resolve_config_templates(step.config, context)
        result = await try_legacy_connector_auto_resolve(
            resolved_cfg, context,
            step_name=step.name or "webhook",
            type_name="webhook", subtype_name="generic",
            operation="send",
        )
        return str(result) if result is not None else "ok"

    url = _cfg_resolve(step, "url", context)
    if not url:
        raise ValueError("webhook action requires 'url' in config")

    # SSRF gate + scheme allowlist. Blocks http(s) to private ranges
    # (AWS IMDS, localhost services, RFC1918) by default; operators
    # opt in via AIORCH_ALLOW_PRIVATE_HOSTS=1. Also blocks file://,
    # gopher://, dict://, etc.
    from aiorch.core.http_safety import safe_header_value, safe_http_url
    url = safe_http_url(url, purpose="webhook action")

    method = _cfg_resolve(step, "method", context, default="POST").upper()

    headers: dict[str, str] = {}
    raw_headers = step.config.get("headers")
    if isinstance(raw_headers, dict):
        # Each header value is template-rendered — validate that the
        # result contains no control characters (CR/LF/NUL) which
        # would enable request/response splitting.
        headers = {
            k: safe_header_value(
                template.resolve(v, context),
                name=str(k),
                purpose="webhook action",
            )
            for k, v in raw_headers.items()
        }

    body = _cfg_resolve(step, "body", context) or _cfg_resolve(step, "message", context)

    async def _send():
        async with httpx.AsyncClient() as client:
            resp = await client.request(method, url, headers=headers, content=body)
            resp.raise_for_status()
            return resp.text

    response_text = await _with_retry(_send, label=f"webhook({method} {url[:50]})")
    logger.info("Webhook %s %s → %d chars", method, url[:60], len(response_text))
    return response_text


# ---------------------------------------------------------------------------
# GitHub Comment — post comment on a PR via GitHub API
# ---------------------------------------------------------------------------

async def _action_github_comment(step: Step, context: dict[str, Any]) -> str:
    """Post a comment on a GitHub pull request via REST API.

    YAML:
        action: github-comment
        config:
          pr: "123"
          body: "Review: {{review}}"
          repo: "org/repo"          # or set GITHUB_REPOSITORY env var
    """
    pr = _cfg_resolve(step, "pr", context)
    body = _cfg_resolve(step, "body", context)

    if not pr:
        raise ValueError("github-comment requires 'pr' (PR number) in config")
    if not body:
        raise ValueError("github-comment requires 'body' (comment text) in config")

    from aiorch.runtime.run_env import get_env
    token = get_env(context, "GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN env var required for github-comment action")

    repo = _cfg_resolve(step, "repo", context)
    if not repo:
        repo = get_env(context, "GITHUB_REPOSITORY")
    if not repo:
        raise ValueError(
            "GitHub repository required. Set 'repo' in config "
            "(format: owner/repo) or GITHUB_REPOSITORY env var"
        )

    # Validate repo + pr before assembling the API URL. These
    # values are template-rendered and could otherwise contain
    # slashes, whitespace, or percent-encoded traversal.
    from aiorch.core.http_safety import safe_github_pr, safe_github_repo
    repo = safe_github_repo(repo, purpose="github-comment action")
    pr = safe_github_pr(pr, purpose="github-comment action")

    api_url = f"https://api.github.com/repos/{repo}/issues/{pr}/comments"

    async def _send():
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                api_url,
                json={"body": body},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            resp.raise_for_status()
            return resp.json()

    result = await _with_retry(_send, label=f"github-comment(PR#{pr})")
    comment_url = result.get("html_url", "")
    logger.info("GitHub comment posted on PR #%s: %s", pr, comment_url)
    return f"Commented on PR #{pr}: {comment_url}"


# ---------------------------------------------------------------------------
# GitHub Issue — create an issue via GitHub API
# ---------------------------------------------------------------------------

async def _action_github_issue(step: Step, context: dict[str, Any]) -> str:
    """Create a GitHub issue via REST API.

    YAML:
        action: github-issue
        config:
          title: "Bug found: {{report}}"
          body: "Details: {{details}}"
          repo: "org/repo"              # or set GITHUB_REPOSITORY env var
    """
    title = _cfg_resolve(step, "title", context) or _cfg_resolve(step, "message", context)
    body = _cfg_resolve(step, "body", context)

    if not title:
        raise ValueError("github-issue requires 'title' (issue title) in config")

    from aiorch.runtime.run_env import get_env
    token = get_env(context, "GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN env var required for github-issue action")

    repo = _cfg_resolve(step, "repo", context)
    if not repo:
        repo = get_env(context, "GITHUB_REPOSITORY")
    if not repo:
        raise ValueError("GITHUB_REPOSITORY env var required (format: owner/repo)")

    # Validate repo before assembling the API URL.
    from aiorch.core.http_safety import safe_github_repo
    repo = safe_github_repo(repo, purpose="github-issue action")

    api_url = f"https://api.github.com/repos/{repo}/issues"
    payload: dict[str, Any] = {"title": title}
    if body:
        payload["body"] = body

    async def _send():
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                api_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            resp.raise_for_status()
            return resp.json()

    result = await _with_retry(_send, label=f"github-issue({repo})")
    issue_url = result.get("html_url", "")
    issue_num = result.get("number", "?")
    logger.info("GitHub issue #%s created: %s", issue_num, issue_url)
    return f"Created issue #{issue_num}: {issue_url}"


# ---------------------------------------------------------------------------
# Write File — save output to local filesystem
# ---------------------------------------------------------------------------

async def _action_write_file(step: Step, context: dict[str, Any]) -> str:
    """Write step output as a downloadable artifact.

    YAML (unchanged surface):
        action: write-file
        config:
          path: "/reports/daily-{{date}}.md"
          body: "{{report}}"

    Deployment-aware semantics:

      - CLI mode (LocalArtifactStore): writes bytes to disk at the
        given path so existing CLI users see their files exactly
        where they expect them. ALSO records the artifact in the
        local store so the file is discoverable via `aiorch history`.

      - Platform mode (MinIOArtifactStore): writes bytes to MinIO
        under outputs/<pipeline>/<YYYY-MM-DD>/run-<run_id>/<uuid>.<ext>
        and creates a run_artifacts row linking this run to the new
        artifact with role='output'. The UI Trace page renders a
        download button for each output artifact. Nothing lands on
        the executor's local disk — safe for multi-node deployments.

    The `path:` field becomes the artifact's display name (what the
    user sees in the UI / history listing), not a filesystem location.
    It's still a string template, so `{{date}}` / `{{run_id}}` /
    `{{_meta.pipeline_name}}` etc. work as before.
    """
    file_path = _cfg_resolve(step, "path", context) or _cfg_resolve(step, "url", context)
    content = _cfg_resolve(step, "body", context) or _cfg_resolve(step, "message", context)

    if not file_path:
        raise ValueError("write-file requires 'path' (file path) in config")

    size = len(content)

    # Detect deployment mode via the artifact store implementation
    from aiorch.artifacts import Artifact, get_artifact_store, init_artifact_store
    from aiorch.artifacts.minio import MinIOArtifactStore

    try:
        artifact_store = get_artifact_store()
    except RuntimeError:
        # Executor / CLI may not have initialized yet during tests
        init_artifact_store()
        artifact_store = get_artifact_store()

    is_platform = isinstance(artifact_store, MinIOArtifactStore)

    # Common: figure out the content type from the path extension
    import mimetypes
    content_type = mimetypes.guess_type(file_path)[0] or "text/plain"

    # Encode content as bytes exactly once — both modes use the same bytes
    content_bytes = content.encode("utf-8") if isinstance(content, str) else content

    # Pull pipeline + run attribution from the execution context so the
    # MinIO layout slots the artifact under the right run folder
    from aiorch.constants import RUNTIME_META_KEY
    meta = context.get(RUNTIME_META_KEY, {}) if isinstance(context, dict) else {}
    pipeline_name = meta.get("pipeline_name") if isinstance(meta, dict) else None
    run_id_val = context.get("__run_id__") if isinstance(context, dict) else None
    workspace_id = context.get("__workspace_id__") if isinstance(context, dict) else None
    org_id = context.get("__org_id__") if isinstance(context, dict) else None

    if is_platform:
        # Platform: upload to MinIO, record run binding for the Trace UI
        if not workspace_id:
            raise RuntimeError(
                "write-file on Platform requires a workspace context. "
                "This run has no workspace_id — check the run record or "
                "executor startup."
            )

        artifact: Artifact = artifact_store.put(
            name=file_path,
            content=content_bytes,
            content_type=content_type,
            role="output",
            workspace_id=workspace_id,
            org_id=org_id,
            pipeline_name=pipeline_name,
            run_id=run_id_val,
        )

        if run_id_val is not None:
            artifact_store.record_run_binding(
                run_id=int(run_id_val),
                artifact_id=artifact.id,
                binding_name=step.name or file_path,
                role="output",
            )

        logger.info(
            "Wrote %d bytes as artifact %s (%s) from run %s",
            size, artifact.id[:8], file_path, run_id_val,
        )
        return f"Wrote {size} chars as artifact {artifact.id[:8]} ({file_path})"

    # CLI mode — local disk (as before) + record in local store for history.
    # Path confinement: the file_path is a template-rendered string
    # that may include untrusted context values. Confine to CWD (or
    # AIORCH_SAFE_ROOTS) so a malicious template can't escape the
    # sandbox. /dev/stdout etc. are accepted as symbolic sinks —
    # many example CLI pipelines use `path: /dev/stdout` to print
    # output to the terminal.
    from aiorch.core.paths import safe_path

    resolved_path = safe_path(
        file_path,
        purpose="write-file action target",
        default_root=Path.cwd(),
        allow_symbolic=True,
    )
    path = resolved_path

    if str(path) not in ("/dev/stdout", "/dev/stderr", "/dev/null"):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)

    # Also record the artifact in the local store so CLI users have
    # lineage + dedup + a record in `aiorch history` later.
    try:
        artifact = artifact_store.put(
            name=file_path,
            content=content_bytes,
            content_type=content_type,
            role="output",
            pipeline_name=pipeline_name,
            run_id=run_id_val,
        )
        if run_id_val is not None:
            artifact_store.record_run_binding(
                run_id=int(run_id_val),
                artifact_id=artifact.id,
                binding_name=step.name or file_path,
                role="output",
            )
    except Exception as exc:
        logger.debug("Local artifact registration skipped: %s", exc)

    logger.info("Wrote %d chars to %s", size, path)
    return f"Wrote {size} chars to {path}"
