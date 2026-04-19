# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""LLM client abstraction — LiteLLM-powered, provider-agnostic.

LiteLLM is the default backend for all LLM calls (100+ providers).
OpenAIClient is kept as a lightweight fallback for CLI/offline use.

Cost tracking uses litellm.completion_cost() for accurate per-model pricing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("aiorch.llm")

# Configure litellm globally at import time
try:
    import litellm as _litellm
    _litellm.drop_params = True           # Drop unsupported params per provider
    _litellm.suppress_debug_info = True    # Suppress noisy debug banners
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Fallback pricing — used only when litellm is unavailable
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-haiku-4-20250514": (0.80, 4.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
}

DEFAULT_INPUT_COST_PER_M = 0.15
DEFAULT_OUTPUT_COST_PER_M = 0.60


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Fallback cost estimation when litellm is unavailable."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        for name, (ic, oc) in MODEL_PRICING.items():
            if model.startswith(name) or name.startswith(model):
                pricing = (ic, oc)
                break
    ic, oc = pricing or (DEFAULT_INPUT_COST_PER_M, DEFAULT_OUTPUT_COST_PER_M)
    return (prompt_tokens * ic / 1_000_000) + (completion_tokens * oc / 1_000_000)


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

_ROUTING_PREFIXES = ("openrouter/", "anthropic/", "gemini/", "cohere/", "ollama/", "bedrock/")


def _strip_routing_prefix(model: str) -> str:
    """Remove litellm routing prefixes so the stored model name is clean."""
    for prefix in _ROUTING_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    content: str
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    tool_calls: list[Any] | None = None
    raw_message: Any = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# LiteLLM client (default) — 100+ providers, accurate cost tracking
# ---------------------------------------------------------------------------

class LitellmClient:
    """Default LLM client via litellm — supports all major providers.

    Cost is calculated using litellm.completion_cost() which has a
    comprehensive pricing database updated with each release.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 provider_type: str = "openai", default_model: str | None = None,
                 provider_id: str | None = None,
                 workspace_id: str | None = None):
        self._api_key = api_key
        self._base_url = base_url
        self._provider_type = provider_type
        self._default_model = default_model
        # Cost gate context (Phase A in the audit). When provider_id
        # is set, every call passes through aiorch.server.spend's
        # check_provider_budget + charge_provider helpers. When None
        # (single-user CLI mode), the gates are no-ops.
        self._provider_id = provider_id
        self._workspace_id = workspace_id

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        import litellm

        # Model priority: step YAML → provider default → error
        effective_model = model or self._default_model
        if not effective_model:
            raise RuntimeError(
                "No model configured. Either set 'model' in the pipeline YAML step, "
                "or assign a provider with a default model to this pipeline."
            )
        resolved_model = self._resolve_model(effective_model)

        # Cost gate pre-check (Phase A): refuse the call if the
        # provider is already at its daily or monthly cap. Pre-check
        # uses incoming_usd=0 because we don't know the actual cost
        # until the call returns. The post-flight charge below
        # enforces the cap for real and the *next* call after an
        # over-cap charge will be rejected here.
        if self._provider_id:
            from aiorch.runtime._hooks import pre_call_hook
            hook = pre_call_hook()
            if hook is not None:
                hook(self._provider_id, incoming_usd=0.0)

        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
        if self._api_key:
            kwargs["api_key"] = self._api_key

        # Only pass api_base for truly custom endpoints.
        # For known providers (openrouter/, anthropic/, etc.), litellm
        # handles routing natively via the model prefix — passing api_base
        # forces a different code path that breaks param handling.
        native_prefixes = ("openrouter/", "anthropic/", "gemini/", "cohere/", "ollama/", "bedrock/")
        if self._base_url and not resolved_model.startswith(native_prefixes):
            kwargs["api_base"] = self._base_url

        start = time.time()
        response = await litellm.acompletion(**kwargs, drop_params=True)
        duration_ms = (time.time() - start) * 1000

        # Extract tokens
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        # Accurate cost via litellm's pricing database
        cost = self._get_cost(response, resolved_model, prompt_tokens, completion_tokens)

        # Post-flight charge against the provider's cost gates
        # (Phase A). The call has already happened, so this records
        # the spend even if it pushes the provider over its cap —
        # the next call's pre-check will then reject. Matches
        # OpenAI's hard-limit semantics: the call that crosses the
        # line is the last one allowed.
        if self._provider_id and cost > 0:
            from aiorch.runtime._hooks import post_call_hook
            hook = post_call_hook()
            if hook is not None:
                hook(
                    self._provider_id,
                    cost,
                    workspace_id=self._workspace_id,
                )

        choice = response.choices[0]
        message = choice.message
        finish_reason = getattr(choice, "finish_reason", None)

        # Log finish_reason for debugging agent empty-response issues
        if finish_reason and finish_reason not in ("stop", "tool_calls", "end_turn"):
            logger.warning(
                "LLM returned finish_reason=%s (model=%s, tokens=%d+%d). "
                "Content empty=%s, tool_calls=%s",
                finish_reason, resolved_model,
                prompt_tokens, completion_tokens,
                not bool(message.content),
                bool(message.tool_calls),
            )

        tool_calls = None
        if message.tool_calls:
            tool_calls = message.tool_calls

        logger.debug(
            "LLM call: model=%s tokens=%d+%d cost=$%.6f dur=%.0fms finish=%s",
            resolved_model, prompt_tokens, completion_tokens, cost, duration_ms,
            finish_reason,
        )

        # Metrics: LLM request
        from aiorch.metrics import observe_llm_request
        clean_model = _strip_routing_prefix(resolved_model)
        observe_llm_request(
            provider=self._provider_type, model=clean_model,
            duration=duration_ms / 1000,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cost=cost,
        )

        return LLMResponse(
            content=message.content or "",
            cost=cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=clean_model,
            tool_calls=tool_calls,
            raw_message=message,
        )

    def _resolve_model(self, model: str) -> str:
        """Apply provider-specific model name conventions for litellm routing."""
        # OpenRouter — detected by provider_type or base_url
        if self._provider_type == "openrouter" or (self._base_url and "openrouter.ai" in self._base_url):
            if not model.startswith("openrouter/"):
                return f"openrouter/{model}"
            return model
        # Other providers — litellm uses prefixes for non-OpenAI models
        prefix_map = {
            "anthropic": "anthropic/",
            "google": "gemini/",
            "cohere": "cohere/",
            "ollama": "ollama/",
        }
        prefix = prefix_map.get(self._provider_type, "")
        if prefix and not model.startswith(prefix):
            return f"{prefix}{model}"
        return model

    @staticmethod
    def _get_cost(response, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Extract cost — tries litellm.completion_cost, falls back to estimate."""
        # Method 1: response._hidden_params (most reliable)
        if hasattr(response, "_hidden_params"):
            hidden = response._hidden_params
            if isinstance(hidden, dict):
                resp_cost = hidden.get("response_cost")
                if resp_cost and resp_cost > 0:
                    return float(resp_cost)

        # Method 2: litellm.completion_cost() — uses built-in pricing DB
        try:
            import litellm
            cost = litellm.completion_cost(completion_response=response)
            if cost and cost > 0:
                return cost
        except Exception:
            pass

        # Method 3: fallback to manual estimate
        if prompt_tokens or completion_tokens:
            return estimate_cost(model, prompt_tokens, completion_tokens)

        return 0.0


# ---------------------------------------------------------------------------
# OpenAI client (fallback for CLI without litellm)
# ---------------------------------------------------------------------------

class OpenAIClient:
    """Lightweight fallback using openai SDK directly."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        if not model:
            raise RuntimeError(
                "No model configured. Either set 'model' in the pipeline YAML step, "
                "or assign a provider with a default model to this pipeline."
            )

        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0 if response.usage else 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0 if response.usage else 0
        cost = estimate_cost(model, prompt_tokens, completion_tokens)

        # Use the model from the response if available (actual model used)
        actual_model = getattr(response, "model", model) or model
        return LLMResponse(
            content=message.content or "",
            cost=cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=actual_model,
            tool_calls=message.tool_calls if message.tool_calls else None,
            raw_message=message,
        )


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

LLM_CLIENT_KEY = "__llm_client__"
PROVIDER_ID_KEY = "__provider_id__"
WORKSPACE_ID_KEY = "__workspace_id__"


def _provider_cache_key(provider_id: str) -> str:
    from aiorch.cache import build_key
    return build_key("cache", "provider", provider_id)


def _get_provider_row_cached(store, provider_id: str) -> dict | None:
    """Fetch an active provider row with a 10-min Redis cache.

    The provider row contains ``api_key_encrypted`` — the ciphertext is
    safe in Redis because the decryption key lives in the app process
    (AIORCH_SECRET_KEY), not in Redis. Redis sees only what Postgres
    had anyway.

    Falls through to a direct Postgres query when Redis is unavailable
    or the cache layer errors out.
    """
    import json
    from aiorch.cache import get_redis

    client = get_redis()
    cache_key = _provider_cache_key(provider_id)

    if client is not None:
        try:
            cached = client.get(cache_key)
            if cached is not None:
                return json.loads(cached)
        except Exception:
            client = None  # fall through to DB

    row = store.query_one(
        "SELECT * FROM llm_providers WHERE id = ? AND is_active = TRUE",
        (provider_id,),
    )
    if row and client is not None:
        try:
            client.set(cache_key, json.dumps(dict(row), default=str), ex=600)
        except Exception:
            pass
    return row


def invalidate_provider_cache(provider_id: str) -> None:
    """Drop the cached provider row so other workers re-read from DB.

    Called by provider create/edit/delete endpoints so changes are
    immediately visible across API replicas (no 10-min stale window).
    """
    from aiorch.cache import get_redis
    client = get_redis()
    if client is None:
        return
    try:
        client.delete(_provider_cache_key(provider_id))
    except Exception:
        pass


def _resolve_managed_provider(
    provider_id: str | None = None,
    workspace_id: str | None = None,
    org_id: str | None = None,
) -> dict[str, Any] | None:
    """Resolve an LLM provider from the database.

    Resolution chain:
        1. Explicit provider_id (scope-checked against caller's org/workspace)
        2. Workspace default
        3. Org default

    Tenant isolation (Bugs #2/#3 in the security audit):
      - Explicit provider_id: the returned row's scope is verified
        against the caller's workspace/org. A workspace-scoped
        provider must match the caller's workspace exactly; an org-
        wide provider must match the caller's org. A workspace from
        another tenant cannot "borrow" a foreign provider by UUID.

      - Default resolution: workspace-default lookups include an
        org_id filter, so an attacker who plants a row with
        ``(attacker_org_id, victim_workspace_id, is_default=True)``
        is not returned when the victim workspace runs.
    """
    try:
        from aiorch.storage import get_store
        store = get_store()
    except Exception:
        return None

    # Resolve the caller's org_id up-front so every branch can use it
    # for scope checks. This is cheap (one lookup) and avoids duplicate
    # work further down.
    resolved_org_id = org_id
    if not resolved_org_id and workspace_id:
        ws_row = store.query_one("SELECT org_id FROM workspaces WHERE id = ?", (workspace_id,))
        if ws_row:
            resolved_org_id = ws_row["org_id"]

    if provider_id:
        row = _get_provider_row_cached(store, provider_id)
        if row:
            row_ws = row.get("workspace_id")
            row_org = row.get("org_id")
            if row_ws:
                # Workspace-scoped provider: must match caller workspace.
                if workspace_id and row_ws != workspace_id:
                    logger.warning(
                        "provider %s is workspace-scoped to %s, not %s — refusing",
                        provider_id, row_ws, workspace_id,
                    )
                    return None
            else:
                # Org-wide provider: must match caller org.
                if resolved_org_id and row_org != resolved_org_id:
                    logger.warning(
                        "provider %s is org-wide for %s, not %s — refusing",
                        provider_id, row_org, resolved_org_id,
                    )
                    return None
            return row

    if not workspace_id and not resolved_org_id:
        return None

    if workspace_id:
        # Workspace default — require org_id match as well so a row
        # with (attacker_org, victim_workspace) is never returned.
        if resolved_org_id:
            row = store.query_one(
                "SELECT * FROM llm_providers "
                "WHERE workspace_id = ? AND org_id = ? "
                "AND is_default = TRUE AND is_active = TRUE",
                (workspace_id, resolved_org_id),
            )
        else:
            row = store.query_one(
                "SELECT * FROM llm_providers "
                "WHERE workspace_id = ? AND is_default = TRUE AND is_active = TRUE",
                (workspace_id,),
            )
        if row:
            return row

    if resolved_org_id:
        row = store.query_one(
            "SELECT * FROM llm_providers WHERE org_id = ? AND workspace_id IS NULL AND is_default = TRUE AND is_active = TRUE",
            (resolved_org_id,),
        )
        if row:
            return row

    return None


def _client_from_provider(provider: dict[str, Any]) -> LLMClient:
    """Create an LLM client from a managed provider record.

    Always uses LiteLLM for accurate cost tracking and 100+ provider support.
    Falls back to OpenAI SDK only if litellm import fails.

    SSRF gate (Round 2 #15 in the audit, defense-in-depth): the
    base_url stored on the provider row is re-validated here before
    the client uses it. The API boundary already validates on
    create/update, but this catches rows that arrive via migration,
    direct SQL, or any future code path that bypasses the route.
    """
    encrypted_key = provider.get("api_key_enc", "")
    try:
        from aiorch.runtime._hooks import provider_key_decryptor
        decryptor = provider_key_decryptor()
        if decryptor is not None and encrypted_key:
            api_key = decryptor(encrypted_key)
        else:
            # CLI mode or no ciphertext: treat the stored value as
            # already-plaintext (env-var-resolved or literal key).
            api_key = encrypted_key
    except Exception:
        api_key = encrypted_key

    base_url = provider.get("base_url")
    if base_url:
        from aiorch.core.http_safety import HttpSafetyError, safe_http_url
        try:
            base_url = safe_http_url(base_url, purpose="provider base_url (runtime)")
        except HttpSafetyError as exc:
            logger.warning(
                "Refusing provider %s with unsafe base_url %r: %s",
                provider.get("id"), base_url, exc,
            )
            raise

    provider_type = provider.get("provider_type", "openai")

    default_model = provider.get("default_model")
    # Cost gates wiring (Phase A): the client needs to know which
    # provider row it represents so the post-flight charge can hit
    # the right llm_provider_spend row. Workspace is denormalized
    # for filtering on the spend rollup endpoint.
    provider_id = provider.get("id")
    provider_workspace = provider.get("workspace_id")

    try:
        import litellm  # noqa: F401
        return LitellmClient(
            api_key=api_key, base_url=base_url,
            provider_type=provider_type, default_model=default_model,
            provider_id=provider_id, workspace_id=provider_workspace,
        )
    except ImportError:
        return OpenAIClient(api_key=api_key, base_url=base_url)


PROVIDER_NAME_KEY = "__provider_name__"


def get_llm_client(context: dict[str, Any] | None = None) -> LLMClient:
    """Get the LLM client with provider resolution.

    Priority:
        1. context[LLM_CLIENT_KEY] — injected client (tests)
        2. Managed provider from DB (by provider_id, workspace, or org)
        3. Config/env fallback (OPENAI_API_KEY)

    Side effect: stores resolved provider name in context[PROVIDER_NAME_KEY]
    so it can be captured in step metadata for cost tracking.
    """
    # Level 1: injected
    if context and LLM_CLIENT_KEY in context:
        return context[LLM_CLIENT_KEY]

    # Level 2: managed provider from DB
    if context:
        provider = _resolve_managed_provider(
            context.get(PROVIDER_ID_KEY),
            context.get(WORKSPACE_ID_KEY),
            context.get("__org_id__"),
        )
        if provider:
            context[PROVIDER_NAME_KEY] = provider.get("name", provider.get("provider_type", "managed"))
            return _client_from_provider(provider)

    # Level 3: config/env fallback
    if context:
        context[PROVIDER_NAME_KEY] = "env"

    from aiorch.core.config import get_config
    cfg = get_config().llm

    try:
        import litellm  # noqa: F401
        return LitellmClient(
            api_key=cfg.api_key, base_url=cfg.base_url,
            provider_type=cfg.provider, default_model=cfg.model,
        )
    except ImportError:
        return OpenAIClient(api_key=cfg.api_key, base_url=cfg.base_url)
