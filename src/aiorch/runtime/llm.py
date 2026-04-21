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
                 provider_type: str = "openai", default_model: str | None = None):
        self._api_key = api_key
        self._base_url = base_url
        self._provider_type = provider_type
        self._default_model = default_model

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

        native_prefixes = ("openrouter/", "anthropic/", "gemini/", "cohere/", "ollama/", "bedrock/")
        if self._base_url and not resolved_model.startswith(native_prefixes):
            kwargs["api_base"] = self._base_url

        start = time.time()
        response = await litellm.acompletion(**kwargs, drop_params=True)
        duration_ms = (time.time() - start) * 1000

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        cost = self._get_cost(response, resolved_model, prompt_tokens, completion_tokens)

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

        clean_model = _strip_routing_prefix(resolved_model)

        try:
            from aiorch.metrics import observe_llm_request
            observe_llm_request(
                provider=self._provider_type, model=clean_model,
                duration=duration_ms / 1000,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                cost=cost,
            )
        except ImportError:
            pass

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
        if self._provider_type == "openrouter" or (self._base_url and "openrouter.ai" in self._base_url):
            if not model.startswith("openrouter/"):
                return f"openrouter/{model}"
            return model
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
        if hasattr(response, "_hidden_params"):
            hidden = response._hidden_params
            if isinstance(hidden, dict):
                resp_cost = hidden.get("response_cost")
                if resp_cost and resp_cost > 0:
                    return float(resp_cost)

        try:
            import litellm
            cost = litellm.completion_cost(completion_response=response)
            if cost and cost > 0:
                return cost
        except Exception:
            pass

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


LLM_CLIENT_KEY = "__llm_client__"
PROVIDER_NAME_KEY = "__provider_name__"


def get_llm_client(context: dict[str, Any] | None = None) -> LLMClient:
    """Get the LLM client. Returns the injected client if one was placed on
    the context (tests); otherwise builds one from aiorch.yaml + env."""
    if context and LLM_CLIENT_KEY in context:
        return context[LLM_CLIENT_KEY]

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
