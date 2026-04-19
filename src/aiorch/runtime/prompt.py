# Copyright 2026 Eresh Gorantla
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM prompt execution runtime via the LLMClient protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class PromptResult:
    """Result from an LLM call with cost and token metadata."""
    content: Any
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "" 


async def execute_prompt(
    prompt: str,
    model: str = "gpt-4o-mini",
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    output_format: str = "text",
    context: dict[str, Any] | None = None,
) -> PromptResult:
    """Send a prompt to an LLM and return the response with cost data.

    Uses the LLMClient protocol — works with OpenAI, Anthropic, or any
    compatible provider. No direct litellm dependency.
    """
    from aiorch.runtime.llm import get_llm_client

    messages: list[dict[str, Any]] = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    client = get_llm_client(context)

    # Track whether model was explicitly set or inherited from provider default
    if context is not None:
        context["__model_source__"] = "explicit" if model else "provider-default"

    try:
        llm_response = await client.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens, 
        )
    except Exception as e:
        raise _wrap_llm_error(e, model)

    content = llm_response.content

    if output_format == "json":
        try:
            content = _parse_json_output(content)
        except Exception:
            raise RuntimeError(
                f"LLM returned invalid JSON. Model: {model}\n"
                f"Raw output:\n{content[:500]}"
            )

    return PromptResult(
        content=content,
        cost=llm_response.cost,
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
        model=llm_response.model or model,  # actual model from response, fallback to requested
    )


async def execute_prompt_streaming(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    output_format: str = "text",
    context: dict[str, Any] | None = None,
    on_chunk: Any = None,
) -> PromptResult:
    """Stream a prompt to an LLM, calling on_chunk(partial_text) as tokens arrive.

    Resolves the managed provider (workspace/org default) via get_llm_client()
    so api_key, base_url, and default_model all work the same as the
    non-streaming path. Falls back to non-streaming if litellm is missing.
    """
    import time
    from aiorch.runtime.llm import get_llm_client, LitellmClient, _strip_routing_prefix

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if context is not None:
        context["__model_source__"] = "explicit" if model else "provider-default"

    # Resolve the provider via the same path the non-streaming call uses.
    # This gives us the decrypted api_key, base_url, and default_model from
    # the workspace/org default provider (if any).
    client = get_llm_client(context)
    effective_model = model or getattr(client, "_default_model", None)
    if not effective_model:
        raise RuntimeError(
            "No model configured. Either set 'model' in the pipeline YAML step, "
            "or assign a provider with a default model to this workspace/org."
        )

    api_key = getattr(client, "_api_key", None)
    base_url = getattr(client, "_base_url", None)

    try:
        import litellm
        start = time.time()

        # Mirror LitellmClient.complete()'s routing: only pass api_base for
        # truly custom endpoints; native prefixes (openrouter/, anthropic/, etc.)
        # are handled by litellm directly via the model prefix.
        resolved_model = effective_model
        if isinstance(client, LitellmClient):
            resolved_model = client._resolve_model(effective_model)

        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "stream": True,
            "drop_params": True,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if api_key:
            kwargs["api_key"] = api_key
        native_prefixes = ("openrouter/", "anthropic/", "gemini/", "cohere/", "ollama/", "bedrock/")
        if base_url and not resolved_model.startswith(native_prefixes):
            kwargs["api_base"] = base_url

        # Keep the local `model` variable in sync for error messages / cost calcs
        model = resolved_model

        response = await litellm.acompletion(**kwargs)

        chunks = []
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                chunks.append(delta.content)
                partial = "".join(chunks)
                if on_chunk:
                    on_chunk(partial)

            # Token usage from final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

        content = "".join(chunks)
        duration_ms = (time.time() - start) * 1000

        # Estimate cost from collected token counts. We deliberately do
        # NOT call litellm.stream_chunk_builder(response) here — by this
        # point the CustomStreamWrapper is exhausted and stream_chunk_builder
        # raises TypeError (subscriptable) which litellm logs at ERROR level
        # even though we'd silently swallow it. Our per-model rate table is
        # accurate to a few percent and produces clean logs.
        from aiorch.runtime.llm import estimate_cost
        clean_model = _strip_routing_prefix(model)
        cost = estimate_cost(clean_model, prompt_tokens, completion_tokens)

        if output_format == "json":
            try:
                content = _parse_json_output(content)
            except Exception:
                raise RuntimeError(
                    f"LLM returned invalid JSON. Model: {model}\n"
                    f"Raw output:\n{content[:500]}"
                )

        return PromptResult(
            content=content,
            cost=cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=clean_model,
        )

    except ImportError:
        # No litellm — fall back to non-streaming
        return await execute_prompt(
            prompt=prompt, model=model, system=system,
            temperature=temperature, max_tokens=max_tokens,
            output_format=output_format, context=context,
        )
    except Exception as e:
        raise _wrap_llm_error(e, model)


def _parse_json_output(content: str) -> Any:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def _wrap_llm_error(error: Exception, model: str) -> RuntimeError:
    """Wrap LLM errors with helpful, actionable messages."""
    err_str = str(error).lower()
    err_type = type(error).__name__

    if "auth" in err_str or "api key" in err_str or "401" in err_str or "credentials" in err_str:
        return RuntimeError(
            f"Authentication failed for model '{model}'.\n"
            f"Fix: Set your API key in aiorch.yaml or as an environment variable.\n"
            f"  export OPENAI_API_KEY=sk-...\n"
            f"  export ANTHROPIC_API_KEY=sk-ant-...\n"
            f"Run 'aiorch doctor' to check your setup."
        )

    if "not found" in err_str or "does not exist" in err_str or "invalid model" in err_str:
        return RuntimeError(
            f"Model '{model}' not found or not available.\n"
            f"Fix: Check the model name in your step or aiorch.yaml.\n"
            f"Common models: gpt-4o-mini, claude-sonnet-4-20250514, gemini-2.5-flash"
        )

    if "rate limit" in err_str or "429" in err_str or "quota" in err_str:
        return RuntimeError(
            f"Rate limited by provider for model '{model}'.\n"
            f"Fix: Wait a moment and retry, or use a different model/provider."
        )

    if "timeout" in err_str or "timed out" in err_str:
        return RuntimeError(
            f"LLM call timed out for model '{model}'.\n"
            f"Fix: Try a shorter prompt, or increase timeout in your step config."
        )

    if "connection" in err_str or "network" in err_str:
        return RuntimeError(
            f"Cannot connect to LLM provider for model '{model}'.\n"
            f"Fix: Check your internet connection and base_url in aiorch.yaml."
        )

    # Fallback — still better than raw error
    return RuntimeError(
        f"LLM call failed for model '{model}': {err_type}\n"
        f"Details: {str(error)[:300]}"
    )
