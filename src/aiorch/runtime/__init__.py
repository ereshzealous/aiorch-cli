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

"""Step runtime — executes each primitive type with cost tracking and caching.

Responsibilities are split into focused modules:
  - condition.py  — condition evaluation
  - foreach.py    — foreach loop orchestration
  - output.py     — result unwrapping, cost accumulation, save-to-file
  - registry.py   — primitive and action dispatch registries
  - llm.py        — LLM client protocol and litellm implementation
"""

from __future__ import annotations

import asyncio
from typing import Any

from aiorch.core.parser import Step
from aiorch.runtime.prompt import execute_prompt, execute_prompt_streaming
from aiorch.runtime.condition import eval_condition
from aiorch.runtime.foreach import resolve_foreach_items, run_foreach
from aiorch.runtime.output import unwrap_result, save_to_file
from aiorch.core import template
from aiorch.core.utils import resolve_input as _resolve_input_shared


# Re-export context keys from constants (used by cli.py, main.py)
from aiorch.constants import COST_KEY, META_KEY, LOGGER_KEY, CONFIG_KEY


def _get_config(context: dict[str, Any] | None = None):
    """Get config from context (preferred) or global singleton (fallback)."""
    if context and CONFIG_KEY in context:
        return context[CONFIG_KEY]
    from aiorch.core.config import get_config
    return get_config()


class StepSkipped(Exception):
    """Sentinel raised when a step's `condition:` evaluates false.

    Not an error — it's control flow. The DAG runner catches it and
    emits a ``step_skipped`` event instead of ``step_done`` so the Trace
    page can render a "skipped" pill distinct from "ran and returned
    empty". Previously the runtime returned ``None`` on skip and the
    DAG runner treated it as a successful empty run, which made it
    impossible to distinguish the two cases in the UI.
    """
    def __init__(self, step_name: str, condition_source: str, resolved: str):
        self.step_name = step_name
        self.condition_source = condition_source
        self.resolved = resolved
        super().__init__(
            f"Step '{step_name}' skipped by condition "
            f"(resolved: {resolved!r})"
        )


async def execute_step(step: Step, context: dict[str, Any]) -> Any:
    """Execute a single step, dispatching to the correct runtime.

    Orchestrates: vars → condition → input → foreach → dispatch → save.
    Each concern is delegated to a focused module.
    """
    # Resolve vars into context
    merged = {**context, **step.vars}

    if COST_KEY not in context:
        context[COST_KEY] = {}
    merged[COST_KEY] = context[COST_KEY]

    if META_KEY not in context:
        context[META_KEY] = {}
    merged[META_KEY] = context[META_KEY]

    from aiorch.constants import RUNTIME_META_KEY
    if RUNTIME_META_KEY in context:
        merged[RUNTIME_META_KEY] = context[RUNTIME_META_KEY]

    if step.condition:
        try:
            resolved_cond = template.resolve(step.condition, merged)
        except Exception as e:
            raise RuntimeError(
                f"Step '{step.name}': failed to resolve condition '{step.condition}'\n"
                f"Available variables: {[k for k in merged if not k.startswith('__')]}\n"
                f"Error: {e}"
            )
        if not eval_condition(resolved_cond):
            raise StepSkipped(step.name, step.condition, resolved_cond)

    try:
        resolved_input = _resolve_input(step.input, merged)
    except Exception as e:
        raise RuntimeError(
            f"Step '{step.name}': failed to resolve input\n"
            f"Available variables: {[k for k in merged if not k.startswith('__')]}\n"
            f"Error: {e}"
        )

    # Handle foreach
    if step.foreach is not None:
        items = resolve_foreach_items(step.foreach, merged)

        async def _run_one(item_ctx):
            return await _dispatch(step, item_ctx, resolved_input)

        return await run_foreach(
            items, merged, _run_one,
            parallel=step.parallel is True,
            timeout=_parse_timeout(step.timeout),
            step_name=step.name,
            skip_on_error=step.skip_on_error,
        )

    step_timeout = _parse_timeout(step.timeout)
    if step_timeout is not None:
        try:
            result = await asyncio.wait_for(
                _dispatch(step, merged, resolved_input),
                timeout=step_timeout,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Step '{step.name}' exceeded timeout={step.timeout}. "
                f"MCP sessions (if any) recover via 5-minute idle pool TTL. "
                f"Likely causes: LLM reasoning stall, slow upstream service, "
                f"or timeout set too tight. Inspect executor/registry logs."
            )
    else:
        result = await _dispatch(step, merged, resolved_input)

    if step.save and result is not None:
        save_to_file(result, step.save, merged)

    return result


async def _dispatch(step: Step, context: dict[str, Any], resolved_input: str | None) -> Any:
    """Dispatch to the correct primitive runtime via the registry."""
    from aiorch.runtime.registry import get_primitive, get_registered_primitives

    ptype = step.primitive_type

    # Prompt is special — it needs resolved_input and caching logic
    if ptype == "prompt":
        return await _dispatch_prompt(step, context, resolved_input)

    spec = get_primitive(ptype)
    if spec is None:
        registered = ", ".join(get_registered_primitives())
        raise ValueError(
            f"Unknown primitive: '{ptype}'. "
            f"Registered: {registered}. "
            f"Use register_primitive() to add new types."
        )
    return await spec.handler(step, context)


async def _dispatch_prompt(step: Step, context: dict[str, Any], resolved_input: str | None) -> Any:
    """Execute a prompt step with optional caching and output validation."""
    from aiorch.storage import cache_key, cache_get, cache_put

    cfg = _get_config(context).llm
    prompt_text = template.resolve(step.prompt, context)
    if resolved_input:
        prompt_text = f"{prompt_text}\n\n{resolved_input}"

    model = _resolve_step_model(step, context)  # CLI --model > step YAML (Jinja) > provider default
    system = step.system
    temperature = step.temperature if step.temperature is not None else cfg.temperature
    max_tokens = step.max_tokens or cfg.max_tokens

    logger = context.get(LOGGER_KEY)

    cache_model = model or "provider-default"
    if step.cache:
        key = cache_key(prompt_text, cache_model, system, temperature)
        cached = cache_get(key)
        if cached is not None:
            if logger:
                logger.log(step.name, "INFO", "Cache hit — skipped LLM call")
            return cached["response"]

    needs_validation = bool(step.output_schema or step.assertions)
    max_attempts = 1 + step.retry_on_invalid if needs_validation else 1
    current_prompt = prompt_text
    result = None

    for attempt in range(max_attempts):
        if logger:
            logger.log(step.name, "DEBUG", "Prompt sent to LLM", {
                "model": model,
                "system": system[:200] if system else None,
                "prompt": current_prompt[:1000],
                "temperature": temperature,
                "attempt": attempt + 1 if needs_validation else None,
            })

        try:
            run_id = None
            if run_id and step.name:
                result = None
            else:
                result = await execute_prompt(
                    prompt=current_prompt,
                    model=model,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    output_format=step.format.value,
                    context=context,
                )
        except RuntimeError as e:
            # Handle JSON parse failures with retry
            if needs_validation and "invalid JSON" in str(e) and attempt < max_attempts - 1:
                from aiorch.runtime.validation import build_retry_prompt
                if logger:
                    logger.log(step.name, "WARNING", f"JSON parse failed (attempt {attempt + 1}), retrying")
                current_prompt = build_retry_prompt(prompt_text, str(e), ["Output is not valid JSON"])
                continue
            raise

        # Log the response
        if logger:
            logger.log(step.name, "DEBUG", "LLM response received", {
                "model": result.model,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "cost": result.cost,
                "response": str(result.content)[:1000],
            })

        # Validate output if schema or assertions are configured
        if needs_validation:
            from aiorch.runtime.validation import validate_output, build_retry_prompt
            errors = validate_output(result.content, step.output_schema, step.assertions, context)
            if errors:
                if attempt < max_attempts - 1:
                    if logger:
                        logger.log(step.name, "WARNING",
                                   f"Validation failed (attempt {attempt + 1}/{max_attempts}), retrying",
                                   {"errors": errors})
                    current_prompt = build_retry_prompt(
                        prompt_text, str(result.content), errors
                    )
                    continue
                else:
                    error_text = "\n".join(f"  - {e}" for e in errors)
                    raise RuntimeError(
                        f"Step '{step.name}': output validation failed "
                        f"after {max_attempts} attempt(s):\n{error_text}"
                    )

        break  # valid or no validation needed

    # Store in cache (only valid results)
    if step.cache:
        key = cache_key(prompt_text, cache_model, system, temperature)
        cache_put(key, cache_model, result.content, result.cost)

    return _unwrap_result(result, context, step.name, max_cost=step.max_cost)


def _unwrap_result(result: Any, context: dict[str, Any], step_name: str, max_cost: float | None = None) -> Any:
    """Unwrap PromptResult. Delegates to output module."""
    return unwrap_result(result, context, step_name, max_cost=max_cost)


def _resolve_input(
    input_val: str | list[str] | None, context: dict[str, Any]
) -> str | None:
    """Resolve step input to a string. Delegates to shared utility."""
    return _resolve_input_shared(input_val, context)


def _eval_condition(condition: str) -> bool:
    """Evaluate a simple condition string. Delegates to condition module."""
    return eval_condition(condition)


def _parse_timeout(timeout_str: str | None) -> float | None:
    """Parse a timeout string like '30s', '2m', '120' into seconds."""
    from aiorch.core.utils import parse_duration
    return parse_duration(timeout_str)


def _resolve_step_model(step: Step, context: dict[str, Any]) -> str | None:
    """Resolve the effective model for a step.

    Precedence (first non-empty wins):
      1. ``__model_override__`` in context — set by CLI ``--model`` flag.
      2. ``step.model`` with Jinja substitution against context — lets
         pipelines accept a ``model`` input and pass it through as
         ``model: "{{ model }}"`` without hardcoding.
      3. Provider default (indicated by returning ``None`` so the LLM
         layer applies its configured fallback).

    Empty strings (e.g. the common case where a pipeline declares a
    ``model`` input with ``default: ""``) collapse to ``None`` so the
    provider default kicks in — this is what lets "leave blank to use
    the workspace provider" work cleanly.
    """
    override = context.get("__model_override__") if context else None
    if override:
        return override
    raw = step.model
    if not raw:
        return None
    try:
        resolved = template.resolve(raw, context)
    except Exception:
        resolved = raw
    resolved = str(resolved).strip() if resolved else ""
    if not resolved:
        return None
    if "{{" in resolved or "}}" in resolved:
        return None
    return resolved
