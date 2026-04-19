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

"""Agent primitive runtime — autonomous multi-step LLM loop with tools."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("aiorch.agent")

import yaml

from aiorch.constants import CONFIG_KEY
from aiorch.core import template
from aiorch.core.parser import Step
from aiorch.core.utils import resolve_input
from aiorch.runtime.llm import get_llm_client
from aiorch.runtime.prompt import PromptResult
from aiorch.tools import resolve_tools, tools_to_openai_schema, execute_tool


# Built-in agents loaded from prompts module
from aiorch.prompts import BUILTIN_AGENTS
from aiorch.constants import DEFAULT_AGENT_SYSTEM_PROMPT


def _is_mcp_tool_error(tool_output: str) -> bool:
    """True when a tool's output is our internal error-string sentinel.

    The MCP tool wrappers in ``aiorch.tools`` return strings prefixed with
    ``[MCP Error]`` / ``[MCP Registry Error]`` / ``[ERROR]`` when a tool
    call fails. The agent loop uses this to count failed tool calls and
    fail the step cleanly when every tool invocation errored.
    """
    if not isinstance(tool_output, str):
        return False
    head = tool_output.lstrip()[:32]
    return (
        head.startswith("[MCP Error]")
        or head.startswith("[MCP Registry Error]")
        or head.startswith("[ERROR]")
    )


# Phrases the LLM typically emits when it received error text as tool
# output and parroted it back as the final answer. Matched
# case-insensitively. Conservative list — only consulted when every
# tool call already errored, so a legitimate final answer that happens
# to mention "error" in passing is still safe.
_ERROR_PROPAGATION_MARKERS = (
    "[mcp error]",
    "[mcp registry error]",
    "tool call failed",
    "could not be fetched",
    "could not be accessed",
    "url not fetched",
    "url fetching error",
    "failed registry request",
    "mcp registry request",
    "session destroyed",
    "readtimeout",
    "connecterror",
)


def _looks_like_error_propagation(content: str) -> bool:
    """True when the agent's final answer appears to just restate the
    tool error rather than produce real content.

    Only used when every tool call already errored. Checks the first
    400 chars so a buried mention of "error" in a long essay doesn't
    trip it.
    """
    if not isinstance(content, str):
        return True
    head = content[:400].lower()
    return any(marker in head for marker in _ERROR_PROPAGATION_MARKERS)


# ---------------------------------------------------------------------------
# Misplaced-key detection (silent schema-validation gap)
# ---------------------------------------------------------------------------

# Keys the runtime reads from the agent dict itself. Anything else inside
# an agent dict is silently dropped because the Step schema declares
# `agent: str | dict | None` without structural validation of the dict —
# Pydantic doesn't recurse, so typos and wrong nesting slip through.
_VALID_AGENT_DICT_KEYS = {"name", "system", "constraints", "tools"}

# Keys that look like agent config but are actually STEP-level fields.
# Nesting one of these under `agent:` drops it silently — the runtime
# never reads it, and the agent runs with defaults (no MCP, "Begin."
# user prompt, workspace default model). This list is kept short on
# purpose — it covers the real footguns we've observed in practice.
_MISPLACED_STEP_KEYS = {
    "mcp", "goal", "max_iterations", "model", "temperature",
    "max_tokens", "output", "depends", "condition", "retry",
    "retry_delay", "timeout", "foreach", "cache", "on_failure",
    "input", "vars", "format", "save", "schema", "assertions",
    "retry_on_invalid", "max_cost", "trigger", "secrets", "parallel",
}


def _warn_misplaced_agent_keys(step: Step, agent_def: dict[str, Any]) -> None:
    """Log a warning when step-level keys are nested inside agent:.

    The Step.agent field is `str | dict | None`. When a dict is provided,
    Pydantic accepts any keys without validation. The runtime only reads
    system/constraints/tools from that dict — anything else is silently
    dropped. That's a sharp edge: pipelines with ``mcp:`` or ``goal:``
    nested inside ``agent:`` run with no MCP tools and a fallback
    "Begin." prompt, giving users a confusing LLM response like
    "please provide a URL" with no surface hint at the actual cause.

    This helper surfaces the problem at run time so users catch it
    immediately instead of debugging through LLM output.
    """
    if not isinstance(agent_def, dict):
        return
    keys = set(agent_def.keys())

    # Misplaced step-level fields — known problematic
    misplaced = sorted(keys & _MISPLACED_STEP_KEYS)
    if misplaced:
        logger.warning(
            "Step %r: keys %s are nested inside `agent:` but they belong at "
            "the STEP level. They will be IGNORED. Move them out one indent "
            "so they sit next to `agent:`, not under it. Correct shape:\n"
            "  step_name:\n"
            "    %s  # step level\n"
            "    agent:\n"
            "      system: ...       # only system/constraints/tools belong here",
            step.name or "<unnamed>",
            misplaced,
            "\n    ".join(f"{k}: ..." for k in misplaced),
        )

    # Unknown keys inside the agent dict — mild warning so typos don't
    # silently get dropped. Exclude the known-valid keys and the
    # already-warned misplaced keys (to avoid duplicate noise).
    unknown = sorted(keys - _VALID_AGENT_DICT_KEYS - _MISPLACED_STEP_KEYS)
    if unknown:
        logger.warning(
            "Step %r: keys %s inside `agent:` are not recognized and will "
            "be ignored. Valid agent-dict keys: %s.",
            step.name or "<unnamed>",
            unknown,
            sorted(_VALID_AGENT_DICT_KEYS),
        )


# ---------------------------------------------------------------------------
# Agent definition resolution
# ---------------------------------------------------------------------------

def _resolve_agent_def(step: Step) -> dict[str, Any]:
    """Resolve the agent definition from step.agent.

    Supports:
      - Built-in name (string matching BUILTIN_AGENTS)
      - YAML file path (string ending in .yaml)
      - Inline dict (keys: name, system, tools)
      - Generic goal string (treated as a generic agent)
    """
    agent = step.agent

    if isinstance(agent, dict):
        return agent

    if isinstance(agent, str):
        # Built-in agent?
        if agent in BUILTIN_AGENTS:
            return BUILTIN_AGENTS[agent]

        # YAML file?
        if agent.endswith(".yaml"):
            with open(agent, "r") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Agent file must be a YAML mapping: {agent}")
            return data

        # Generic agent — treat the string as the goal
        return {
            "name": "generic-agent",
            "system": DEFAULT_AGENT_SYSTEM_PROMPT,
            "tools": [],
        }

    raise ValueError(f"Invalid agent spec: {agent!r}")


def _build_system_prompt(agent_def: dict[str, Any], step: Step) -> str:
    """Build the full system prompt from agent definition and step overrides."""
    parts: list[str] = []

    # Agent's base system prompt
    if agent_def.get("system"):
        parts.append(agent_def["system"])

    # Step-level system override/addition
    if step.system:
        parts.append(step.system)

    # Agent constraints
    if agent_def.get("constraints"):
        constraints = agent_def["constraints"]
        if isinstance(constraints, list):
            parts.append("Constraints:\n" + "\n".join(f"- {c}" for c in constraints))
        elif isinstance(constraints, str):
            parts.append(f"Constraints: {constraints}")

    # Goal
    if step.goal:
        parts.append(f"Goal: {step.goal}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def execute_agent(step: Step, context: dict[str, Any]) -> Any:
    """Execute an agent step — autonomous multi-step LLM loop with tools.

    Uses the LLMClient protocol (runtime/llm.py) for all LLM calls,
    ensuring provider abstraction is consistent across prompt and agent paths.

    The agent repeatedly calls the LLM, executes any requested tool calls,
    and feeds results back until the LLM produces a final text response
    or the iteration limit is reached.
    """
    if context and CONFIG_KEY in context:
        cfg = context[CONFIG_KEY].llm
    else:
        from aiorch.core.config import get_config
        cfg = get_config().llm

    # Get LLM client — same abstraction used by prompt runtime
    client = get_llm_client(context)

    # Resolve agent definition
    agent_def = _resolve_agent_def(step)

    # Warn when step-level fields are nested inside agent: — the parser
    # accepts this silently (agent dict is untyped so Pydantic can't
    # validate it), but the runtime only reads system/constraints/tools
    # from the agent dict. Keys like mcp/goal/max_iterations/model sit
    # at the STEP level — nesting them under agent: drops them with no
    # indication, and the agent then runs with no tools and a "Begin."
    # fallback prompt. See the Step schema in src/aiorch/core/parser.py.
    _warn_misplaced_agent_keys(step, agent_def)

    # Resolve tools — step-level tools override agent-level tools
    tool_specs = step.tools or agent_def.get("tools", [])
    tools = resolve_tools(tool_specs) if tool_specs else []

    # Resolve MCP tools if specified
    mcp_sessions = []
    mcp_config = None
    if isinstance(tool_specs, list):
        # Check for mcp key in dict-style tool specs
        for spec in tool_specs:
            if isinstance(spec, dict) and "mcp" in spec:
                mcp_config = spec["mcp"]
    elif isinstance(tool_specs, dict):
        mcp_config = tool_specs.get("mcp")

    # Also check step-level mcp field
    if hasattr(step, "mcp") and step.mcp:
        mcp_config = step.mcp

    if mcp_config:
        from aiorch.tools import resolve_mcp_tools
        mcp_tools, mcp_sessions = await resolve_mcp_tools(mcp_config, context=context)
        tools.extend(mcp_tools)

    tool_schemas = tools_to_openai_schema(tools) if tools else None
    tool_map = {t.name: t for t in tools}

    # Build system prompt
    system_prompt = _build_system_prompt(agent_def, step)

    # Resolve input
    resolved_input = resolve_input(step.input, context) or ""

    # Build initial messages
    from aiorch.runtime import _resolve_step_model
    model = _resolve_step_model(step, context)  # CLI --model > step YAML (Jinja) > provider default
    temperature = step.temperature if step.temperature is not None else cfg.temperature
    max_tokens = step.max_tokens or cfg.max_tokens
    messages: list[dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = ""
    if step.goal:
        user_content = template.resolve(step.goal, context)
    if resolved_input:
        if user_content:
            user_content += f"\n\n{resolved_input}"
        else:
            user_content = resolved_input
    if not user_content:
        user_content = "Begin."

    messages.append({"role": "user", "content": user_content})

    # Track whether model was explicitly set or inherited from provider default
    context["__model_source__"] = "explicit" if model else "provider-default"

    # Agent loop — with MCP session cleanup
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # Track tool-call outcomes so we can fail the step cleanly when every
    # tool call hit an MCP error. Without this check, the LLM sees the
    # error string as "context" and often parrots it back as the final
    # answer — downstream steps then faithfully summarize the error text
    # as if it were real content.
    tool_calls_attempted = 0
    tool_calls_errored = 0

    try:
        for _iteration in range(step.max_iterations):
            llm_response = await client.complete(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tool_schemas,
            )

            total_cost += llm_response.cost
            total_prompt_tokens += llm_response.prompt_tokens
            total_completion_tokens += llm_response.completion_tokens

            # Check for tool calls
            if llm_response.tool_calls:
                # Append the assistant message with tool calls
                messages.append(llm_response.raw_message.model_dump())

                for tool_call in llm_response.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    if fn_name in tool_map:
                        result = await execute_tool(tool_map[fn_name], fn_args)
                        tool_output = result.output if not result.error else f"[ERROR]: {result.error}"
                    else:
                        tool_output = f"[ERROR]: Unknown tool: {fn_name}"

                    tool_calls_attempted += 1
                    if _is_mcp_tool_error(tool_output):
                        tool_calls_errored += 1

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    })
            else:
                # No tool calls — agent believes it's done.
                # But: if the LLM returned both empty content AND no tool
                # calls, that's almost always a provider error (LiteLLM
                # logs "finish_reason='error'" upstream) masquerading as
                # a successful stop. Fail the step loudly rather than
                # returning silently with an empty output — the run
                # would otherwise be marked successful with no artifact.
                content = (llm_response.content or "").strip()
                if not content:
                    # Retry once — some providers (Gemini) return empty on
                    # first tool-calling attempt but succeed on retry.
                    if _iteration == 0:
                        logger.warning(
                            "Agent got empty response on iteration 1, retrying once "
                            "(model=%s, prompt_tokens=%d)",
                            llm_response.model, total_prompt_tokens,
                        )
                        continue
                    raise RuntimeError(
                        f"Agent step received empty response from LLM on "
                        f"iteration {_iteration + 1} with no tool calls. "
                        f"This typically means the provider returned an "
                        f"error (e.g. finish_reason='error', rate limit, "
                        f"context length exceeded, or transient upstream "
                        f"failure). Prompt tokens: {total_prompt_tokens}. "
                        f"Consider using a more capable model, reducing "
                        f"the conversation size, or retrying."
                    )
                # Fail the step if every tool call errored out AND the
                # final content is plausibly just a restatement of the
                # error. Without this, error text silently becomes the
                # step output and corrupts downstream LLM reasoning.
                if (
                    tool_calls_attempted > 0
                    and tool_calls_errored == tool_calls_attempted
                    and _looks_like_error_propagation(content)
                ):
                    raise RuntimeError(
                        f"Agent step completed but every tool call ({tool_calls_errored}/"
                        f"{tool_calls_attempted}) returned an MCP error and the "
                        f"final answer appears to restate the error rather than "
                        f"real content. Failing the step so downstream prompts "
                        f"don't summarize error text as data. "
                        f"First ~200 chars of final answer: {content[:200]!r}"
                    )
                return PromptResult(
                    content=content,
                    cost=total_cost,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    model=llm_response.model or model,
                )

        # Max iterations reached. If every tool call errored, fail the
        # step — the agent never got real data to work with, so returning
        # a placeholder would let downstream steps treat the error loop
        # as a successful completion.
        if tool_calls_attempted > 0 and tool_calls_errored == tool_calls_attempted:
            raise RuntimeError(
                f"Agent hit max_iterations={step.max_iterations} with every "
                f"tool call ({tool_calls_errored}/{tool_calls_attempted}) "
                f"returning an MCP error. No real data was obtained. "
                f"Raise max_iterations, inspect MCP registry logs, or "
                f"check tool-call timeout configuration."
            )
        return PromptResult(
            content="[Agent reached maximum iterations without completing]",
            cost=total_cost,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            model=llm_response.model if llm_response else model,
        )
    finally:
        # Clean up MCP sessions. Shielded so that a parent wait_for timeout
        # doesn't cancel the DELETE /sessions/{id} request mid-flight —
        # without shield the session would leak on the registry until the
        # 5-minute idle pool TTL recovers it.
        import asyncio as _asyncio
        for session in mcp_sessions:
            try:
                await _asyncio.shield(session.close())
            except (_asyncio.CancelledError, Exception):
                pass
