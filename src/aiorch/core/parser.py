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

"""YAML parser and Pydantic models for aiorch schema."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


# --- Input schema types ---
_INPUT_TYPES = {
    "string", "integer", "number", "boolean", "list",
    "file",        # local-disk file, passed via `-i key=@./path.ext`
    "artifact",    # content-addressed file in the artifact store (Platform)
    "http",        # live HTTP fetch (lazy via LazyHttpInput)
    "connector",   # named managed connector (Platform only)
}

# Input types that resolve automatically at run start time and are NEVER
# user-supplied. parse_input_schema() filters these out of the returned
# schema dict so the UI Run dialog only shows fields the user actually
# fills in. The runtime in dag.py reads inputs from the raw YAML directly
# (not via parse_input_schema), so filtering here is a UI-contract
# concern only — the runtime resolves these inputs regardless.
_AUTO_RESOLVED_INPUT_TYPES = {"connector", "http"}

# Types that were removed and now surface a migration error at parse
# time instead of a generic "unknown type" message. `json` / `csv` were
# subsumed by `artifact` (Platform) and `file` (CLI). `env` was removed
# because workspace_secrets + the per-step `secrets:` allowlist cover
# the same need with audit, scope, rotation, and UI visibility.
_RETIRED_INPUT_TYPES = {"json", "csv", "env"}


class InputField(BaseModel):
    """Typed input field definition.

    Supports two YAML forms:
      Simple:  topic: "AI trends"          → string default, no validation
      Rich:    topic:
                 type: string
                 default: "AI trends"
                 description: "Research topic"
                 required: true

    For the unified artifact type:
                 document:
                   type: artifact
                   format: text              # text | json | csv | binary
                   content_type: text/plain  # optional MIME hint
                   description: "Document to analyze"

    Type-specific fields (http url/method/headers, env var, artifact format)
    are stored as extra fields and passed to the input loader at runtime.
    """
    model_config = {"extra": "allow"}

    type: str = "string"
    default: Any = None
    description: str | None = None
    required: bool = False

    # Inclusive numeric bounds for integer / number inputs. Mirror the
    # JSON Schema keywords of the same name. Ignored for non-numeric
    # types — validate_inputs() only applies them when the field type
    # is integer or number, so declaring them on a string field is a
    # no-op rather than a schema error. That matches how draft-2020-12
    # treats them (annotations only apply where the type matches).
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_type(self) -> InputField:
        if self.type == "env":
            raise ValueError(
                f"Input type 'env' has been removed. Use a workspace "
                f"secret (sensitive values) or workspace config (plain "
                f"values) instead:\n"
                f"  1. Add the variable to Connectors & Secrets in the UI\n"
                f"  2. Reference it from the step that needs it via "
                f"`secrets: [VAR_NAME]` (for secrets) or as a plain "
                f"environment lookup (for configs)\n"
                f"Workspace secrets give you audit logging, per-step "
                f"allowlist scoping, and rotation without restarting "
                f"the executor — none of which env reads had."
            )
        if self.type in _RETIRED_INPUT_TYPES:
            raise ValueError(
                f"Input type '{self.type}' has been removed. "
                f"Use 'type: artifact' with the appropriate 'format:' instead:\n"
                f"  type: artifact\n"
                f"  format: {'text' if self.type == 'file' else self.type}\n"
                f"See docs/migration-artifacts.md for the full migration guide."
            )
        if self.type not in _INPUT_TYPES:
            raise ValueError(
                f"Unknown input type '{self.type}'. "
                f"Valid types: {', '.join(sorted(_INPUT_TYPES))}"
            )
        return self


def parse_input_schema(raw_input: dict[str, Any] | None) -> dict[str, InputField] | None:
    """Parse the pipeline input: block into typed InputField definitions.

    Returns None if no input block. Returns dict of field_name → InputField,
    EXCLUDING auto-resolved types (connector / env / http). Those resolve at
    run start time without user input, so they don't belong in the UI Run
    dialog. The runtime reads inputs from the raw YAML directly, so this
    filter only affects what the UI sees.

    Auto-detects simple vs rich form per field.
    """
    if not raw_input:
        return None

    fields: dict[str, InputField] = {}
    for key, value in raw_input.items():
        if isinstance(value, dict) and "type" in value:
            # Rich form: skip auto-resolved types entirely so the UI Run
            # dialog never tries to render a text input for a connector
            # query result.
            if value.get("type") in _AUTO_RESOLVED_INPUT_TYPES:
                continue
            fields[key] = InputField(**value)
        else:
            # Simple form: infer type from value
            inferred_type = "string"
            if isinstance(value, bool):
                inferred_type = "boolean"
            elif isinstance(value, int):
                inferred_type = "integer"
            elif isinstance(value, float):
                inferred_type = "number"
            elif isinstance(value, list):
                inferred_type = "list"
            fields[key] = InputField(type=inferred_type, default=value)

    # If filtering left nothing, return None so callers can short-circuit
    # the "this pipeline has inputs" check cleanly.
    return fields if fields else None


def validate_inputs(
    schema: dict[str, InputField],
    provided: dict[str, Any],
) -> list[str]:
    """Validate provided inputs against the schema. Returns list of error messages."""
    errors: list[str] = []

    for name, field in schema.items():
        value = provided.get(name)

        # Check required
        if field.required and value is None and field.default is None:
            errors.append(f"Required input '{name}' is missing")
            continue

        # Skip validation if not provided and has default
        if value is None:
            continue

        # Type check — note `bool` is a subclass of `int` in Python, so
        # the integer branch has to exclude booleans explicitly or every
        # boolean input would also pass the integer type check.
        type_ok = True
        if field.type == "string" and not isinstance(value, str):
            errors.append(f"Input '{name}' expected string, got {type(value).__name__}")
            type_ok = False
        elif field.type == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            errors.append(f"Input '{name}' expected integer, got {type(value).__name__}")
            type_ok = False
        elif field.type == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            errors.append(f"Input '{name}' expected number, got {type(value).__name__}")
            type_ok = False
        elif field.type == "boolean" and not isinstance(value, bool):
            errors.append(f"Input '{name}' expected boolean, got {type(value).__name__}")
            type_ok = False
        elif field.type == "list" and not isinstance(value, list):
            errors.append(f"Input '{name}' expected list, got {type(value).__name__}")
            type_ok = False

        # Bounds check — only applies to numeric types and only when
        # the type check passed (so an "integer got string" error isn't
        # followed by a nonsensical "string < minimum" error).
        if type_ok and field.type in ("integer", "number"):
            if field.minimum is not None and value < field.minimum:
                errors.append(
                    f"Input '{name}' must be >= {field.minimum}, got {value}"
                )
            if field.maximum is not None and value > field.maximum:
                errors.append(
                    f"Input '{name}' must be <= {field.maximum}, got {value}"
                )

    return errors


class Step(BaseModel):
    """A single step in a pipeline pipeline."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str = ""

    # Primitive — exactly one must be set
    run: str | None = None
    prompt: str | None = None
    agent: str | dict | None = None
    action: str | None = None
    flow: str | None = None
    python: str | None = None

    # Input
    input: str | list[str] | None = None
    vars: dict[str, Any] = Field(default_factory=dict)

    # LLM config
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    system: str | None = None

    # Agent config
    tools: list[str | dict] | None = None
    mcp: str | list | dict | None = None  # MCP server(s) for agent tools
    max_iterations: int = 10
    goal: str | None = None

    # Action / flow config — action-specific parameters live here
    # e.g. config: { channel: "#alerts", message: "Done" }
    config: dict[str, Any] = Field(default_factory=dict)

    # Flow control
    depends: list[str] = Field(default_factory=list)
    parallel: str | bool | None = None
    condition: str | None = None
    retry: int = 0
    retry_delay: str | None = None
    timeout: str | None = None
    cache: bool = False

    # Loop
    foreach: list[str] | str | None = None
    # When set on a foreach step, the name of a previously-produced
    # list output. Before running each iteration, the runner checks
    # if that list's value at the iteration index is a sentinel
    # ([TIMEOUT]:, [MCP Error]:, [ERROR]:, [SKIPPED]:). If so, the
    # iteration is skipped and its own output slot becomes a
    # [SKIPPED]: sentinel that propagates the upstream cause. Saves
    # LLM cost/latency when upstream already failed for this item.
    skip_on_error: str | None = None

    # Output
    output: str | None = None
    format: OutputFormat = OutputFormat.text
    save: str | None = None

    # Output validation
    output_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    retry_on_invalid: int = 0
    assertions: list[str] = Field(default_factory=list)

    # Cost control
    max_cost: float | None = None

    # Error handling
    on_failure: str | None = None
    trigger: str | None = None

    # Secret allowlist — for `run:` steps, declares which workspace
    # secrets the shell command needs access to. Listed keys are
    # injected into the subprocess environ; all other workspace
    # secrets are stripped. Empty list (the default) means no
    # workspace secrets reach the subprocess. In-process callers
    # (connectors, prompts, LLM providers) still have full access
    # via aiorch.runtime.run_env.get_env.
    secrets: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def exactly_one_primitive(self) -> Step:
        set_fields = [f for f in _PRIMITIVE_FIELDS if getattr(self, f, None) is not None]
        if len(set_fields) == 0:
            names = ", ".join(_PRIMITIVE_FIELDS)
            raise ValueError(f"Step '{self.name}': must specify one of {names}")
        if len(set_fields) > 1:
            names = ", ".join(_PRIMITIVE_FIELDS)
            raise ValueError(f"Step '{self.name}': only one primitive allowed ({names})")
        return self

    @model_validator(mode="after")
    def schema_implies_json_format(self) -> Step:
        """When output_schema is set, output must be JSON — auto-set format."""
        if self.output_schema is not None and self.format == OutputFormat.text:
            self.format = OutputFormat.json
        return self

    @property
    def primitive_type(self) -> str:
        for name in _PRIMITIVE_FIELDS:
            if getattr(self, name, None) is not None:
                return name
        raise ValueError("No primitive set")


# Primitive field names — the canonical list.
# The registry (runtime.registry) maps these to handlers.
_PRIMITIVE_FIELDS = ("run", "prompt", "agent", "action", "flow", "python")


class Agentfile(BaseModel):
    """Top-level Aiorch pipeline schema.

    Required: name, steps
    Recommended: version, outputs
    Optional: description, input, env, logging

    Note: `trigger:` was removed in 2026-04-13. It was non-functional —
    the scheduler reads from the `schedules` table (managed via the UI
    or POST /api/schedules), not from YAML. Existing YAML files with a
    `trigger:` block continue to parse cleanly (Pydantic's default
    extra="ignore" silently drops the field).
    """

    # Spec version — for forward compatibility
    version: str = "1"

    name: str
    description: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    steps: dict[str, Step] = Field(default_factory=dict)

    # Top-level input (values can be strings, dicts for typed inputs, or any default)
    input: dict[str, Any] | None = None

    # Pipeline outputs — what the run formally returns
    # result: primary output variable name
    # artifacts: list of step output names that produce saved files/reports
    outputs: dict[str, Any] | None = None

    # Per-pipeline logging override
    logging: dict[str, Any] | None = None

    # Source file path (set by parse_file, not from YAML)
    source_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def inject_step_names(self) -> Agentfile:
        for step_name, step in self.steps.items():
            step.name = step_name
        return self

    @property
    def input_schema(self) -> dict[str, InputField] | None:
        """Parse the input block into typed InputField definitions."""
        return parse_input_schema(self.input)


def parse_file(path: str | Path) -> Agentfile:
    """Parse a pipeline from a YAML file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")

    raw = path.read_text()
    af = parse_string(raw)
    af.source_path = path.resolve()
    return af


def parse_string(raw: str, *, skip_schema: bool = False) -> Agentfile:
    """Parse a pipeline from a YAML string.

    Validation layers:
      1. JSON Schema — structural correctness (field names, types, enums)
      2. Pydantic    — type coercion, exactly-one-primitive, defaults
      3. DAG         — dependency refs, cycle detection (done at execution time)
    """
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("Pipeline must be a YAML mapping")

    # Layer 1: JSON Schema validation (structural)
    if not skip_schema:
        from aiorch.core.schema_validator import validate_pipeline_schema
        schema_errors = validate_pipeline_schema(data)
        if schema_errors:
            raise ValueError(
                f"Pipeline schema validation failed:\n"
                + "\n".join(f"  - {e}" for e in schema_errors[:10])
            )

    # Layer 2: Pydantic validation (types + semantic)
    raw_steps = data.get("steps", {})
    steps = {}
    for step_name, step_data in raw_steps.items():
        if isinstance(step_data, dict):
            steps[step_name] = Step(**step_data)
        else:
            raise ValueError(f"Step '{step_name}' must be a mapping")

    data["steps"] = steps
    return Agentfile(**data)
