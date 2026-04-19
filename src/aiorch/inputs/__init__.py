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

"""Input loader system — pluggable typed inputs for pipelines.

Each input value is either:
  - A plain string → returned as-is
  - A dict with type: → dispatched to the correct loader

Supported types:
  text     → plain string
  artifact → fetch content from the artifact store (MinIO on Platform,
             local filesystem on CLI). Parses per ``format`` field
             into text / json / csv / binary.
  http     → fetch URL → string or parsed JSON
  connector → resolve via aiorch.connectors (Platform only, async)

Retired types:
  file     → use artifact with format: text
  json     → use artifact with format: json
  csv      → use artifact with format: csv
  env      → use workspace_secrets (sensitive) or workspace_configs
             (plain values) instead. Removed because env reads bypass
             the audit log, can't rotate without an executor restart,
             and aren't scoped per workspace. workspace_secrets +
             per-step `secrets:` allowlist gives the same value with
             audit, scope, and rotation.

Usage:
    from aiorch.inputs import load_input

    # Plain string
    load_input("cricket")  → "cricket"

    # Typed artifact (by ID — the store is queried)
    load_input({"type": "artifact", "artifact_id": "abc-123", "format": "text"})
        → file contents as str

    load_input({"type": "artifact", "artifact_id": "abc-123", "format": "json"})
        → dict or list

    load_input({"type": "artifact", "artifact_id": "abc-123", "format": "csv"})
        → list[dict]

    load_input({"type": "artifact", "artifact_id": "abc-123", "format": "binary"})
        → raw bytes
"""

from __future__ import annotations

import csv as csv_module
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml


class InputLoader(ABC):
    """Abstract base for input loaders."""

    @abstractmethod
    def load(self, config: dict) -> Any:
        ...


class TextInputLoader(InputLoader):
    """Returns the value as-is."""

    def load(self, config: dict) -> str:
        return config.get("value", config.get("default", ""))


class ArtifactInputLoader(InputLoader):
    """Fetches content from the artifact store and parses it per format.

    Expects a config dict of the shape:
        {"type": "artifact", "artifact_id": "<uuid>", "format": "<format>"}

    The format decides how bytes are interpreted:
        text   → str (default)
        json   → dict or list (from json.loads)
        csv    → list[dict]  (from csv.DictReader)
        binary → raw bytes

    The content_type on the stored artifact is informational for
    display and validation — it does NOT drive parse dispatch.
    Format is authoritative, so a pipeline author always knows what
    Python type their step receives.
    """

    def load(self, config: dict) -> Any:
        artifact_id = config.get("artifact_id")
        if not artifact_id:
            raise ValueError(
                "Artifact input requires 'artifact_id'. "
                "Use `aiorch run ... -i key=@./file.txt` to upload via CLI, "
                "or the Run dialog upload widget in the UI."
            )

        fmt = config.get("format", "text")

        from aiorch.artifacts import get_artifact_store
        _, content = get_artifact_store().get(artifact_id)

        if fmt == "binary":
            return content

        if fmt == "text":
            return content.decode("utf-8", errors="replace")

        if fmt == "json":
            return json.loads(content.decode("utf-8"))

        if fmt == "csv":
            import io
            delimiter = config.get("delimiter", ",")
            text = content.decode("utf-8")
            reader = csv_module.DictReader(io.StringIO(text), delimiter=delimiter)
            return list(reader)

        raise ValueError(
            f"Unknown artifact format: '{fmt}'. "
            f"Valid formats: text, json, csv, binary"
        )


class HttpInputLoader(InputLoader):
    """Fetches a URL → string or parsed JSON.

    SSRF + header gates (Pass 3 #2 in the audit): the URL and any
    custom header values come from pipeline YAML (or input
    overrides), so they go through the same safety helpers as the
    webhook action. AIORCH_ALLOW_PRIVATE_HOSTS=1 is the operator
    opt-out for legitimate internal-service inputs.
    """

    def load(self, config: dict) -> Any:
        import httpx
        from aiorch.core.http_safety import safe_header_value, safe_http_url

        url = config["url"]
        method = config.get("method", "GET").upper()
        headers = config.get("headers", {})
        body = config.get("body")
        fmt = config.get("format", "text")

        # Resolve env vars in headers
        resolved_headers = {}
        for k, v in headers.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                env_name = v[2:-1]
                resolved_headers[k] = os.environ.get(env_name, "")
            else:
                resolved_headers[k] = v

        # SSRF + CRLF gates before httpx sees anything.
        url = safe_http_url(url, purpose="http: input")
        resolved_headers = {
            k: safe_header_value(v, name=str(k), purpose="http: input")
            for k, v in resolved_headers.items()
        }

        with httpx.Client(timeout=30) as client:
            response = client.request(
                method=method,
                url=url,
                headers=resolved_headers,
                content=body,
            )
            response.raise_for_status()

        if fmt == "json":
            return response.json()
        return response.text


class LazyHttpInput:
    """Deferred-fetch proxy for ``type: http`` inputs.

    Wraps an HTTP config dict and defers the actual network call
    until the first time the value is read (Jinja2 ``__str__``,
    ``__getattr__``, or ``__getitem__``). Once fetched, the result
    is cached so a second step referencing the same input doesn't
    re-issue the request.

    This is what Phase B.4 of the input-artifacts work gets us:
    ``type: http`` inputs no longer block pipeline start, and
    dead http inputs (declared but never referenced) never fire
    a network call at all.

    The proxy is drop-in compatible with whatever the underlying
    loader would have returned — str for ``format: text``, dict
    or list for ``format: json``. Jinja2's template rendering
    handles the transparency via ``__str__`` / ``__getitem__``
    dispatch.
    """

    __slots__ = ("_config", "_value", "_fetched", "_error")

    # The names that live on this class itself, not on the resolved
    # value. __getattr__ MUST NOT forward these to _resolve() — doing
    # so creates infinite recursion when an instance is in a broken
    # state (e.g. created via cls.__new__(cls) without __init__, as
    # copy.deepcopy does for slotted classes that don't define
    # __deepcopy__).
    _OWN_NAMES = frozenset(("_config", "_value", "_fetched", "_error"))

    def __init__(self, config: dict):
        self._config = dict(config)
        self._value: Any = None
        self._fetched = False
        self._error: Exception | None = None

    def _resolve(self) -> Any:
        if not self._fetched:
            try:
                self._value = HttpInputLoader().load(self._config)
            except Exception as e:
                self._error = e
                self._value = None
            self._fetched = True
        if self._error is not None:
            raise self._error
        return self._value

    def __str__(self) -> str:
        val = self._resolve()
        return str(val) if val is not None else ""

    def __repr__(self) -> str:
        if self._fetched:
            return f"LazyHttpInput(resolved, url={self._config.get('url')!r})"
        return f"LazyHttpInput(pending, url={self._config.get('url')!r})"

    def __getitem__(self, key):
        return self._resolve()[key]

    def __iter__(self):
        return iter(self._resolve())

    def __len__(self):
        return len(self._resolve())

    def __contains__(self, key):
        return key in self._resolve()

    def __bool__(self) -> bool:
        # Don't fetch just for truthiness. An input that was
        # declared in the YAML is "present" regardless of its value.
        return True

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal attribute lookup fails.
        # Refuse to forward our own slot names and dunder methods —
        # if they're missing, the instance is in a broken state and
        # the right thing is to raise AttributeError so the caller
        # (copy.deepcopy, pickle, debugger introspection) knows.
        # Forwarding them would call _resolve() which accesses the
        # very slots that aren't set, recursing until the stack
        # overflows.
        if name.startswith("_") or name in self._OWN_NAMES:
            raise AttributeError(name)
        return getattr(self._resolve(), name)

    def __deepcopy__(self, memo):
        # Define an explicit __deepcopy__ so copy.deepcopy() doesn't
        # go through cls.__new__(cls) + slot-poking, which leaves
        # the instance in a broken state and triggers __getattr__
        # recursion. The python: primitive's input handler deep-
        # copies the entire context dict for immutability; without
        # this hook, every http input crashes that path.
        #
        # Semantics: preserve the lazy/fetched state. If the original
        # already fetched, copy the resolved value (deepcopied so
        # downstream mutations stay isolated). If not yet fetched,
        # the copy starts unfetched too — it'll do its own fetch on
        # first access. Cache benefit is lost across copies but
        # correctness is preserved.
        import copy as _copy
        new = LazyHttpInput.__new__(LazyHttpInput)
        new._config = _copy.deepcopy(self._config, memo)
        new._fetched = self._fetched
        new._value = _copy.deepcopy(self._value, memo) if self._fetched else None
        new._error = self._error
        return new


class ConnectorInputLoader(InputLoader):
    """Stub loader for ``type: connector`` inputs.

    Connector operations are asynchronous and require the pipeline
    execution context (org_id / workspace_id / run_id) for scope
    resolution, metrics and lineage. They cannot be resolved through
    the synchronous ``load_input()`` entry point.

    Resolution happens in the dag context builder (``execute_dag``),
    which awaits ``aiorch.connectors.integration.resolve_and_execute``
    before any step runs. This stub guards the synchronous path so a
    stray ``load_input()`` call surfaces a clear error instead of a
    mysterious KeyError or silent string return.
    """

    def load(self, config: dict) -> Any:
        raise ValueError(
            "type: connector inputs must be resolved via the async dag "
            "context builder. This error means a sync code path tried to "
            "load a connector input directly — wire the call through "
            "aiorch.connectors.integration.resolve_and_execute instead."
        )


# --- Loader registry ---

_LOADERS: dict[str, InputLoader] = {}


def register_input_loader(name: str, loader: InputLoader) -> None:
    """Register an input loader for a given type name.

    Args:
        name: Type name (e.g., "sql", "s3", "parquet").
        loader: An InputLoader instance.

    Example:
        register_input_loader("parquet", ParquetInputLoader())
    """
    _LOADERS[name] = loader


def _register_builtin_loaders() -> None:
    """Register the built-in input loaders."""
    text_loader = TextInputLoader()
    register_input_loader("text", text_loader)
    register_input_loader("artifact", ArtifactInputLoader())
    register_input_loader("http", HttpInputLoader())
    register_input_loader("connector", ConnectorInputLoader())

    # Aliases — InputField schema uses these names for scalar types,
    # all of which are stored as plain values in the context dict and
    # never routed through a file/content loader.
    register_input_loader("string", text_loader)
    register_input_loader("integer", text_loader)
    register_input_loader("number", text_loader)
    register_input_loader("boolean", text_loader)
    register_input_loader("list", text_loader)


_register_builtin_loaders()


# Map retired type names to helpful migration messages. The parser
# already rejects these at validation time, but load_input() also
# guards against them in case a run context slips through with an
# old typed dict (e.g. from a replayed run_events row).
_RETIRED_TYPES_HELP = {
    "file": "type: artifact, format: text",
    "json": "type: artifact, format: json",
    "csv":  "type: artifact, format: csv",
    # `env` was removed because workspace_secrets and workspace_configs
    # already cover the same need with audit trails, scoping, rotation,
    # and UI visibility — none of which env reads had. Migration is
    # mechanical: declare the variable in Connectors & Secrets and
    # reference it via `secrets: [VAR_NAME]` on the step that uses it.
    "env":  "workspace_secrets + per-step `secrets:` allowlist",
}


def load_input(value: Any) -> Any:
    """Load a single input value.

    Args:
        value: Either a plain string/value, or a dict with type: key.

    Returns:
        The loaded value (string, dict, list, etc.)
    """
    # Plain value — return as-is
    if not isinstance(value, dict):
        return _smart_load_string(value) if isinstance(value, str) else value

    # Dict without type — return as-is (it's a plain dict value)
    if "type" not in value:
        return value

    loader_type = value["type"]

    # Retired types get a clear migration error instead of
    # "unknown type". This catches replays of old run contexts
    # and any YAML that slipped past the parser.
    if loader_type in _RETIRED_TYPES_HELP:
        raise ValueError(
            f"Input type '{loader_type}' has been removed. "
            f"Use `{_RETIRED_TYPES_HELP[loader_type]}` instead. "
            f"See docs/migration-artifacts.md for the full migration guide."
        )

    loader = _LOADERS.get(loader_type)
    if loader is None:
        raise ValueError(
            f"Unknown input type: '{loader_type}'. "
            f"Registered: {', '.join(sorted(_LOADERS.keys()))}. "
            f"Use register_input_loader() to add new types."
        )
    return loader.load(value)


def load_inputs(raw: dict[str, Any]) -> dict[str, Any]:
    """Load all inputs from a dict (pipeline input: block or --input file).

    Each value is either a plain value or a typed dict.
    """
    result = {}
    for key, value in raw.items():
        result[key] = load_input(value)
    return result


def parse_input_arg(arg: str) -> dict[str, Any]:
    """Parse the --input CLI argument.

    Accepts:
        - File path (.yaml, .yml, .json) → reads and parses
        - Inline JSON string → parses
        - Inline YAML string → parses
    """
    # File path?
    path = Path(arg)
    if path.exists() and path.is_file():
        content = path.read_text()
        ext = path.suffix.lower()
        if ext in (".json",):
            return json.loads(content)
        else:
            return yaml.safe_load(content) or {}

    # Inline JSON?
    stripped = arg.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Inline YAML?
    if ":" in stripped:
        try:
            parsed = yaml.safe_load(stripped)
            if isinstance(parsed, dict):
                return parsed
        except yaml.YAMLError:
            pass

    raise ValueError(
        f"Cannot parse --input: '{arg[:50]}...'\n"
        f"Expected: YAML/JSON file path, inline JSON, or inline YAML"
    )


def parse_kv_inputs(pairs: tuple[str, ...] | list[str]) -> dict:
    """Parse -i key=value pairs from the CLI.

    Formats:
        key=value       -> string (or int/float if parseable)
        key=@file.txt   -> uploads file to the artifact store, returns
                           {type: artifact, artifact_id, format: text}
        key=@file.json  -> uploads file, returns format: json
        key=@file.csv   -> uploads file, returns format: csv
        key=@file.bin   -> uploads file, returns format: binary
        key=@-          -> reads stdin and uploads (NOT YET IMPLEMENTED)

    File uploads go through ``get_artifact_store().put()`` so CLI
    and Platform use the same mechanism. Content is deduped by
    sha256 — uploading the same file twice returns the same
    artifact_id both times.
    """
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"Invalid input '{pair}' — use key=value or key=@file"
            )
        key, raw = pair.split("=", 1)
        key = key.strip()

        if raw.startswith("@"):
            path = Path(raw[1:]).expanduser()
            if not path.exists():
                raise ValueError(f"Input file not found: {path}")

            # Infer format from extension; user can override in the
            # pipeline YAML's `format:` field, but this default
            # covers the 99% case.
            suffix = path.suffix.lower()
            if suffix == ".json":
                fmt = "json"
                content_type = "application/json"
            elif suffix in (".yaml", ".yml"):
                # YAML files aren't a first-class artifact format —
                # parse them client-side and pass as a dict so the
                # pipeline gets the same shape as type: json.
                fmt = "json"
                content_type = "application/x-yaml"
            elif suffix == ".csv":
                fmt = "csv"
                content_type = "text/csv"
            elif suffix in (".md", ".txt", ".rst"):
                fmt = "text"
                content_type = _content_type_for_ext(suffix)
            else:
                fmt = "text"
                content_type = _content_type_for_ext(suffix)

            # Read bytes and upload. For YAML, we pre-convert to JSON
            # so the store holds portable content; format stays 'json'
            # so the loader parses it back to dict/list.
            if suffix in (".yaml", ".yml"):
                try:
                    parsed = yaml.safe_load(path.read_text())
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML in {path}: {e}") from e
                content = json.dumps(parsed or {}).encode("utf-8")
            else:
                content = path.read_bytes()

            from aiorch.artifacts import get_artifact_store, init_artifact_store
            try:
                store = get_artifact_store()
            except RuntimeError:
                init_artifact_store()
                store = get_artifact_store()

            artifact = store.put(
                name=path.name,
                content=content,
                content_type=content_type,
                role="input",
            )
            result[key] = {
                "type": "artifact",
                "artifact_id": artifact.id,
                "format": fmt,
            }
        else:
            # Try numeric coercion
            try:
                result[key] = int(raw)
                continue
            except ValueError:
                pass
            try:
                result[key] = float(raw)
                continue
            except ValueError:
                pass
            result[key] = raw
    return result


def _content_type_for_ext(suffix: str) -> str:
    """Map a file extension to a best-effort MIME type."""
    import mimetypes
    guess = mimetypes.guess_type(f"x{suffix}")[0]
    return guess or "application/octet-stream"


def _smart_load_string(raw: str) -> Any:
    """Pass-through for plain string inputs.

    Previously did path auto-detection (if the string happened to name
    a file on disk, read its contents). That heuristic is removed now
    that file content lives in the artifact store — a string input
    means a string value, unambiguously.

    Users who want to pass a file's contents should declare the input
    as ``type: artifact`` and upload via the CLI (``-i key=@./file``)
    or the UI Run dialog.
    """
    return raw
