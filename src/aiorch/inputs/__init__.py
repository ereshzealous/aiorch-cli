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

An input value is either a plain string (returned as-is) or a dict with
``type:`` that dispatches to a loader. Types the CLI supports:

  text / string / integer / number / boolean / list  — literal values
  file    → read from disk
  http    → GET URL, return string or parsed JSON (lazy)
  env     → read an environment variable
  stdin   → read from stdin

``artifact`` and ``connector`` types raise a clear error in the CLI —
they require the commercial aiorch Platform.
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
    """Artifact-backed inputs are a commercial Platform feature.

    CLI users pass local files via ``type: file`` or pipe content via
    ``type: stdin`` — there's no artifact store in the CLI runtime.
    """

    def load(self, config: dict) -> Any:
        raise RuntimeError(
            "type: artifact inputs require the commercial aiorch Platform. "
            "In the CLI, use `type: file` with a local path, or "
            "`type: stdin` to pipe content."
        )
        # Unreachable but satisfies type-checkers.
        artifact_id = config.get("artifact_id")
        fmt = config.get("format", "text")
        _, content = None, b""

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
    """Fetches a URL → string or parsed JSON. The URL and header values
    flow through ``safe_http_url`` / ``safe_header_value`` since they
    come from user-supplied YAML. Set ``AIORCH_ALLOW_PRIVATE_HOSTS=1``
    to allow internal-service inputs."""

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
    """Deferred-fetch proxy for ``type: http`` inputs. The network call
    fires on first read (via ``__str__`` / ``__getitem__`` / attribute
    access) and is cached; declared-but-never-referenced inputs never
    hit the network."""

    __slots__ = ("_config", "_value", "_fetched", "_error")

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
        return True

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name in self._OWN_NAMES:
            raise AttributeError(name)
        return getattr(self._resolve(), name)

    def __deepcopy__(self, memo):
        import copy as _copy
        new = LazyHttpInput.__new__(LazyHttpInput)
        new._config = _copy.deepcopy(self._config, memo)
        new._fetched = self._fetched
        new._value = _copy.deepcopy(self._value, memo) if self._fetched else None
        new._error = self._error
        return new


class ConnectorInputLoader(InputLoader):
    """Stub for ``type: connector`` — a Platform-only feature. Dispatch
    normally rejects this at DAG-build time; this is a safety net."""

    def load(self, config: dict) -> Any:
        raise RuntimeError(
            "type: connector inputs require the commercial aiorch Platform. "
            "The CLI supports: text, integer, number, boolean, list, "
            "file (local disk), http, env, stdin."
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

    register_input_loader("string", text_loader)
    register_input_loader("integer", text_loader)
    register_input_loader("number", text_loader)
    register_input_loader("boolean", text_loader)
    register_input_loader("list", text_loader)
    # CLI-only: `type: file` = pre-loaded file content. The CLI's
    # parse_kv_inputs handles `@path/to/file.csv` syntax by reading
    # and parsing the file inline, then binding the parsed value
    # (str / list[dict] / dict / bytes) directly to the input name.
    # The loader here just passes through. Platform overrides this
    # at boot to route through its artifact store instead.
    register_input_loader("file", text_loader)


_register_builtin_loaders()


_RETIRED_TYPES_HELP = {
    "json": "type: artifact, format: json",
    "csv":  "type: artifact, format: csv",
    "env":  "workspace_secrets + per-step `secrets:` allowlist",
}


def load_input(value: Any) -> Any:
    """Load a single input value.

    Args:
        value: Either a plain string/value, or a dict with type: key.

    Returns:
        The loaded value (string, dict, list, etc.)
    """
    if not isinstance(value, dict):
        return _smart_load_string(value) if isinstance(value, str) else value

    if "type" not in value:
        return value

    loader_type = value["type"]

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

            suffix = path.suffix.lower()
            if suffix == ".json":
                fmt = "json"
                content_type = "application/json"
            elif suffix in (".yaml", ".yml"):
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

            # CLI mode: parse the file inline and bind the parsed value
            if fmt == "json" or suffix in (".yaml", ".yml"):
                try:
                    parsed = yaml.safe_load(path.read_text())
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML/JSON in {path}: {e}") from e
                result[key] = parsed if parsed is not None else {}
            elif fmt == "csv":
                import io
                text = path.read_text()
                reader = csv_module.DictReader(io.StringIO(text))
                result[key] = list(reader)
            elif fmt == "text":
                result[key] = path.read_text()
            else:
                # binary — return bytes
                result[key] = path.read_bytes()
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
