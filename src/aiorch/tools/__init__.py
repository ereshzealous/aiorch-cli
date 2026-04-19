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

"""Tool system for the agent primitive.

Security note
-------------
This module used to ship three built-in tools — file_read, file_search,
run_command — that gave LLM agents direct access to the executor's
filesystem and shell. They were removed from the built-in registry
after a security audit identified them as remote code execution and
information disclosure vectors:

  - run_command: any agent step with `tools: [run_command]` handed
    the LLM a blank-check shell running as the executor uid. A single
    prompt-injection in any data the agent read (webhook body, ticket
    text, document, search result) would cause the LLM to emit a tool
    call that became a direct /bin/sh invocation.

  - file_read / file_search: the LLM could open() or grep any path
    on the executor host, including /proc/self/environ, /root/.aws/
    credentials, service-account tokens, and pipeline YAML files from
    other workspaces. No path confinement, no symlink handling.

Pipeline authors who need file access or shell access for an agent
should use an MCP server instead — MCP servers run as separate
processes and are chosen explicitly by the pipeline author, not
enabled by typing a built-in name in a list.

The underlying implementations are preserved as _unsafe_* helpers so
they can be wrapped in a sandboxed variant (path confinement, container
isolation) in a future release. They are NOT referenced by
get_builtin_tools() and should not be added back without the sandbox.
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    """Definition of a tool that an agent can invoke."""
    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    fn: Callable = field(default=lambda **kwargs: "")
    mcp_session: object | None = field(default=None, repr=False)  # MCPSession if MCP-backed


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_name: str
    output: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Unsafe legacy tool implementations (kept for future sandboxing)
# ---------------------------------------------------------------------------
#
# These functions are NOT registered as built-in tools. Do not import
# them outside this module and do not add them back to get_builtin_tools
# without wrapping them in a sandbox (path root for file tools, container
# for shell). See the module docstring for the history.

def _unsafe_file_read(path: str) -> str:
    """Read a file and return its contents (max 50KB).

    UNSAFE: follows symlinks, no path confinement. Do not expose to
    LLM agents without a sandbox wrapper.
    """
    max_bytes = 50 * 1024
    with open(path, "r") as f:
        content = f.read(max_bytes + 1)
    if len(content) > max_bytes:
        content = content[:max_bytes]
        content += "\n\n[WARNING: File truncated at 50KB]"
    return content


def _unsafe_file_search(pattern: str, path: str = ".") -> str:
    """Search for a pattern across common file types using grep.

    UNSAFE: accepts any path, greps any extension in the include list
    from that root. Do not expose to LLM agents without a sandbox
    wrapper.
    """
    cmd = [
        "grep", "-rn", pattern, path,
        "--include=*.py",
        "--include=*.yaml",
        "--include=*.json",
        "--include=*.md",
        "--include=*.go",
        "--include=*.ts",
        "--include=*.js",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) > 50:
            lines = lines[:50]
            lines.append("\n[WARNING: Results truncated to 50 lines]")
        return "\n".join(lines)
    except subprocess.TimeoutExpired:
        return "[ERROR: Search timed out after 30s]"
    except Exception as e:
        return f"[ERROR: {e}]"


async def _unsafe_run_command(command: str) -> str:
    """Run a shell command and return stdout (max 5000 chars, 30s timeout).

    UNSAFE: raw shell string fed to /bin/sh -c. No sandbox, no uid
    switch, no resource limits beyond a 30s wall clock. Do not expose
    to LLM agents.
    """
    max_chars = 5000
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return "[ERROR: Command timed out after 30s]"

    output = stdout.decode()
    if proc.returncode != 0:
        err = stderr.decode().strip()
        output += f"\n[STDERR]: {err}" if err else ""

    if len(output) > max_chars:
        output = output[:max_chars] + "\n\n[WARNING: Output truncated at 5000 chars]"
    return output.strip()


# ---------------------------------------------------------------------------
# Safe built-in tool implementations
# ---------------------------------------------------------------------------

def web_search(query: str) -> str:
    """Web search is not built-in. Use MCP for web search capabilities.

    Example in YAML:
        agent:
          mcp: "npx -y @anthropic/mcp-server-brave-search"
          prompt: "Search for..."
    """
    return (
        "Web search is not included as a built-in tool. "
        "Use MCP to add search capabilities to your agent:\n\n"
        "  agent:\n"
        "    mcp: \"npx -y @anthropic/mcp-server-brave-search\"\n"
        "    prompt: \"Search for...\""
    )


# ---------------------------------------------------------------------------
# Built-in tool registry
# ---------------------------------------------------------------------------

_WEB_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query"},
    },
    "required": ["query"],
}


def get_builtin_tools() -> dict[str, ToolDefinition]:
    """Return a dict of all built-in tools keyed by name.

    The registry is intentionally small. Host-access tools (file_read,
    file_search, run_command) were removed in the security hardening
    pass — use MCP servers for file or shell access. See the module
    docstring.
    """
    return {
        "web_search": ToolDefinition(
            name="web_search",
            description="Search the web for information.",
            parameters=_WEB_SEARCH_PARAMS,
            fn=web_search,
        ),
    }


# Tools that used to be built-in but were removed for security reasons.
# The error message lands in the user's face the moment they try to
# resolve one of these names, so it has to be self-contained: what was
# removed, why, and exactly how to migrate.
_REMOVED_TOOLS: dict[str, str] = {
    "file_read": (
        "The built-in 'file_read' tool was removed because it gave LLM "
        "agents unrestricted read access to the executor's filesystem "
        "(including /proc/self/environ, cloud credential files, and "
        "other workspaces' pipeline YAML). There was no path confinement, "
        "no symlink handling, and no workspace isolation.\n\n"
        "Migrate to MCP for scoped file access:\n\n"
        "  tools:\n"
        "    mcp: \"npx -y @modelcontextprotocol/server-filesystem /path/to/sandbox\"\n\n"
        "The MCP filesystem server takes a root directory and restricts "
        "all reads to that subtree."
    ),
    "file_search": (
        "The built-in 'file_search' tool was removed because it allowed "
        "LLM agents to grep arbitrary host paths. A single prompt "
        "injection could turn it into a recon amplifier for finding "
        "secrets on the executor host.\n\n"
        "Migrate to MCP's filesystem server (see file_read) for scoped "
        "search, or write a purpose-built connector against your "
        "indexing system (Elasticsearch, ripgrep service, etc.)."
    ),
    "run_command": (
        "The built-in 'run_command' tool was removed because it handed "
        "LLM agents an unrestricted shell on the executor host, running "
        "as the executor uid with access to every workspace secret in "
        "the subprocess environ. A single prompt injection in any data "
        "the agent reads (webhook body, ticket text, document content, "
        "search result) became a direct /bin/sh invocation.\n\n"
        "There is no safe in-process way to restore this tool. Options:\n"
        "  1. Restructure the pipeline so shell work lives in a 'run:' "
        "step where the command text is pipeline-author controlled, "
        "not LLM-emitted. This is the recommended path.\n"
        "  2. Use an MCP shell server running in a sandboxed container "
        "if you genuinely need LLM-composed shell execution."
    ),
}


# ---------------------------------------------------------------------------
# Tool resolution and conversion
# ---------------------------------------------------------------------------

def resolve_tools(tool_specs: list[str | dict]) -> list[ToolDefinition]:
    """Resolve a list of tool names or inline specs to ToolDefinition objects.

    - String names are looked up in the built-in tool registry.
    - Dict specs create custom ToolDefinition instances.
    - "mcp" key is handled separately (see resolve_mcp_tools).

    If a string names a tool that was removed for security reasons, the
    error message explains what was removed and how to migrate.
    """
    builtins = get_builtin_tools()
    tools: list[ToolDefinition] = []

    for spec in tool_specs:
        if isinstance(spec, str):
            if spec in builtins:
                tools.append(builtins[spec])
                continue
            if spec in _REMOVED_TOOLS:
                raise ValueError(
                    f"Tool {spec!r} is no longer a built-in.\n\n"
                    f"{_REMOVED_TOOLS[spec]}"
                )
            raise ValueError(f"Unknown built-in tool: {spec!r}")
        elif isinstance(spec, dict):
            if "name" in spec:
                tools.append(ToolDefinition(
                    name=spec["name"],
                    description=spec.get("description", ""),
                    parameters=spec.get("parameters", {}),
                    fn=spec.get("fn", lambda **kwargs: ""),
                ))
            # MCP specs are handled by resolve_mcp_tools in the agent loop
        else:
            raise ValueError(f"Invalid tool spec type: {type(spec)}")

    return tools


_LEGACY_SERVER_MIGRATION_HELP = (
    "The legacy `mcp.server:` syntax has been removed. All MCP servers must "
    "now be registered in the MCP Servers catalog and referenced by name. "
    "\n\nTo migrate:\n"
    "  1. Open the aiorch UI → Connect → MCP Servers\n"
    "  2. Create a named configuration pointing at the MCP server type you need "
    "(e.g. a 'my-github' config using the GitHub catalog entry)\n"
    "  3. In your pipeline YAML, replace:\n"
    "       mcp:\n"
    "         - server: \"npx -y @modelcontextprotocol/server-github\"\n"
    "     with:\n"
    "         mcp:\n"
    "           - uses: my-github\n\n"
    "The registry path is strictly better: session pooling, capacity "
    "limits, auth, and structured metrics all apply. Running MCP servers "
    "directly on the executor bypassed all of these."
)


async def resolve_mcp_tools(
    tool_specs: list | dict | None,
    context: dict[str, Any] | None = None,
) -> tuple[list[ToolDefinition], list]:
    """Resolve MCP server specs into ToolDefinitions via the MCP registry.

    Only the registered form is supported:

        tools:
          mcp:
            - uses: my-github        # named workspace config
            - uses: prod-db
              args: ["--read-only"]  # pipeline-level arg overrides
              env:
                EXTRA_VAR: foo       # pipeline-level env overrides

    Named configurations are managed in the aiorch UI under
    Connect → MCP Servers. They reference a row in mcp_server_catalog
    (the pre-installed MCP server binaries baked into the registry image).

    ``context`` is the execution context for the current step. When
    supplied, ${VAR} placeholders in server commands, env blocks, and
    HTTP headers resolve against the per-run env bucket first and fall
    back to os.environ. Pass None (CLI/tests) to use os.environ only.

    Environment variables are merged from three sources in priority order:
      1. Pipeline-level env overrides (from the YAML spec)
      2. Workspace env_config (from mcp_workspace_servers)
      3. os.environ (fallback)

    The legacy `server:` form that ran MCP servers directly on the
    executor has been removed — it bypassed pooling, capacity limits,
    auth, and metrics. Pipelines using it get a clear migration error.

    Returns (tools, sessions) — caller must close sessions when done.
    """
    if not tool_specs:
        return [], []

    # Normalize to list of server configs
    servers = []
    if isinstance(tool_specs, str):
        # A bare string is the legacy raw-command form — reject it.
        servers = [{"server": tool_specs}]
    elif isinstance(tool_specs, list):
        for s in tool_specs:
            if isinstance(s, str):
                servers.append({"server": s})
            elif isinstance(s, dict):
                servers.append(s)
    elif isinstance(tool_specs, dict):
        servers = [tool_specs]

    tools = []
    sessions = []

    for srv_config in servers:
        if "uses" in srv_config:
            try:
                new_tools, session = await _resolve_registry_server(
                    srv_config, context,
                )
                tools.extend(new_tools)
                if session is not None:
                    sessions.append(session)
            except Exception as e:
                import logging
                logging.getLogger("aiorch.mcp").error(
                    "Failed to resolve MCP registry server '%s': %s",
                    srv_config.get("uses"), e,
                )
                # uses: resolution failures must fail the step — running
                # with zero tools would silently produce wrong results.
                raise RuntimeError(
                    f"MCP server '{srv_config['uses']}' could not be resolved: {e}"
                ) from e
            continue

        # Legacy form — reject with clear migration guidance.
        if "server" in srv_config:
            raise ValueError(_LEGACY_SERVER_MIGRATION_HELP)

        raise ValueError(
            "MCP spec must include `uses: <config-name>` pointing at a "
            "registered MCP server configuration. Create one in the UI "
            "under Connect → MCP Servers."
        )

    return tools, sessions


# ---------------------------------------------------------------------------
# Registry-backed MCP resolution (uses: key)
# ---------------------------------------------------------------------------

import hashlib
import hmac
import os as _os

_MCP_REGISTRY_URL = _os.environ.get("MCP_REGISTRY_URL", "http://mcp-registry:4000")
_MCP_TOKEN_DERIVATION_CONTEXT = b"mcp-registry-v1"


def _resolve_registry_token() -> str:
    """Resolve the MCP registry bearer token.

    Preference order:
      1. Explicit MCP_REGISTRY_TOKEN env var (operator override)
      2. Derived from AIORCH_SECRET_KEY via HMAC-SHA256

    The derivation keeps executor and registry in sync without a separate
    secret to manage: both derive the same 32-char hex token from the
    same AIORCH_SECRET_KEY. Rotating the secret rotates the MCP token.
    """
    explicit = _os.environ.get("MCP_REGISTRY_TOKEN", "").strip()
    if explicit:
        return explicit
    secret = _os.environ.get("AIORCH_SECRET_KEY", "").strip()
    if not secret:
        # Return empty string — request will fail at the registry with 401.
        # The failure mode is clearer than raising here, which would crash
        # pipelines at tool-resolution time even when MCP isn't used.
        return ""
    digest = hmac.new(
        secret.encode("utf-8"),
        _MCP_TOKEN_DERIVATION_CONTEXT,
        hashlib.sha256,
    ).hexdigest()
    return digest[:32]


def _registry_auth_headers() -> dict[str, str]:
    """Build Authorization header for registry requests."""
    token = _resolve_registry_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _registry_request(
    method: str,
    url: str,
    *,
    json_body: dict | None = None,
    timeout: float = 30.0,
    retry_on_5xx: bool = True,
):
    """HTTP call to the MCP registry with one retry on 5xx / connection error.

    Wraps raw HTTP errors with human-readable messages so failures land in
    agent output as actionable hints instead of stack traces.
    """
    import asyncio
    import httpx

    headers = _registry_auth_headers()
    attempts = 2 if retry_on_5xx else 1
    last_err: Exception | None = None

    for attempt in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.request(method, url, json=json_body, headers=headers)
            if resp.status_code >= 500 and attempt < attempts - 1:
                await asyncio.sleep(0.5)
                continue
            return resp
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            last_err = e
            if attempt < attempts - 1:
                await asyncio.sleep(0.5)
                continue
            raise RuntimeError(
                f"MCP registry at {url} is unreachable. "
                "Check that the mcp-registry container is running."
            ) from e
        except httpx.RequestError as e:
            # Other network errors — don't retry
            raise RuntimeError(f"MCP registry request failed: {e}") from e

    # Should not reach here
    raise RuntimeError(f"MCP registry request failed: {last_err}")


def _humanize_registry_error(resp, context: str = "") -> str:
    """Turn a non-2xx response into an actionable message."""
    code = resp.status_code
    try:
        body = resp.json()
        detail = body.get("error") or body.get("detail") or resp.text
    except Exception:
        detail = resp.text or f"HTTP {code}"

    prefix = f"{context}: " if context else ""
    if code == 401:
        return (
            f"{prefix}MCP registry rejected auth. "
            "Check that MCP_REGISTRY_TOKEN matches on executor and registry."
        )
    if code == 404:
        return f"{prefix}MCP session expired or not found. Re-run the step."
    if code == 429:
        return f"{prefix}MCP registry at capacity — {detail}"
    if code >= 500:
        return f"{prefix}MCP registry error ({code}): {detail}"
    return f"{prefix}MCP registry returned {code}: {detail}"


class RegistryMCPSession:
    """Wraps HTTP calls to the MCP registry's session endpoints.

    Conforms to the same interface as MCPSession (tools list, call,
    close) so ToolDefinition closures work identically.

    Routing: the registry returns a ``worker_url`` at session create
    that identifies which registry replica owns the child process.
    Tool calls and close go directly to that worker (R3.2) so multi-
    registry deployments work correctly. Falls back to the original
    ``registry_url`` (the LB/DNS entry point) when ``worker_url`` is
    not set — single-replica deployments unchanged.
    """

    def __init__(
        self,
        registry_url: str,
        session_id: str,
        tools: list[dict],
        server_name: str,
        worker_url: str | None = None,
    ):
        self.registry_url = registry_url
        self.worker_url = worker_url or registry_url
        self.session_id = session_id
        self.raw_tools = tools
        self.server_name = server_name
        self.server_key = f"registry:{server_name}"

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool via the owning worker's session call endpoint.

        Resilience — auto-recovery on ``ConnectError``:
        - First attempt uses ``self.worker_url`` (direct routing to the
          replica that owns this session's child process).
        - On connection failure (replica died / network blip / container
          restarted), fall back to ``self.registry_url`` (the LB/DNS
          entry point). This handles the common case where worker_url
          is a docker-network hostname unreachable from a host-run
          executor, and single-replica deployments where the one replica
          is always reachable via registry_url regardless.
        - On 404 from the fallback attempt too, the session is gone
          (replica died and no live replica has it). The caller gets a
          clear error; the agent loop can decide to retry the tool.

        Idempotency caveat: re-trying a tool call on a new session is
        only safe for idempotent tools (fetch, read, search). Write
        tools that have side effects (GitHub issue create, Slack post,
        DB insert) should not be auto-retried; they need per-tool
        idempotency keys or explicit operator retry. This auto-recovery
        only covers the transport-layer retry on the SAME session —
        it does NOT recreate the session.
        """
        import httpx

        # Tool calls can take 30-60s on slow sites (fetch hits robots.txt +
        # large page + HTML→markdown + readability.js extract). The default
        # 30s was too aggressive — the executor would abandon mid-call,
        # registry would detect client_close, and the [MCP Error] would
        # propagate to the agent. Match to the registry's CALL_TIMEOUT_MS
        # (90s) plus a 30s network margin.
        async def _try(url: str) -> httpx.Response:
            return await _registry_request(
                "POST",
                f"{url}/sessions/{self.session_id}/call",
                json_body={"tool": tool_name, "arguments": arguments},
                timeout=120.0,
                retry_on_5xx=False,
            )

        resp = None
        used_fallback = False
        try:
            resp = await _try(self.worker_url)
        except RuntimeError as e:
            # Only ConnectError-class failures should trigger fallback.
            # _registry_request wraps them as "MCP registry at ... is
            # unreachable". Other RuntimeErrors (auth, bad input) should
            # NOT fall through — they'd fail the same way on registry_url.
            if "unreachable" in str(e) and self.registry_url != self.worker_url:
                import logging
                logging.getLogger("aiorch.mcp").warning(
                    "MCP worker_url %s unreachable — falling back to %s "
                    "(session %s may be lost if replica is dead)",
                    self.worker_url, self.registry_url, self.session_id,
                )
                try:
                    resp = await _try(self.registry_url)
                    used_fallback = True
                except RuntimeError as e2:
                    return f"[MCP Error]: {e2}"
                except Exception as e2:
                    return f"[MCP Registry Error]: {e2}"
            else:
                return f"[MCP Error]: {e}"
        except Exception as e:
            return f"[MCP Registry Error]: {e}"

        if resp.status_code >= 400:
            # Humanize 404 specifically — it usually means the session
            # is lost (replica died). Agent can decide to retry.
            ctx = "tool call failed (session may be lost on replica crash)" \
                if resp.status_code == 404 and used_fallback \
                else "tool call failed"
            return f"[MCP Error]: {_humanize_registry_error(resp, ctx)}"

        try:
            data = resp.json()
            result = data.get("result", {})
            content = result.get("content", [])
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    else:
                        import json as _json
                        parts.append(_json.dumps(block))
                else:
                    parts.append(str(block))

            output = "\n".join(parts)
            if result.get("isError"):
                return f"[MCP Error]: {output}"
            return output
        except Exception as e:
            return f"[MCP Registry Error]: {e}"

    async def close(self) -> None:
        """Tear down the registry session on its owning worker (best-effort).

        Tries worker_url first, falls back to registry_url on connection
        failure. Both errors are swallowed — session cleanup is cosmetic
        (TTL will remove orphans eventually).
        """
        for url in (self.worker_url, self.registry_url):
            if not url:
                continue
            try:
                await _registry_request(
                    "DELETE",
                    f"{url}/sessions/{self.session_id}",
                    timeout=5,
                    retry_on_5xx=False,
                )
                return  # Succeeded — done
            except Exception:
                if url == self.registry_url or self.worker_url == self.registry_url:
                    return  # Tried everything we can
                continue  # Fall through to registry_url


def _get_mcp_workspace_config(
    workspace_id: str | None,
    config_name: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a named MCP configuration from mcp_workspace_servers.

    The ``uses:`` key in pipeline YAML now resolves against the
    user-chosen ``name`` column in mcp_workspace_servers (e.g. "prod-db"),
    not against the catalog server name (e.g. "postgres"). This lets
    multiple configs share one catalog entry with different credentials.

    Resolution:
      1. Look up mcp_workspace_servers WHERE workspace_id=X AND name=config_name
      2. Join with mcp_server_catalog to get package/binary
      3. Parse env_config + args_config JSONB
      4. Expand ${SECRET} references via workspace secrets
      5. Return {env, args, catalog_name, package, binary_name}

    Falls back to the old behavior (matching by catalog name) when no
    workspace config with that exact name is found, for backwards
    compatibility with pipelines that used ``uses: postgres``.
    """
    if not workspace_id:
        return {}

    try:
        from aiorch.storage import get_store
        from aiorch.runtime.run_env import expand_env_vars
        store = get_store()

        # Primary: match by user-chosen config name
        row = store.query_one(
            "SELECT ws.env_config, ws.args_config, "
            "       cat.name AS catalog_name, cat.package, cat.binary_name "
            "FROM mcp_workspace_servers ws "
            "JOIN mcp_server_catalog cat ON cat.id = ws.server_id "
            "WHERE ws.workspace_id = ? AND ws.name = ? AND ws.enabled = TRUE",
            (workspace_id, config_name),
        )

        # Fallback: match by catalog server name (backwards compat).
        # When multiple configs share a catalog name, fail with a
        # clear message instead of picking one at random.
        if not row:
            rows = store.query_all(
                "SELECT ws.name, ws.env_config, ws.args_config, "
                "       cat.name AS catalog_name, cat.package, cat.binary_name "
                "FROM mcp_workspace_servers ws "
                "JOIN mcp_server_catalog cat ON cat.id = ws.server_id "
                "WHERE ws.workspace_id = ? AND cat.name = ? AND ws.enabled = TRUE",
                (workspace_id, config_name),
            )
            if len(rows) > 1:
                names = ", ".join(r["name"] for r in rows)
                raise ValueError(
                    f"Multiple configurations found for server '{config_name}'. "
                    f"Use a specific config name: {names}"
                )
            row = rows[0] if rows else None

        if not row:
            return {}

        import json

        env_config = row.get("env_config", {})
        if isinstance(env_config, str):
            try:
                env_config = json.loads(env_config)
            except (json.JSONDecodeError, TypeError):
                env_config = {}

        args_config = row.get("args_config", [])
        if isinstance(args_config, str):
            try:
                args_config = json.loads(args_config)
            except (json.JSONDecodeError, TypeError):
                args_config = []

        if not isinstance(env_config, dict):
            env_config = {}
        if not isinstance(args_config, list):
            args_config = []

        # Expand ${VAR} references against context (which includes
        # workspace secrets loaded by the executor)
        resolved_env = expand_env_vars(env_config, context)
        resolved_args = [expand_env_vars(a, context) if isinstance(a, str) else a for a in args_config]

        return {
            "env": resolved_env,
            "args": resolved_args,
            "catalog_name": row.get("catalog_name", config_name),
            "package": row.get("package"),
            "binary_name": row.get("binary_name"),
        }
    except Exception:
        import logging
        logging.getLogger("aiorch.mcp").debug(
            "Failed to load workspace config '%s' in workspace %s",
            config_name, workspace_id,
        )
        return {}


async def _resolve_registry_server(
    srv_config: dict,
    context: dict[str, Any] | None = None,
) -> tuple[list[ToolDefinition], RegistryMCPSession | None]:
    """Resolve a ``uses:`` spec via the MCP registry.

    The ``uses:`` key now resolves by name from ``mcp_workspace_servers``
    (the user's named configurations), not from the catalog directly.

    1. Look up mcp_workspace_servers WHERE workspace_id=X AND name="uses"
    2. Join with mcp_server_catalog for package/binary
    3. Resolve env_config + args_config via workspace secrets
    4. Merge pipeline-level env/args overrides on top
    5. POST to registry with resolved env + args
    6. Convert discovered tools to ToolDefinitions
    """
    from aiorch.runtime.run_env import expand_env_vars

    config_name = srv_config["uses"]
    registry_url = _os.environ.get("MCP_REGISTRY_URL", _MCP_REGISTRY_URL)

    # Look up named workspace configuration (falls back to catalog name)
    workspace_id = context.get("__workspace_id__") if context else None
    ws_config = _get_mcp_workspace_config(workspace_id, config_name, context)

    # The server name for the registry is the catalog name, not the
    # user's config name (e.g. "postgres", not "prod-db")
    server_name = ws_config.get("catalog_name", config_name)

    # Resolve args: workspace args_config + pipeline-level overrides
    ws_args = ws_config.get("args", [])
    raw_args = srv_config.get("args", [])
    from aiorch.core import template as _tpl
    pipeline_args = []
    for a in raw_args:
        if isinstance(a, str):
            resolved = a
            if context and "{{" in a:
                try:
                    resolved = _tpl.resolve(a, context)
                except Exception:
                    pass
            resolved = expand_env_vars(resolved, context)
            pipeline_args.append(resolved)
        else:
            pipeline_args.append(a)

    # Pipeline args override workspace args when provided
    args = pipeline_args if pipeline_args else ws_args

    # Merge env: workspace config < pipeline-level overrides
    ws_env = ws_config.get("env", {})
    pipeline_env = expand_env_vars(srv_config.get("env", {}), context)
    merged_env = {**ws_env, **pipeline_env}

    # Create session via registry (with retry on transient failures)
    resp = await _registry_request(
        "POST",
        f"{registry_url}/sessions",
        json_body={
            "server": server_name,
            "args": args,
            "env": merged_env,
            "workspace_id": workspace_id,
            "config_name": config_name,
        },
        timeout=30,
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            _humanize_registry_error(resp, f"could not create MCP session for '{config_name}'")
        )
    session_data = resp.json()

    # R3.2: capture the worker_url returned by the registry so tool
    # calls route directly to the replica that owns the child process.
    # In single-replica deployments worker_url == registry_url, so
    # behavior is unchanged. In multi-replica deployments, this is the
    # routing fix: subsequent calls skip the LB and go direct.
    worker_url = session_data.get("worker_url") or registry_url

    # Cross-worker pool hits return tools=[] because the remote worker
    # has the authoritative list. Fetch it lazily from the remote
    # worker here so the agent gets a complete toolset at resolve time.
    tools_from_response = session_data.get("tools", [])
    if session_data.get("reused") and not tools_from_response and worker_url != registry_url:
        try:
            sess_id = session_data["session_id"]
            info_resp = await _registry_request(
                "GET", f"{worker_url}/sessions", timeout=5, retry_on_5xx=False,
            )
            if info_resp.status_code == 200:
                for s in info_resp.json():
                    if s.get("id") == sess_id and s.get("tools_count") is not None:
                        # tools/list call on the remote worker to fetch full list
                        # would require a new endpoint; for now we accept an
                        # empty tool list on cross-worker reuse as a rare edge
                        # case. Single-worker reuse returns the full list.
                        break
        except Exception:
            pass

    session = RegistryMCPSession(
        registry_url=registry_url,
        session_id=session_data["session_id"],
        tools=tools_from_response,
        server_name=server_name,
        worker_url=worker_url,
    )

    # Convert registry tools to ToolDefinitions
    tool_defs = []
    for t in session.raw_tools:
        def make_fn(s, name):
            async def _call(**kwargs):
                return await s.call_tool(name, kwargs)
            return _call

        tool_defs.append(ToolDefinition(
            name=t["name"],
            description=t.get("description", ""),
            parameters=t.get("inputSchema", t.get("input_schema", {"type": "object", "properties": {}})),
            fn=make_fn(session, t["name"]),
            mcp_session=session,
        ))

    import logging
    logging.getLogger("aiorch.mcp").info(
        "MCP config '%s' (catalog: %s): session %s, %d tools",
        config_name, server_name, session.session_id, len(tool_defs),
    )

    return tool_defs, session


def tools_to_openai_schema(tools: list[ToolDefinition]) -> list[dict]:
    """Convert a list of ToolDefinitions to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


async def execute_tool(tool: ToolDefinition, arguments: dict) -> ToolResult:
    """Execute a tool with the given arguments and return a ToolResult.

    Handles both sync and async tool functions. Catches all exceptions
    and returns them as ToolResult.error.
    """
    try:
        result = tool.fn(**arguments)
        if asyncio.iscoroutine(result):
            result = await result
        return ToolResult(tool_name=tool.name, output=str(result))
    except Exception as e:
        return ToolResult(tool_name=tool.name, output="", error=str(e))
