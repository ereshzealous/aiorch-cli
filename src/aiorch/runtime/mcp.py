# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""MCP (Model Context Protocol) client — stdio + HTTP transports.

Connects to MCP servers via subprocess (stdio) or HTTP (Streamable HTTP/SSE).
HTTP is the recommended transport for production — no subprocess overhead.

Usage in YAML:
    steps:
      analyze:
        agent:
          model: gpt-4o-mini
          tools:
            mcp:
              # HTTP transport (recommended for production)
              - server: "https://mcp.company.com/tools"
                headers:
                  Authorization: "Bearer ${MCP_TOKEN}"

              # stdio transport (for local/dev)
              - server: "npx -y @modelcontextprotocol/server-filesystem"
                args: ["/tmp"]
          prompt: "List files and summarize"
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("aiorch.mcp")

_next_id = 0


def _get_id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


@dataclass
class MCPTool:
    """A tool discovered from an MCP server."""
    name: str
    description: str
    input_schema: dict
    server_key: str


class MCPTransport(ABC):
    """Abstract transport for MCP communication."""

    @abstractmethod
    async def send_request(self, method: str, params: dict | None = None) -> Any: ...

    @abstractmethod
    async def send_notification(self, method: str, params: dict | None = None) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Stdio transport — spawns a subprocess
# ---------------------------------------------------------------------------

class StdioTransport(MCPTransport):
    """MCP transport over subprocess stdin/stdout."""

    def __init__(self, process: asyncio.subprocess.Process):
        self._process = process
        self._read_lock = asyncio.Lock()

    async def send_request(self, method: str, params: dict | None = None) -> Any:
        msg_id = _get_id()
        request = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params:
            request["params"] = params

        data = json.dumps(request) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

        async with self._read_lock:
            while True:
                line = await asyncio.wait_for(self._process.stdout.readline(), timeout=30)
                if not line:
                    raise ConnectionError("MCP server closed connection")
                line = line.decode().strip()
                if not line:
                    continue
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" not in response:
                    continue
                if response.get("id") == msg_id:
                    if "error" in response:
                        err = response["error"]
                        raise RuntimeError(f"MCP error: {err.get('message', err)}")
                    return response.get("result")

    async def send_notification(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params:
            msg["params"] = params
        data = json.dumps(msg) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

    async def close(self) -> None:
        try:
            self._process.stdin.close()
            self._process.terminate()
            await asyncio.wait_for(self._process.wait(), timeout=5)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# HTTP transport — Streamable HTTP (MCP 2025 spec)
# ---------------------------------------------------------------------------

class HttpTransport(MCPTransport):
    """MCP transport over HTTP — no subprocess, async HTTP calls.

    Uses the MCP Streamable HTTP protocol:
    - POST requests for JSON-RPC messages
    - SSE responses for streaming results
    - Session management via Mcp-Session-Id header
    """

    def __init__(self, url: str, headers: dict[str, str] | None = None):
        self._url = url.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(headers or {}),
        }
        self._session_id: str | None = None
        self._client = None

    def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=30, follow_redirects=False)
        return self._client

    async def send_request(self, method: str, params: dict | None = None) -> Any:
        msg_id = _get_id()
        request = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params:
            request["params"] = params

        headers = dict(self._headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        client = self._get_client()
        resp = await client.post(self._url, json=request, headers=headers)

        # Capture session ID from response
        if "mcp-session-id" in resp.headers:
            self._session_id = resp.headers["mcp-session-id"]

        if resp.status_code >= 400:
            raise RuntimeError(f"MCP HTTP error {resp.status_code}: {resp.text[:200]}")

        content_type = resp.headers.get("content-type", "")

        # JSON response (simple request/response)
        if "application/json" in content_type:
            data = resp.json()
            if "error" in data:
                err = data["error"]
                raise RuntimeError(f"MCP error: {err.get('message', err)}")
            return data.get("result")

        # SSE response (streaming) — collect all data events
        if "text/event-stream" in content_type:
            result = None
            for line in resp.text.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("id") == msg_id:
                            if "error" in data:
                                err = data["error"]
                                raise RuntimeError(f"MCP error: {err.get('message', err)}")
                            result = data.get("result")
                    except json.JSONDecodeError:
                        continue
            return result

        # Fallback: try to parse as JSON
        try:
            data = resp.json()
            return data.get("result")
        except Exception:
            raise RuntimeError(f"Unexpected MCP response type: {content_type}")

    async def send_notification(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params:
            msg["params"] = params

        headers = dict(self._headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        client = self._get_client()
        await client.post(self._url, json=msg, headers=headers)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# MCPSession — transport-agnostic session
# ---------------------------------------------------------------------------

@dataclass
class MCPSession:
    """An active MCP server session (works with any transport)."""
    transport: MCPTransport
    server_key: str
    tools: list[MCPTool] = field(default_factory=list)

    async def send_request(self, method: str, params: dict | None = None) -> Any:
        return await self.transport.send_request(method, params)

    async def close(self):
        await self.transport.close()


def _is_url(server: str) -> bool:
    """Check if a server spec is a URL (HTTP transport) or command (stdio)."""
    return server.startswith("http://") or server.startswith("https://")


async def start_mcp_session(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> MCPSession:
    """Start an MCP session — auto-detects transport from the server string.

    - URLs (http:// or https://) → HTTP transport (no subprocess)
    - Commands (npx, python, etc.) → stdio transport (subprocess)

    Args:
        command: URL or command to run
        args: Additional arguments (stdio only)
        env: Environment variables (stdio only)
        headers: HTTP headers for auth (HTTP only)

    Returns:
        An initialized MCPSession with discovered tools
    """
    if _is_url(command):
        return await _start_http_session(command, headers=headers)
    else:
        return await _start_stdio_session(command, args=args, env=env)


async def _start_http_session(url: str, headers: dict[str, str] | None = None) -> MCPSession:
    """Start an MCP session over HTTP."""
    # SSRF gate (Round 2 #14 in the audit): MCP server URLs come
    # from pipeline YAML and can be templated from context, so an
    # attacker-controlled value could point the executor at IMDS,
    # internal services, or non-http schemes. safe_http_url
    # enforces the scheme allowlist and private-host block.
    # Local MCP servers (e.g. http://localhost:8765) are a real
    # use case, so operators with that need set
    # AIORCH_ALLOW_PRIVATE_HOSTS=1.
    from aiorch.core.http_safety import safe_header_value, safe_http_url
    url = safe_http_url(url, purpose="MCP HTTP transport")
    if headers:
        headers = {
            k: safe_header_value(v, name=str(k), purpose="MCP HTTP transport")
            for k, v in headers.items()
        }

    server_key = url.split("//")[-1].split("/")[0]  # domain as key
    logger.info("Connecting to MCP server (HTTP): %s", url)

    transport = HttpTransport(url, headers=headers)
    session = MCPSession(transport=transport, server_key=server_key)

    # MCP handshake
    try:
        await transport.send_request("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "aiorch", "version": "0.4.1"},
        })
        await transport.send_notification("notifications/initialized")
    except Exception as e:
        await transport.close()
        raise RuntimeError(f"MCP HTTP initialization failed for {url}: {e}") from e

    # Discover tools
    try:
        result = await transport.send_request("tools/list")
        tools_data = result.get("tools", []) if result else []
        for t in tools_data:
            session.tools.append(MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
                server_key=server_key,
            ))
        logger.info("MCP HTTP %s: discovered %d tools", server_key, len(session.tools))
    except Exception as e:
        logger.warning("MCP tool discovery failed for %s: %s", server_key, e)

    return session


def _resolve_mcp_command(command: str) -> str:
    """Resolve npx-style MCP commands to pre-installed binaries.

    In production (Docker), MCP servers are pre-installed as global npm
    packages. This function rewrites `npx -y @modelcontextprotocol/server-X`
    to just `mcp-server-X` (the global binary name). If the binary isn't
    found on PATH, it falls back to the original npx command so local dev
    still works.

    This means pipeline YAML doesn't need to change between dev and
    production — the same `npx -y @modelcontextprotocol/server-github`
    works everywhere, but in Docker it's instant (no download).
    """
    import shutil
    import re

    # Match: npx -y @modelcontextprotocol/server-<name>
    m = re.match(r'^npx\s+(?:-y\s+)?@modelcontextprotocol/(server-\w+)$', command.strip())
    if m:
        binary = f"mcp-{m.group(1)}"  # e.g. mcp-server-github
        if shutil.which(binary):
            logger.debug("Resolved MCP command: %s → %s", command, binary)
            return binary
        # Also try without mcp- prefix (some packages register both)
        alt = m.group(1)  # e.g. server-github
        if shutil.which(alt):
            logger.debug("Resolved MCP command: %s → %s", command, alt)
            return alt
    return command


async def _start_stdio_session(
    command: str, args: list[str] | None = None, env: dict[str, str] | None = None,
) -> MCPSession:
    """Start an MCP session over stdio (subprocess)."""
    import os
    import shlex

    command = _resolve_mcp_command(command)
    parts = shlex.split(command)
    if args:
        parts.extend(args)

    proc_env = dict(os.environ)
    if env:
        proc_env.update(env)

    server_key = command.split()[-1] if command else "unknown"
    logger.info("Starting MCP server (stdio): %s", " ".join(parts))

    process = await asyncio.create_subprocess_exec(
        *parts,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=proc_env,
    )

    transport = StdioTransport(process)
    session = MCPSession(transport=transport, server_key=server_key)

    # MCP handshake
    try:
        await transport.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "aiorch", "version": "0.4.1"},
        })
        await transport.send_notification("notifications/initialized")
    except Exception as e:
        await transport.close()
        raise RuntimeError(f"MCP stdio initialization failed: {e}") from e

    # Discover tools
    try:
        result = await transport.send_request("tools/list")
        tools_data = result.get("tools", []) if result else []
        for t in tools_data:
            session.tools.append(MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
                server_key=server_key,
            ))
        logger.info("MCP stdio %s: discovered %d tools", server_key, len(session.tools))
    except Exception as e:
        logger.warning("MCP tool discovery failed for %s: %s", server_key, e)

    return session


async def call_mcp_tool(session: MCPSession, tool_name: str, arguments: dict) -> str:
    """Execute a tool on an MCP server and return the result as a string."""
    try:
        result = await session.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        if not result:
            return ""

        content = result.get("content", [])
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    parts.append(f"[Image: {block.get('mimeType', 'image')}]")
                else:
                    parts.append(json.dumps(block))
            else:
                parts.append(str(block))

        output = "\n".join(parts)

        if result.get("isError"):
            return f"[MCP Error]: {output}"

        return output

    except Exception as e:
        return f"[MCP Error]: {e}"
