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

"""Built-in agent system prompts and definitions.

All system prompts live here — not embedded in runtime code.
To add a new built-in agent: add an entry to BUILTIN_AGENTS.

Security note
-------------
Earlier versions shipped these agents with ``file_read``,
``file_search``, and ``run_command`` enabled by default. Those host-
access tools were removed from the built-in registry in the security
hardening pass (see ``aiorch.tools`` module docstring). The agents
below now ship with an empty ``tools`` list so they do not reference
removed tools.

If you want these agents to actually read code, add file access via
MCP at the step level:

    steps:
      review:
        agent: code-reviewer
        tools:
          mcp: "npx -y @modelcontextprotocol/server-filesystem /path/to/repo"
        input: "Review the src/ directory"
"""

from __future__ import annotations

from typing import Any


_MCP_HINT = (
    " Note: this agent does not ship with any tools by default. "
    "To give it file or shell access, attach an MCP server at the "
    "step level via `tools: { mcp: ... }`."
)


BUILTIN_AGENTS: dict[str, dict[str, Any]] = {
    "code-reviewer": {
        "system": (
            "You are a senior code reviewer. Given source code in the "
            "user message, provide specific, actionable feedback. Reference "
            "file names and line numbers where possible. Focus on bugs, "
            "logic errors, and maintainability." + _MCP_HINT
        ),
        "tools": [],
    },
    "security-auditor": {
        "system": (
            "You are a security engineer performing a code audit. Given "
            "source code in the user message, find vulnerabilities including "
            "injection risks, hardcoded secrets, unsafe file operations, "
            "authentication issues, and data exposure. Rate findings by "
            "severity." + _MCP_HINT
        ),
        "tools": [],
    },
    "test-generator": {
        "system": (
            "You are a test engineer. Given source code in the user message, "
            "generate comprehensive test cases covering happy paths, edge "
            "cases, and error conditions. Use the project's existing test "
            "framework and conventions." + _MCP_HINT
        ),
        "tools": [],
    },
    "doc-writer": {
        "system": (
            "You are a technical writer. Given source code in the user "
            "message, generate clear, concise documentation. Include: what "
            "the code does, how to use it, parameters, return values, and "
            "examples." + _MCP_HINT
        ),
        "tools": [],
    },
}
