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

"""Error classification for step failures."""

from __future__ import annotations


def classify_error(error: Exception, primitive_type: str) -> str:
    """Classify an exception into a typed error category."""
    err_str = str(error).lower()
    err_type = type(error).__name__.lower()

    # Rate limit / auth / API errors from LLM providers
    if any(x in err_str for x in ['rate_limit', 'ratelimit', '429', 'quota', 'insufficient_quota']):
        return 'model_error'
    if any(x in err_str for x in ['authentication', 'unauthorized', '401', 'invalid_api_key']):
        return 'model_error'
    if any(x in err_str for x in ['api_error', 'server_error', '500', '503', 'overloaded']):
        return 'model_error'
    if 'openai' in err_type or 'anthropic' in err_type:
        return 'model_error'

    # Schema/validation errors
    if any(x in err_str for x in ['validation failed', 'schema', 'retry_on_invalid', 'jsondecodeerror', 'invalid json']):
        return 'schema_error'
    if 'validationerror' in err_type:
        return 'schema_error'

    # Tool/MCP errors
    if any(x in err_str for x in ['mcp error', 'unknown tool', 'tool_error']):
        return 'tool_error'
    if '[error]' in err_str and primitive_type == 'agent':
        return 'tool_error'

    # Timeout
    if any(x in err_str for x in ['timeout', 'timed out', 'deadline']):
        return 'timeout_error'
    if 'timeouterror' in err_type or 'asyncio.timeouterror' in err_type:
        return 'timeout_error'

    # Command/shell errors
    if primitive_type == 'run':
        return 'command_error'
    if any(x in err_str for x in ['exit code', 'non-zero', 'command failed', 'subprocess']):
        return 'command_error'

    # Default based on primitive
    if primitive_type in ('prompt', 'agent'):
        return 'model_error'
    if primitive_type == 'action':
        return 'action_error'

    return 'runtime_error'
