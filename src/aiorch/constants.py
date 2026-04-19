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

"""Shared constants — magic strings, numbers, and defaults in one place."""

# --- Preview lengths ---
PROMPT_PREVIEW_LENGTH = 200
OUTPUT_PREVIEW_LENGTH = 200
TRACE_PREVIEW_LENGTH = 500
CONSOLE_PREVIEW_LENGTH = 120

# --- Tool limits ---
FILE_READ_MAX_BYTES = 50 * 1024       # 50KB
COMMAND_TIMEOUT_SECONDS = 30
COMMAND_OUTPUT_MAX_CHARS = 5000
GREP_MAX_LINES = 50

# --- Cost estimation defaults ---
DEFAULT_INPUT_COST_PER_M = 0.15       # $/1M input tokens
DEFAULT_OUTPUT_COST_PER_M = 0.60      # $/1M output tokens
ASSUMED_OUTPUT_TOKENS = 500

# --- Time constants ---
SECONDS_IN_DAY = 86400
SECONDS_IN_WEEK = 7 * 86400

# --- Status values ---
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial"   # succeeded with foreach iteration warnings
STATUS_FAILED = "failed"
STATUS_STARTED = "started"
STATUS_SKIPPED = "skipped"

# Statuses that represent a successfully-completed run (outputs are
# valid and downstream consumers can read them). Partial is included
# because the run reached the end; some foreach slots may be sentinels.
SUCCESSFUL_STATUSES = (STATUS_SUCCESS, STATUS_PARTIAL)

# --- Context keys (internal, not exposed in templates) ---
COST_KEY = "__step_costs__"
META_KEY = "__step_meta__"
LOGGER_KEY = "__logger__"
SOURCE_DIR_KEY = "__source_dir__"
CONFIG_KEY = "__config__"
RUN_ENV_KEY = "__run_env__"  # Per-run configs + secrets, NEVER written to os.environ

# --- User-visible runtime metadata (exposed in templates as {{_meta.key}}) ---
RUNTIME_META_KEY = "_meta"

# --- Default system prompt ---
DEFAULT_AGENT_SYSTEM_PROMPT = "You are a helpful AI agent. Complete the given task."
