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

"""Jinja2-based template resolution for {{var}} references.

Two resolvers live in this module:

1. ``resolve()`` / ``resolve_dict()`` — plain Jinja substitution for
   non-shell contexts (LLM prompts, action config, connector fields).

2. ``resolve_for_shell()`` — shell-safe variant used by the ``run:``
   primitive. Every substituted value is passed through
   ``shlex.quote()`` via Jinja's ``finalize`` callback, so attacker-
   controlled values (webhook bodies, LLM output, DB rows) cannot
   inject shell metacharacters.

The shell-safe resolver has one hard rule: **Jinja expressions must
appear in bare position, not wrapped in shell quotes**. The rule
exists because ``shlex.quote`` outputs a single-quoted token
(``'foo'``), and if the template author already wrapped the expression
in double quotes (``"{{ x }}"``), the inner ``'`` chars still let
attacker-controlled ``"`` chars break out of the user's outer quoting.
A state machine in ``_scan_shell_context`` walks the template source
and raises ``ShellTemplateError`` if it finds any ``{{`` inside a
``"..."`` or ``'...'`` span. Fix: rewrite the template to use bare
``{{ x }}`` (no surrounding quotes).
"""

from __future__ import annotations

import shlex
from typing import Any

from jinja2 import BaseLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment


_env = SandboxedEnvironment(
    loader=BaseLoader(),
    undefined=StrictUndefined,
    variable_start_string="{{",
    variable_end_string="}}",
    keep_trailing_newline=True,
)


def _template_context(context: dict[str, Any]) -> dict[str, Any]:
    """Build template-safe context: user variables + _meta, excluding __ internals."""
    return {k: v for k, v in context.items() if not k.startswith("__")}


def resolve(template_str: str, context: dict[str, Any]) -> str:
    """Resolve {{var}} references in a string using the given context.

    Internal __ keys (logger, costs, provider resolution) are excluded.
    User-visible metadata is available via {{_meta.run_id}}, {{_meta.total_cost}}, etc.
    """
    if "{{" not in template_str:
        return template_str
    tmpl = _env.from_string(template_str)
    return tmpl.render(**_template_context(context))


def resolve_dict(data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve all string values in a dict."""
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = resolve(value, context)
        elif isinstance(value, dict):
            result[key] = resolve_dict(value, context)
        elif isinstance(value, list):
            result[key] = [
                resolve(item, context) if isinstance(item, str) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def has_variables(text: str) -> bool:
    """Check if a string contains unresolved {{var}} references."""
    return "{{" in text and "}}" in text


# ---------------------------------------------------------------------------
# Shell-safe resolver (used by the `run:` primitive)
# ---------------------------------------------------------------------------


class ShellTemplateError(ValueError):
    """Raised when a run: template has an unsafe structure that the
    shell-safe resolver cannot make safe by auto-quoting alone.

    The canonical case: the template author wrapped a Jinja expression
    in shell quotes (``"{{ x }}"`` or ``'{{ x }}'``). Auto-quoting the
    value doesn't protect against metacharacters there, because the
    shlex-quoted output can break out of the author's outer quoting.
    """


class _Raw(str):
    """Marker str subclass that bypasses shell quoting.

    Used by the ``raw`` filter for cases where the author explicitly
    wants an unquoted substitution — typically inside a quoted heredoc
    body where the shell does not re-parse the content. Misuse of
    ``| raw`` in normal shell position defeats the safety — grep for it
    during code review.
    """


def _raw_filter(value: Any) -> _Raw:
    """Jinja filter: wrap a value so the shell-safe finalize leaves it
    unquoted. Example: ``{{ data | tojson | raw }}`` inside a quoted
    Python heredoc."""
    if isinstance(value, str):
        return _Raw(value)
    return _Raw(str(value))


def _shell_finalize(value: Any) -> str:
    """Jinja ``finalize`` callback for shell-safe substitution.

    Every ``{{ expr }}`` in a run: template passes through this
    function with the final evaluated value (after all filters). The
    return value is spliced into the command string as-is.

    Rules:
      - ``_Raw`` marker: pass through without quoting (escape hatch).
      - ``None``: render as empty single-quoted string (valid empty argv).
      - non-string: serialize via ``json.dumps`` when possible so
        structured values (lists, dicts) produce deterministic output;
        fall back to ``str()`` otherwise.
      - string: ``shlex.quote`` — produces a single-quoted token that
        the shell parses as one literal argument.
    """
    if isinstance(value, _Raw):
        return str(value)
    if value is None:
        return "''"
    if not isinstance(value, str):
        try:
            import json as _json

            value = _json.dumps(value)
        except (TypeError, ValueError):
            value = str(value)
    return shlex.quote(value)


_shell_env = SandboxedEnvironment(
    loader=BaseLoader(),
    undefined=StrictUndefined,
    variable_start_string="{{",
    variable_end_string="}}",
    keep_trailing_newline=True,
    finalize=_shell_finalize,
)
_shell_env.filters["raw"] = _raw_filter
# Reuse tojson from the plain env so `{{ x | tojson }}` works in
# shell templates too. The output is still shell-quoted by finalize.
_shell_env.filters["tojson"] = _env.filters["tojson"]


def _scan_shell_context(template_str: str) -> None:
    """Walk the template source and reject Jinja expressions that
    appear inside shell quotes.

    This is a small state machine, not a full shell parser. It tracks
    three states — NORMAL (bare), DQ (inside ``"..."``), SQ (inside
    ``'...'``) — and only looks for ``{{`` within DQ and SQ spans.
    It deliberately does not track heredoc bodies: no in-repo pipeline
    uses ``{{ }}`` inside a heredoc today, and adding that would
    require escaping rules for unquoted heredocs which have their own
    subtleties. Follow-up work if needed.

    Raises:
        ShellTemplateError: with line number and a fix suggestion.
    """
    state = "NORMAL"
    i = 0
    n = len(template_str)
    while i < n:
        c = template_str[i]
        nxt = template_str[i + 1] if i + 1 < n else ""

        if state == "NORMAL":
            if c == "\\" and nxt:
                i += 2
                continue
            if c == '"':
                state = "DQ"
                i += 1
                continue
            if c == "'":
                state = "SQ"
                i += 1
                continue
            if c == "{" and nxt == "{":
                # Bare-position expression — fast-forward past `}}`.
                end = template_str.find("}}", i + 2)
                if end == -1:
                    return  # malformed, let Jinja raise the real error
                i = end + 2
                continue
            i += 1
            continue

        if state == "DQ":
            if c == "\\" and nxt:
                i += 2
                continue
            if c == '"':
                state = "NORMAL"
                i += 1
                continue
            if c == "{" and nxt == "{":
                _raise_quoted_jinja(template_str, i, quote_kind="double")
            i += 1
            continue

        if state == "SQ":
            # Inside single quotes, the only thing that ends the span
            # is a literal `'`. Backslashes are NOT escape characters
            # in shell single quotes.
            if c == "'":
                state = "NORMAL"
                i += 1
                continue
            if c == "{" and nxt == "{":
                _raise_quoted_jinja(template_str, i, quote_kind="single")
            i += 1
            continue


def _raise_quoted_jinja(source: str, offset: int, *, quote_kind: str) -> None:
    """Build a helpful error for Jinja-inside-shell-quotes and raise."""
    lineno = source[:offset].count("\n") + 1
    line_start = source.rfind("\n", 0, offset) + 1
    line_end = source.find("\n", offset)
    if line_end == -1:
        line_end = len(source)
    line_text = source[line_start:line_end]
    col = offset - line_start + 1
    pointer = " " * (col - 1) + "^"

    if quote_kind == "double":
        why = (
            "Auto-quoting with shlex cannot protect the value — the "
            "shlex-quoted output can break out of the surrounding "
            "double quotes. Rewrite the expression in bare position."
        )
    else:
        why = (
            "Shell single quotes prevent any substitution at all, so "
            "the Jinja expression would not be interpreted even if it "
            "were safe. Rewrite the expression in bare position."
        )

    raise ShellTemplateError(
        f"Jinja expression found inside {quote_kind}-quoted shell "
        f"string at line {lineno}, column {col}.\n\n"
        f"    {line_text}\n"
        f"    {pointer}\n\n"
        f"{why}\n\n"
        f"Fix: drop the surrounding quotes around the expression.\n"
        f"  Before:  run: echo \"hello {{{{ name }}}}\"\n"
        f"  After:   run: echo hello {{{{ name }}}}\n\n"
        f"shlex.quote handles spaces and special characters in the "
        f"value automatically — the surrounding quotes are not "
        f"needed and actually defeat the protection."
    )


def resolve_for_shell(template_str: str, context: dict[str, Any]) -> str:
    """Resolve a Jinja template destined for a ``run:`` shell command.

    Every ``{{ expr }}`` substitution is auto-quoted via
    ``shlex.quote`` so attacker-controlled values cannot inject shell
    metacharacters. Values passed through the ``| raw`` filter skip
    quoting — use it only inside quoted heredoc bodies where the
    shell does not re-parse the content.

    Raises:
        ShellTemplateError: if the template wraps any Jinja expression
            in shell quotes (``"{{ x }}"`` or ``'{{ x }}'``). The
            error message includes the line and column and a fix hint.
    """
    if "{{" not in template_str:
        return template_str
    _scan_shell_context(template_str)
    tmpl = _shell_env.from_string(template_str)
    return tmpl.render(**_template_context(context))
