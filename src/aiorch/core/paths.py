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

"""Path confinement helper.

``safe_path()`` resolves a user-supplied path, follows symlinks, and
rejects anything that lands outside the configured roots. Default
root is process CWD; operators extend it via ``AIORCH_SAFE_ROOTS``
(colon-separated). Callers that want ``/dev/stdout`` and friends
pass ``allow_symbolic=True``.

TOCTOU note: the check and the subsequent write are not atomic — a
racing symlink swap could still defeat this. Proper mitigation needs
``O_NOFOLLOW`` / dirfd anchoring.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final


class PathSafetyError(ValueError):
    """Raised when a user-supplied path cannot be made safe.

    Callers should let this exception propagate — the message is
    written to be directly user-visible.
    """


# Destinations that are not filesystem paths but get passed through
# write_text transparently on Unix. /dev/null is the bit bucket;
# /dev/stdout and /dev/stderr are the running process's file
# descriptors 1 and 2. Writes to these do not touch disk and cannot
# be used for path traversal in any meaningful sense.
_SYMBOLIC_DESTINATIONS: Final[frozenset[str]] = frozenset({
    "/dev/stdout",
    "/dev/stderr",
    "/dev/null",
})


def _resolve_extra_roots() -> list[Path]:
    """Parse AIORCH_SAFE_ROOTS into a list of resolved absolute paths.

    Malformed entries are silently skipped — we don't want a typo in
    the env var to bring the executor down. Operators can verify
    their configuration by running any pipeline that hits safe_path
    and reading the error message, which lists the roots in effect.
    """
    roots: list[Path] = []
    raw = os.environ.get("AIORCH_SAFE_ROOTS", "")
    if not raw:
        return roots
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        try:
            roots.append(Path(part).expanduser().resolve())
        except (OSError, RuntimeError):
            continue
    return roots


def safe_path(
    user_path: str,
    *,
    purpose: str,
    default_root: Path,
    allow_symbolic: bool = False,
) -> Path:
    """Resolve ``user_path`` and confirm it lands inside an allowed root.

    Args:
        user_path: The template-rendered path string from the
            pipeline. May be relative or absolute.
        purpose: Short human-readable label for error messages —
            typically the runtime sink name (``"save: target"``,
            ``"write-file action target"``, ``"flow: sub-pipeline"``).
        default_root: The primary sandbox for this call site. For
            ``save:`` and ``write-file`` this is ``Path.cwd()``. For
            ``flow:`` it is the parent pipeline's source directory.
        allow_symbolic: If True, ``/dev/stdout``, ``/dev/stderr``,
            and ``/dev/null`` are accepted as-is without confinement
            checks. CLI example pipelines use ``path: /dev/stdout``
            to print to the terminal.

    Returns:
        A ``Path`` known to be inside one of the allowed roots.
        The caller may still need to create parent directories.

    Raises:
        PathSafetyError: with a message that names the rejected
            path, the allowed roots, and the env var fix.
    """
    if not user_path:
        raise PathSafetyError(f"{purpose}: empty path is not allowed")

    if "\x00" in user_path:
        raise PathSafetyError(
            f"{purpose}: path contains a null byte — rejected "
            f"(original: {user_path!r})"
        )

    if allow_symbolic and user_path in _SYMBOLIC_DESTINATIONS:
        return Path(user_path)

    # Resolve against the default root so relative paths work.
    # Path.resolve(strict=False) walks as far as the path exists and
    # appends the remaining components literally, which is what we
    # want for paths we're about to create.
    candidate = Path(user_path).expanduser()
    default_root_resolved = default_root.resolve()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (default_root_resolved / candidate).resolve()

    allowed_roots: list[Path] = [default_root_resolved] + _resolve_extra_roots()

    for root in allowed_roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved

    roots_display = ", ".join(str(r) for r in allowed_roots)
    raise PathSafetyError(
        f"{purpose}: resolved path {resolved} is outside the allowed "
        f"root(s).\n"
        f"  Original input: {user_path!r}\n"
        f"  Allowed roots:  {roots_display}\n\n"
        f"Fix options:\n"
        f"  1. Move the output under an allowed root — use a relative "
        f"path like 'reports/output.md' instead of an absolute path.\n"
        f"  2. Add the target directory to AIORCH_SAFE_ROOTS "
        f"(colon-separated) in the executor's environment.\n"
        f"  3. For CLI output to the terminal, use '/dev/stdout' — "
        f"it's accepted as a symbolic destination on write-file and "
        f"save: sinks."
    )


# ---------------------------------------------------------------------------
# Pipeline directory — process-global "where are the YAML files".
#
# Used by flow:-primitive and schedule-file lookups to resolve relative
# pipeline names. CLI sets it from --pipeline-dir or the YAML dir of the
# invoked file; Platform sets it during app lifespan from
# aiorch.yaml [server] pipeline_dir. Lives here rather than under
# aiorch.server.* so the CLI (OSS) package can own it cleanly.
# ---------------------------------------------------------------------------

_pipeline_dir: Path | None = None


def set_pipeline_dir(d: Path | str | None) -> None:
    global _pipeline_dir
    _pipeline_dir = Path(d) if d else None


def get_pipeline_dir() -> Path:
    """Return the configured pipeline directory, or CWD as default."""
    return _pipeline_dir or Path.cwd()
