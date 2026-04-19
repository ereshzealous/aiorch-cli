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

"""HTTP surface hardening — validation helpers for outbound requests.

The audit identified three HTTP-side vectors in ``runtime/action.py``:

  - Bug 6 (webhook action URL SSRF): the ``url`` field is a template-
    rendered string that went straight to ``httpx.request()``. An
    attacker-controlled value could target ``http://169.254.169.254``
    (AWS IMDS) for cloud credential theft, ``http://localhost:6379``
    to probe internal Redis, or ``file://`` / ``gopher://`` for
    protocol smuggling.

  - Bug 7 (webhook action CRLF header injection): header values are
    also template-rendered. A value containing ``\\r\\n`` splits the
    HTTP request and enables response splitting / request smuggling.

  - Bug 10 (GitHub action URL interpolation): ``repo`` and ``pr``
    values are f-stringed into the GitHub API URL. Without
    validation, a malformed value could produce an unintended API
    request path.

All three are fixed by input-validation helpers in this module. No
runtime I/O behaviour changes on the happy path — legitimate URLs,
headers, and GitHub identifiers pass through unchanged. The helpers
only raise on structurally unsafe input.

Design notes
------------

- ``safe_http_url`` blocks non-http/https schemes unconditionally
  and, by default, rejects hostnames that resolve to private /
  loopback / link-local / reserved ranges. Operators who need to
  call internal services from pipelines opt out via
  ``AIORCH_ALLOW_PRIVATE_HOSTS=1`` — that's a workspace-wide
  decision the operator accepts, not a per-request choice.

- DNS-rebinding note: ``safe_http_url`` resolves the hostname once
  at check time. An attacker who controls DNS can return a public
  IP on the first lookup and a private IP on the actual connect.
  Full mitigation requires pinning the resolved IP and rewriting
  the Host header, which is a bigger refactor. This check still
  catches static attacker-supplied IPs, accidental internal-URL
  pipelines, and DNS records pointing at private space.

- ``safe_header_value`` rejects any header value containing ASCII
  control characters. That covers CR / LF (the smuggling vector)
  and also null bytes (the truncation vector).

- GitHub validators use conservative regexes matching GitHub's own
  rules. Any deviation raises before the API URL is assembled.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from typing import Iterable
from urllib.parse import urlparse


class HttpSafetyError(ValueError):
    """Raised when a URL, header value, or API identifier fails
    validation. Callers should let this exception propagate — the
    message is written to be directly user-visible.
    """


_DEFAULT_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# GitHub owner / repo name rules (conservative):
#   - Owner: 1-39 chars, alphanumeric + dash, cannot start/end with dash
#   - Repo:  1-100 chars, alphanumeric + dash/dot/underscore, cannot start with dot
#
# We keep these strict enough to prevent URL injection (no slashes in
# the middle, no percent-encoded traversal, no whitespace).
_GITHUB_REPO_RE: re.Pattern[str] = re.compile(
    r"^[A-Za-z0-9]"              # owner first char
    r"[A-Za-z0-9-]{0,38}"        # owner rest
    r"/"                         # separator
    r"[A-Za-z0-9_][A-Za-z0-9._-]{0,99}"  # repo name
    r"$"
)

_GITHUB_PR_RE: re.Pattern[str] = re.compile(r"^[1-9][0-9]{0,9}$")


def _env_allows_private_hosts() -> bool:
    """Return True if the operator opted into allowing private /
    loopback / link-local hostnames from pipeline HTTP actions."""
    return os.environ.get("AIORCH_ALLOW_PRIVATE_HOSTS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP belongs to a non-public range that we
    block by default to prevent SSRF."""
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolve_hostname_ips(hostname: str) -> list[ipaddress._BaseAddress]:
    """Resolve a hostname to all its addresses. Returns an empty
    list if resolution fails — the caller decides whether to treat
    that as fatal."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, UnicodeError):
        return []
    out: list[ipaddress._BaseAddress] = []
    seen: set[str] = set()
    for info in infos:
        addr = info[4][0]
        if addr in seen:
            continue
        seen.add(addr)
        try:
            out.append(ipaddress.ip_address(addr))
        except ValueError:
            continue
    return out


def safe_http_url(
    url: str,
    *,
    purpose: str,
    allowed_schemes: Iterable[str] = _DEFAULT_ALLOWED_SCHEMES,
    allow_private: bool | None = None,
) -> str:
    """Validate a URL destined for an outbound HTTP request.

    Args:
        url: The URL string to validate.
        purpose: Short label for error messages — typically the
            action name (``"webhook action"``, ``"slack action"``,
            ``"github-comment action"``).
        allowed_schemes: Iterable of allowed URL schemes. Default
            is ``{"http", "https"}``. No other protocols are ever
            allowed — ``file``, ``gopher``, ``dict``, ``ftp``, and
            friends are outright rejected.
        allow_private: Whether to allow hostnames resolving to
            private / loopback / link-local ranges. ``None`` (the
            default) reads ``AIORCH_ALLOW_PRIVATE_HOSTS`` from the
            environment.

    Returns:
        The validated URL unchanged (so callers can chain it into
        ``httpx.request(url=safe_http_url(...), ...)``).

    Raises:
        HttpSafetyError: with a message that names the purpose,
            the offending URL, and the specific reason.
    """
    if not url:
        raise HttpSafetyError(f"{purpose}: empty URL is not allowed")
    if any(c in url for c in ("\r", "\n", "\x00")):
        raise HttpSafetyError(
            f"{purpose}: URL contains control characters — rejected "
            f"(original: {url!r})"
        )

    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise HttpSafetyError(f"{purpose}: URL is malformed: {exc}") from exc

    scheme = (parsed.scheme or "").lower()
    if scheme not in allowed_schemes:
        allowed_display = ", ".join(sorted(allowed_schemes))
        raise HttpSafetyError(
            f"{purpose}: URL scheme {scheme!r} is not allowed.\n"
            f"  URL:             {url}\n"
            f"  Allowed schemes: {allowed_display}\n\n"
            f"Only http:// and https:// URLs are accepted for "
            f"outbound HTTP actions. file://, gopher://, dict://, "
            f"ftp://, and similar protocols are blocked to prevent "
            f"SSRF-style protocol smuggling."
        )

    hostname = parsed.hostname
    if not hostname:
        raise HttpSafetyError(
            f"{purpose}: URL has no hostname (original: {url!r})"
        )

    effective_allow_private = (
        allow_private if allow_private is not None else _env_allows_private_hosts()
    )
    if effective_allow_private:
        return url

    # Literal IP or DNS name?
    literal_ip: ipaddress._BaseAddress | None = None
    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        if _ip_is_blocked(literal_ip):
            _raise_blocked_host(purpose, url, hostname, literal_ip)
        return url

    # Common private hostnames that don't necessarily resolve to a
    # private IP in every DNS config — block them by name to be safe.
    if hostname.lower() in {"localhost", "localhost.localdomain", "broadcasthost"}:
        _raise_blocked_host(purpose, url, hostname, None)

    # DNS lookup
    ips = _resolve_hostname_ips(hostname)
    for ip in ips:
        if _ip_is_blocked(ip):
            _raise_blocked_host(purpose, url, hostname, ip)

    return url


def _raise_blocked_host(
    purpose: str,
    url: str,
    hostname: str,
    ip: ipaddress._BaseAddress | None,
) -> None:
    ip_display = f" (resolved to {ip})" if ip is not None else ""
    raise HttpSafetyError(
        f"{purpose}: host {hostname!r}{ip_display} is in a "
        f"private / loopback / link-local / reserved range. "
        f"Blocked by default to prevent SSRF.\n"
        f"  URL: {url}\n\n"
        f"If the pipeline genuinely needs to call an internal "
        f"service, set AIORCH_ALLOW_PRIVATE_HOSTS=1 on the executor "
        f"process. This is a workspace-wide decision — weigh the "
        f"SSRF risk for your deployment before enabling it."
    )


def safe_header_value(value: str, *, name: str, purpose: str) -> str:
    """Validate an HTTP header value.

    Rejects any value containing ASCII control characters (C0 +
    DEL, which covers ``\\r\\n`` header smuggling and null-byte
    truncation). Returns the value unchanged on success.

    Args:
        value: Header value to validate.
        name: Header name, for error messages.
        purpose: Caller label (usually action name).

    Raises:
        HttpSafetyError: if the value contains control characters.
    """
    if value is None:
        return ""
    s = str(value)
    for ch in s:
        code = ord(ch)
        if code < 0x20 or code == 0x7F:
            # Find which character for the error message
            display = repr(ch)
            raise HttpSafetyError(
                f"{purpose}: header {name!r} contains control "
                f"character {display} — rejected.\n"
                f"  Header: {name}\n"
                f"  Value:  {s!r}\n\n"
                f"Control characters (including \\r and \\n) in "
                f"HTTP header values enable request/response "
                f"splitting. Strip them before setting the header, "
                f"or use a different transport for binary data."
            )
    return s


def safe_github_repo(repo: str, *, purpose: str) -> str:
    """Validate a GitHub repository identifier.

    Accepts strings matching ``owner/repo`` where ``owner`` is a
    GitHub username or org name (alphanumeric + dash, max 39
    chars) and ``repo`` is a GitHub repo name (alphanumeric +
    ``._-``, max 100 chars, cannot start with a dot).

    Args:
        repo: The repository identifier to validate.
        purpose: Caller label for error messages.

    Returns:
        The validated repo string unchanged.

    Raises:
        HttpSafetyError: if the string doesn't match the expected
            shape.
    """
    if not repo or not isinstance(repo, str):
        raise HttpSafetyError(
            f"{purpose}: GitHub repo must be a non-empty string"
        )
    if not _GITHUB_REPO_RE.match(repo):
        raise HttpSafetyError(
            f"{purpose}: GitHub repo {repo!r} is not in the expected "
            f"shape 'owner/repo'.\n\n"
            f"  - owner: 1-39 chars, alphanumeric + dash, cannot "
            f"start/end with dash\n"
            f"  - repo:  1-100 chars, alphanumeric + . _ -, cannot "
            f"start with .\n\n"
            f"Example: 'anthropics/claude-code'"
        )
    return repo


def safe_github_pr(pr: str | int, *, purpose: str) -> str:
    """Validate a GitHub PR / issue number.

    Accepts strings or ints representing positive decimal
    integers up to 10 digits (GitHub numbers never get close to
    this but it keeps the bound sane).

    Args:
        pr: PR / issue number as string or int.
        purpose: Caller label for error messages.

    Returns:
        The validated number as a string.

    Raises:
        HttpSafetyError: if the value is not a positive integer.
    """
    if pr is None or pr == "":
        raise HttpSafetyError(
            f"{purpose}: GitHub PR / issue number must not be empty"
        )
    s = str(pr).strip()
    if not _GITHUB_PR_RE.match(s):
        raise HttpSafetyError(
            f"{purpose}: GitHub PR / issue number {pr!r} is not a "
            f"positive integer."
        )
    return s
