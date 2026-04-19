# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Email connector handler — SMTP via aiosmtplib.

Implements the canonical ``send`` operation for email.

Per Q2 of the connectors design doc, every send opens a fresh SMTP
connection, sends the message, and closes. Credential rotation is
automatic. No connection pool.
"""

from __future__ import annotations

import logging
from typing import Any

from aiorch.connectors import ConnectorAuthError
from aiorch.connectors.handlers import ConnectorHandler, register_handler

logger = logging.getLogger("aiorch.connectors.handlers.email")


class EmailHandler(ConnectorHandler):
    """SMTP email handler via aiosmtplib."""

    type = "email"
    subtype = "smtp"
    supported_operations = {"send"}

    @classmethod
    def get_client_info(cls) -> dict | None:
        try:
            import aiosmtplib  # type: ignore
            version = getattr(aiosmtplib, "__version__", "unknown")
        except ImportError:
            return None
        return {
            "client_library": "aiosmtplib",
            "client_version": version,
            "docs_url": "https://aiosmtplib.readthedocs.io/en/stable/",
        }

    async def execute(
        self,
        *,
        operation: str,
        params: dict,
        connector_config: dict,
        context: dict,
    ) -> Any:
        self._check_supported(operation)
        return await self._send(params, connector_config)

    async def _send(self, params: dict, connector_config: dict) -> dict:
        to = params.get("to") or []
        subject = params.get("subject", "")
        body = params.get("body") or params.get("message", "")
        cc = params.get("cc") or []
        bcc = params.get("bcc") or []
        attachments = params.get("attachments") or []
        html = params.get("html", False)

        # Normalize to lists
        if isinstance(to, str):
            to = [to]
        if isinstance(cc, str):
            cc = [cc]
        if isinstance(bcc, str):
            bcc = [bcc]

        if not to:
            raise ValueError("email send requires 'to' (recipient list)")
        if not subject:
            raise ValueError("email send requires 'subject'")

        host = connector_config.get("host")
        if not host:
            raise ValueError("Email connector requires 'host' in config")
        port = int(connector_config.get("port", 587))
        username = connector_config.get("username")
        password = connector_config.get("password")
        # TLS resolution — new `tls_mode` enum is authoritative; legacy
        # `use_tls` / `use_starttls` bools kept for backward-compat with
        # connectors created before tls_mode existed.
        tls_mode = connector_config.get("tls_mode")
        if tls_mode is None:
            legacy_use_tls = bool(connector_config.get("use_tls", True))
            legacy_use_starttls = bool(connector_config.get("use_starttls", False))
            if legacy_use_starttls:
                tls_mode = "starttls"
            elif legacy_use_tls:
                # Prior default behavior: "Use STARTTLS" checkbox → use_tls=True,
                # which aiosmtplib interprets as implicit TLS. Preserve exactly.
                tls_mode = "tls"
            else:
                tls_mode = "none"

        tls_mode = str(tls_mode).lower()
        if tls_mode == "starttls":
            use_tls = False
            use_starttls = True
        elif tls_mode == "tls":
            use_tls = True
            use_starttls = False
        elif tls_mode == "none":
            use_tls = False
            use_starttls = False
        else:
            raise ValueError(
                f"email tls_mode must be 'starttls', 'tls', or 'none' — got {tls_mode!r}"
            )
        from_address = connector_config.get("from_address") or username
        timeout_seconds = int(connector_config.get("timeout_seconds", 30))

        if not from_address:
            raise ValueError(
                "Email connector requires 'from_address' in config "
                "(or 'username' as fallback)"
            )

        # Build the message
        from email.message import EmailMessage

        message = EmailMessage()
        message["From"] = from_address
        message["To"] = ", ".join(to)
        if cc:
            message["Cc"] = ", ".join(cc)
        message["Subject"] = subject

        if html:
            message.set_content("Plain-text version unavailable.")
            message.add_alternative(body, subtype="html")
        else:
            message.set_content(body)

        # Add attachments if present. Each attachment is a dict:
        #   {filename, content (bytes), mime_type: "text/plain"}
        for att in attachments:
            if not isinstance(att, dict):
                continue
            filename = att.get("filename", "attachment.bin")
            content = att.get("content", b"")
            mime_type = att.get("mime_type", "application/octet-stream")
            maintype, _, subtype = mime_type.partition("/")

            if isinstance(content, str):
                content = content.encode("utf-8")

            message.add_attachment(
                content,
                maintype=maintype or "application",
                subtype=subtype or "octet-stream",
                filename=filename,
            )

        # Send via aiosmtplib
        import aiosmtplib

        try:
            await aiosmtplib.send(
                message,
                hostname=host,
                port=port,
                username=username or None,
                password=password or None,
                use_tls=use_tls and not use_starttls,
                start_tls=use_starttls,
                timeout=timeout_seconds,
                recipients=list(to) + list(cc) + list(bcc),
            )
        except aiosmtplib.SMTPAuthenticationError as exc:
            raise ConnectorAuthError(
                f"SMTP authentication failed for {from_address}: {exc}"
            ) from exc
        except aiosmtplib.SMTPConnectError as exc:
            raise ConnectionError(
                f"Cannot reach SMTP server {host}:{port}: {exc}"
            ) from exc

        return {
            "message_id": message.get("Message-ID", ""),
            "recipients": len(to) + len(cc) + len(bcc),
            "from": from_address,
        }


# Register on import
register_handler(EmailHandler())
