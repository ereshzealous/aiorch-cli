# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Object store connector handler — S3, MinIO, R2, GCS via boto3.

Implements canonical object-store operations (read / write / list /
delete) against any S3-compatible endpoint. Same boto3 client
configuration pattern as ``MinIOArtifactStore`` from the
feature/input-artifacts branch, but scoped to user-defined external
buckets rather than the platform-managed artifact bucket.

**Distinction from the artifact store:** the artifact store
(``type: artifact`` inputs, ``write-file`` action outputs) handles
files the pipeline *owns* — system-managed, lineage-tracked,
sha256-deduped, workspace-quota-enforced, single platform bucket.
Object-store connectors handle arbitrary external buckets — user
data lakes, partner exports, CI artifacts, cross-region replicas.
Both backed by boto3 internally, different user stories, deliberately
separate surfaces.

Per Q2 of the connectors design doc, every operation opens a fresh
S3 client, executes, and returns. boto3 clients are cheap to
construct — no TCP pool shared across operations. This keeps
rotation automatic and avoids any state held between invocations.
"""

from __future__ import annotations

import logging
from typing import Any

from aiorch.connectors import ConnectorAuthError, NotSupportedError
from aiorch.connectors.handlers import ConnectorHandler, register_handler

logger = logging.getLogger("aiorch.connectors.handlers.object_store")


# ---------------------------------------------------------------------------
# Key validation (defense in depth over key_prefix)
# ---------------------------------------------------------------------------


def _validate_key(key: str, key_prefix: str | None) -> str:
    """Validate and prefix a key.

    Rejects anything that tries to escape the configured key_prefix
    via '..' traversal or absolute paths. Returns the fully-qualified
    S3 key (prefix + key) that the caller should use with boto3.

    The prefix is a containment boundary, not an S3 policy replacement
    — operators should ALSO configure bucket policies if they want
    hard isolation. This check is for defense in depth.
    """
    if not isinstance(key, str) or not key:
        raise ValueError("Object store key must be a non-empty string")

    # Reject absolute paths and traversal attempts
    if key.startswith("/"):
        raise ValueError(f"Object store key must be relative: {key!r}")
    if ".." in key.split("/"):
        raise ValueError(f"Object store key may not contain '..': {key!r}")

    if not key_prefix:
        return key

    # Normalize prefix (ensure trailing slash)
    prefix = key_prefix.rstrip("/") + "/"
    return prefix + key


def _strip_prefix(full_key: str, prefix: str | None) -> str:
    """Remove the configured prefix from a key returned by list()."""
    if not prefix:
        return full_key
    p = prefix.rstrip("/") + "/"
    if full_key.startswith(p):
        return full_key[len(p):]
    return full_key


# ---------------------------------------------------------------------------
# Client builder
# ---------------------------------------------------------------------------


def _build_client(connector_config: dict):
    """Construct a fresh boto3 S3 client from connector config.

    boto3 clients are lightweight — constructing one per operation
    is a few ms of overhead. No shared state, no connection cache.
    """
    import boto3
    from botocore.config import Config

    endpoint_url = connector_config.get("endpoint_url")
    region = connector_config.get("region", "us-east-1")
    access_key = (
        connector_config.get("access_key_id")
        or connector_config.get("access_key")
    )
    secret_key = (
        connector_config.get("secret_access_key")
        or connector_config.get("secret_key")
    )
    session_token = connector_config.get("session_token")
    use_ssl = bool(connector_config.get("use_ssl", True))
    addressing_style = connector_config.get("addressing_style", "path")

    kwargs: dict = {
        "region_name": region,
        "use_ssl": use_ssl,
        "config": Config(
            signature_version="s3v4",
            s3={"addressing_style": addressing_style},
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    }
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if access_key:
        kwargs["aws_access_key_id"] = access_key
    if secret_key:
        kwargs["aws_secret_access_key"] = secret_key
    if session_token:
        kwargs["aws_session_token"] = session_token

    return boto3.client("s3", **kwargs)


def _map_boto_error(exc: Exception) -> Exception:
    """Translate botocore exceptions into friendly messages."""
    try:
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        return exc

    if isinstance(exc, NoCredentialsError):
        return ConnectorAuthError(
            "S3 credentials missing. Check the connector's access_key_id and "
            "secret_access_key."
        )

    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("InvalidAccessKeyId", "SignatureDoesNotMatch", "AccessDenied"):
            return ConnectorAuthError(
                f"S3 access denied ({code}). Check credentials and bucket policy."
            )
        if code in ("NoSuchBucket",):
            return ValueError(f"S3 bucket does not exist: {exc}")
        if code in ("NoSuchKey", "404"):
            return FileNotFoundError(f"S3 key does not exist: {exc}")
        if code in ("EntityTooLarge",):
            return ValueError(f"S3 upload exceeds bucket's max size: {exc}")

    return exc


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class ObjectStoreHandler(ConnectorHandler):
    """Canonical S3-compatible handler for S3, MinIO, R2, GCS.

    All four subtypes share the same code path — the only differences
    are endpoint_url, addressing_style, and default region, which come
    from the connector config.
    """

    type = "object_store"
    subtype = "s3"  # shared implementation for s3/minio/r2/gcs
    supported_operations = {"read", "write", "list", "delete"}

    @classmethod
    def get_client_info(cls) -> dict | None:
        try:
            import boto3  # type: ignore
            version = getattr(boto3, "__version__", "unknown")
        except ImportError:
            return None
        return {
            "client_library": "boto3",
            "client_version": version,
            "docs_url": "https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html",
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

        bucket = connector_config.get("bucket")
        if not bucket:
            raise ValueError(
                "Object store connector requires 'bucket' in config"
            )
        key_prefix = connector_config.get("key_prefix")

        client = _build_client(connector_config)

        try:
            if operation == "read":
                return await _async(self._read, client, bucket, key_prefix, params)
            if operation == "write":
                return await _async(self._write, client, bucket, key_prefix, params)
            if operation == "list":
                return await _async(self._list, client, bucket, key_prefix, params)
            if operation == "delete":
                return await _async(self._delete, client, bucket, key_prefix, params)
            raise NotSupportedError(f"ObjectStoreHandler cannot {operation}")
        except Exception as exc:
            mapped = _map_boto_error(exc)
            if mapped is exc:
                raise
            raise mapped from exc

    # ------------------------------------------------------------------
    # read — fetch bytes
    # ------------------------------------------------------------------

    def _read(self, client, bucket: str, key_prefix: str | None, params: dict) -> dict:
        key = params.get("key")
        if not key:
            raise ValueError("read operation requires 'key'")

        full_key = _validate_key(key, key_prefix)
        response = client.get_object(Bucket=bucket, Key=full_key)
        body = response["Body"].read()
        return {
            "content": body,
            "content_type": response.get("ContentType", "application/octet-stream"),
            "size": len(body),
            "bytes_transferred": len(body),
            "last_modified": response.get("LastModified").isoformat()
                if response.get("LastModified") else None,
            "etag": response.get("ETag", "").strip('"'),
        }

    # ------------------------------------------------------------------
    # write — upload bytes or string
    # ------------------------------------------------------------------

    def _write(self, client, bucket: str, key_prefix: str | None, params: dict) -> dict:
        key = params.get("key")
        content = params.get("content") or params.get("body")
        content_type = params.get("content_type", "application/octet-stream")

        if not key:
            raise ValueError("write operation requires 'key'")
        if content is None:
            raise ValueError("write operation requires 'content' (bytes or string)")

        full_key = _validate_key(key, key_prefix)
        content_bytes = (
            content.encode("utf-8") if isinstance(content, str) else content
        )

        response = client.put_object(
            Bucket=bucket,
            Key=full_key,
            Body=content_bytes,
            ContentType=content_type,
        )
        return {
            "etag": response.get("ETag", "").strip('"'),
            "size": len(content_bytes),
            "bytes_transferred": len(content_bytes),
            "key": key,   # original (unprefixed) key for caller readability
        }

    # ------------------------------------------------------------------
    # list — enumerate keys under a prefix
    # ------------------------------------------------------------------

    def _list(self, client, bucket: str, key_prefix: str | None, params: dict) -> list[dict]:
        prefix = params.get("prefix", "")
        limit = int(params.get("limit", 100))

        full_prefix = _validate_key(prefix, key_prefix) if prefix else (key_prefix or "")
        if key_prefix and not prefix:
            full_prefix = key_prefix.rstrip("/") + "/"

        response = client.list_objects_v2(
            Bucket=bucket,
            Prefix=full_prefix,
            MaxKeys=min(limit, 1000),
        )

        contents = response.get("Contents", [])
        return [
            {
                "key": _strip_prefix(obj["Key"], key_prefix),
                "size": obj.get("Size", 0),
                "last_modified": obj.get("LastModified").isoformat()
                    if obj.get("LastModified") else None,
                "etag": obj.get("ETag", "").strip('"'),
            }
            for obj in contents[:limit]
        ]

    # ------------------------------------------------------------------
    # delete — remove a single key
    # ------------------------------------------------------------------

    def _delete(self, client, bucket: str, key_prefix: str | None, params: dict) -> dict:
        key = params.get("key")
        if not key:
            raise ValueError("delete operation requires 'key'")

        full_key = _validate_key(key, key_prefix)
        client.delete_object(Bucket=bucket, Key=full_key)
        return {"key": key, "deleted": True}


# ---------------------------------------------------------------------------
# asyncio bridge for sync boto3 calls
# ---------------------------------------------------------------------------


async def _async(fn, *args, **kwargs):
    """Run a synchronous boto3 call on the default thread pool.

    boto3 is a blocking library. Wrapping calls in run_in_executor
    keeps the event loop responsive for concurrent runs in the
    executor process.
    """
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


# Register the handler once per subtype. All four subtypes share one
# implementation but get distinct registry entries so `get_handler()`
# finds them by (type, subtype). We use a factory instead of a loop
# with a class closure to avoid the late-binding gotcha.


def _make_subtype_handler(subtype_name: str) -> ObjectStoreHandler:
    h = ObjectStoreHandler()
    h.subtype = subtype_name
    return h


for _subtype in ("s3", "minio", "r2", "gcs"):
    register_handler(_make_subtype_handler(_subtype))
