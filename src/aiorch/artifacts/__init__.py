# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed artifact store — pluggable backend for pipeline file I/O.

The ArtifactStore provides a single place for pipelines to put and get
file content across the CLI/Platform split:

  - CLI mode → LocalArtifactStore (SQLite + ~/.aiorch/artifacts/)
  - Platform  → MinIOArtifactStore (Postgres + MinIO bucket)

Both backends speak the same protocol. Pipelines write against the
abstraction, the deployment decides where bytes live.

Content is addressed by SHA-256:
  - Inputs are deduped per-workspace: uploading the same bytes twice
    returns the same artifact row, saving storage and enabling
    lineage queries ("which runs used this exact file?").
  - Outputs are NOT deduped: every write-file call produces a distinct
    run-scoped artifact even if the bytes happen to match, so that
    deleting one run does not orphan another run's output.

MinIO folder layout (asymmetric — inputs flat, outputs hierarchical):

    <bucket>/
      <org_id>/
        <workspace_id>/
          inputs/
            <artifact_uuid>.<ext>                               # deduped
          outputs/
            <pipeline_name>/
              <YYYY-MM-DD>/
                run-<run_id>/
                  <artifact_uuid>.<ext>                         # per run

This module exposes only the abstract types and the factory. The two
concrete implementations live in ``local.py`` and ``minio.py``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger("aiorch.artifacts")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

ArtifactRole = Literal["input", "output"]


@dataclass
class Artifact:
    """Metadata for a stored artifact.

    The ``storage_key`` is the backend-specific address where the bytes
    live (a MinIO key, a filesystem path, etc.). Callers never need to
    construct this themselves — ``ArtifactStore.put()`` generates it.

    ``sha256`` is the hex digest of the content. Dedup for inputs uses
    ``(workspace_id, sha256)`` as the uniqueness key.

    ``created_by`` and ``workspace_id`` may be None in CLI mode where
    there is no user or workspace concept.
    """

    id: str
    name: str
    storage_key: str
    content_type: str
    size_bytes: int
    sha256: str
    created_at: float
    workspace_id: str | None = None
    org_id: str | None = None
    created_by: str | None = None
    role: ArtifactRole = "input"
    metadata: dict[str, Any] = field(default_factory=dict)
    # Output-only attribution (for outputs/<pipeline>/<date>/run-<id>/ layout)
    pipeline_name: str | None = None
    run_id: int | None = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ArtifactError(Exception):
    """Base class for artifact store failures."""


class ArtifactNotFound(ArtifactError):
    """Raised when an artifact ID is missing."""


class ArtifactTooLarge(ArtifactError):
    """Raised when an upload exceeds the per-artifact size limit."""

    def __init__(self, size_bytes: int, limit_bytes: int):
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"Artifact size {size_bytes} bytes exceeds limit {limit_bytes} bytes "
            f"({size_bytes / 1024 / 1024:.1f} MB > {limit_bytes / 1024 / 1024:.0f} MB)"
        )


class WorkspaceQuotaExceeded(ArtifactError):
    """Raised when an upload would push a workspace over its storage quota."""

    def __init__(self, workspace_id: str, current_bytes: int, quota_bytes: int, incoming_bytes: int):
        self.workspace_id = workspace_id
        self.current_bytes = current_bytes
        self.quota_bytes = quota_bytes
        self.incoming_bytes = incoming_bytes
        super().__init__(
            f"Workspace {workspace_id[:8]} storage quota exceeded: "
            f"{current_bytes / 1024**3:.2f} GB used + "
            f"{incoming_bytes / 1024**2:.1f} MB incoming > "
            f"{quota_bytes / 1024**3:.1f} GB quota"
        )


# ---------------------------------------------------------------------------
# Abstract store protocol
# ---------------------------------------------------------------------------


class ArtifactStore(ABC):
    """Abstract content store. Subclasses implement local or MinIO backends.

    Implementations must honor the asymmetric dedup contract:

      - ``put(..., role="input")``  MUST dedup by ``(workspace_id, sha256)``
                                    and return the existing artifact if the
                                    hash already exists. The caller sees a
                                    single authoritative row per unique file.

      - ``put(..., role="output")`` MUST create a new row every call, even
                                    if the content is byte-identical to an
                                    existing artifact. Outputs are
                                    run-scoped and must not share
                                    identities across runs.
    """

    @abstractmethod
    def put(
        self,
        *,
        name: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        role: ArtifactRole = "input",
        workspace_id: str | None = None,
        org_id: str | None = None,
        created_by: str | None = None,
        pipeline_name: str | None = None,
        run_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        """Store content and return the resulting Artifact metadata.

        Args:
            name: Display name (e.g., "report.md"). Not unique.
            content: Raw bytes to store.
            content_type: MIME type. Used for display + validation.
            role: "input" for user uploads (deduped) or "output" for
                step-produced files (not deduped).
            workspace_id: Required in Platform mode, ignored in CLI.
            org_id: Required in Platform mode (MinIO layout uses it).
            created_by: User UUID that initiated the upload, if any.
            pipeline_name: For outputs, pipeline name for folder layout.
            run_id: For outputs, run id for folder layout and lineage.
            metadata: Arbitrary JSON-serializable annotations.

        Raises:
            ArtifactTooLarge: if ``len(content)`` exceeds the per-artifact limit.
            WorkspaceQuotaExceeded: if the upload would push the workspace over quota.
        """

    @abstractmethod
    def get(self, artifact_id: str) -> tuple[Artifact, bytes]:
        """Fetch an artifact's metadata and content by ID.

        Raises:
            ArtifactNotFound: if the artifact ID doesn't exist or has been deleted.
        """

    @abstractmethod
    def get_metadata(self, artifact_id: str) -> Artifact:
        """Fetch only the metadata. Cheaper than get() when bytes aren't needed."""

    @abstractmethod
    def list(
        self,
        *,
        workspace_id: str | None = None,
        role: ArtifactRole | None = None,
        content_type_prefix: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Artifact]:
        """List artifacts with optional filters, newest first."""

    @abstractmethod
    def delete(self, artifact_id: str) -> None:
        """Soft-delete an artifact. The backend may retain bytes for a grace period."""

    @abstractmethod
    def presigned_download_url(
        self,
        artifact_id: str,
        ttl_seconds: int = 600,
    ) -> str:
        """Return a URL the caller can use to download the artifact directly.

        For MinIO, this is a presigned S3 URL with expiry. For the local
        CLI store, this returns a ``file://`` URL or similar — downloads
        are always local in CLI mode.

        Raises:
            ArtifactNotFound: if the artifact ID doesn't exist.
        """

    # ------------------------------------------------------------------
    # Hooks that subclasses may override — default implementations below
    # ------------------------------------------------------------------

    def workspace_usage_bytes(self, workspace_id: str) -> int:
        """Total bytes stored for a workspace. Used for quota enforcement.

        Default returns 0 (unlimited); MinIO backend overrides with a real
        SUM(size_bytes) query.
        """
        return 0

    def record_run_binding(
        self,
        *,
        run_id: int,
        artifact_id: str,
        binding_name: str,
        role: ArtifactRole,
    ) -> None:
        """Record that a run consumed or produced this artifact.

        This is what the ``run_artifacts`` junction table stores in
        Platform mode. CLI stores it in the equivalent SQLite table.
        No-op default is provided so test doubles don't have to
        implement it for pure put/get tests.
        """
        pass

    def close(self) -> None:
        """Release any backend resources. Optional."""
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_sha256(content: bytes) -> str:
    """Hex-encoded SHA-256 of the given bytes."""
    return hashlib.sha256(content).hexdigest()


def guess_extension(name: str, content_type: str) -> str:
    """Best-effort extension for a storage-key leaf.

    Prefers the user-supplied filename extension (preserves readability
    in the MinIO console), falls back to a mime-type guess, then to
    ``.bin`` as the last resort.
    """
    # Prefer the filename extension
    if "." in name and not name.startswith("."):
        return "." + name.rsplit(".", 1)[-1].lower()

    # Fallback: mime → extension
    import mimetypes

    guessed = mimetypes.guess_extension(content_type)
    if guessed:
        return guessed

    return ".bin"


# ---------------------------------------------------------------------------
# Singleton + factory
# ---------------------------------------------------------------------------


_store: ArtifactStore | None = None


def get_artifact_store() -> ArtifactStore:
    """Get the current artifact store instance.

    Raises:
        RuntimeError: if ``init_artifact_store()`` has not been called.
    """
    if _store is None:
        raise RuntimeError(
            "Artifact store not initialized. Call init_artifact_store() first."
        )
    return _store


def set_artifact_store(store: ArtifactStore) -> None:
    """Replace the artifact store singleton (used by tests)."""
    global _store
    _store = store


def init_artifact_store() -> None:
    """Initialize the artifact store backend.

    Resolution:
      1. ``AIORCH_MINIO_ENDPOINT`` set → ``MinIOArtifactStore`` (Platform)
      2. Otherwise                    → ``LocalArtifactStore`` (CLI)

    Both backends require the storage layer to be initialized first
    (``aiorch.storage.init_storage()``), because metadata lives in the
    same database (Postgres for Platform, SQLite for CLI).
    """
    global _store
    if _store is not None:
        return

    minio_endpoint = os.environ.get("AIORCH_MINIO_ENDPOINT")
    if minio_endpoint:
        from aiorch.artifacts.minio import MinIOArtifactStore

        _store = MinIOArtifactStore(
            endpoint=minio_endpoint,
            access_key=os.environ.get("AIORCH_MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.environ.get("AIORCH_MINIO_SECRET_KEY", "minioadmin"),
            bucket=os.environ.get("AIORCH_MINIO_BUCKET", "aiorch-artifacts"),
            region=os.environ.get("AIORCH_MINIO_REGION", "us-east-1"),
            use_ssl=os.environ.get("AIORCH_MINIO_USE_SSL", "").lower() in ("true", "1", "yes"),
            # Default: off. Real AWS S3 accepts the SSE-S3 header
            # transparently (S3-managed keys, no setup), but MinIO
            # community edition rejects it with NotImplemented unless
            # a KMS is configured via MINIO_KMS_* env vars on the
            # MinIO container. Shipping the default as "off" keeps
            # the out-of-box docker-compose stack working; real AWS
            # deployments can opt back in with AIORCH_MINIO_SSE_S3=true.
            sse_s3=os.environ.get("AIORCH_MINIO_SSE_S3", "false").lower() in ("true", "1", "yes"),
            max_size_mb=int(os.environ.get("AIORCH_ARTIFACT_MAX_SIZE_MB", "100")),
            workspace_quota_gb=int(os.environ.get("AIORCH_WORKSPACE_STORAGE_QUOTA_GB", "2")),
        )
        logger.info(
            "Artifact store: MinIO (%s, bucket=%s)",
            minio_endpoint,
            os.environ.get("AIORCH_MINIO_BUCKET", "aiorch-artifacts"),
        )
    else:
        from aiorch.artifacts.local import LocalArtifactStore

        _store = LocalArtifactStore()
        logger.info("Artifact store: Local (~/.aiorch/artifacts/)")


__all__ = [
    "Artifact",
    "ArtifactRole",
    "ArtifactStore",
    "ArtifactError",
    "ArtifactNotFound",
    "ArtifactTooLarge",
    "WorkspaceQuotaExceeded",
    "compute_sha256",
    "guess_extension",
    "get_artifact_store",
    "set_artifact_store",
    "init_artifact_store",
]
