# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Local filesystem + SQLite artifact store for CLI mode.

Bytes live under ``~/.aiorch/artifacts/<sha[:2]>/<sha>`` (content-addressed).
Metadata rides on the same SQLite database that backs ``SQLiteStore`` —
the ``artifacts`` and ``run_artifacts`` tables are created by
``storage.sqlite._SCHEMA`` so everything is a single-file install.

Dedup: inputs with identical sha256 return the same artifact ID.
Outputs always create a new row.

The presigned URL is a plain ``file://`` URL pointing at the content
path — there's no network, no expiry, no signing. CLI users can open
or cat the artifact directly.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from aiorch.artifacts import (
    Artifact,
    ArtifactNotFound,
    ArtifactRole,
    ArtifactStore,
    compute_sha256,
    guess_extension,
)

logger = logging.getLogger("aiorch.artifacts.local")

DEFAULT_ARTIFACT_DIR = Path.home() / ".aiorch" / "artifacts"


class LocalArtifactStore(ArtifactStore):
    """Single-process, single-user artifact store for CLI mode.

    Shares the CLI history database with ``SQLiteStore``. The two stores
    can coexist on the same ``.db`` file because their tables are
    disjoint and the schema is created idempotently.
    """

    def __init__(
        self,
        artifact_dir: str | Path | None = None,
        db_path: str | Path | None = None,
    ):
        self._artifact_dir = Path(artifact_dir) if artifact_dir else DEFAULT_ARTIFACT_DIR
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

        from aiorch.storage.sqlite import DEFAULT_DB_PATH

        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open a dedicated connection. SQLite allows multiple connections
        # to the same file, and SQLiteStore already creates the schema —
        # here we just connect and rely on its existing schema pass.
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Ensure the schema is applied even if SQLiteStore hasn't been
        # instantiated yet (tests / cli doctor / etc.).
        from aiorch.storage.sqlite import _SCHEMA

        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        logger.info(
            "LocalArtifactStore: bytes=%s, db=%s",
            self._artifact_dir,
            self._db_path,
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _storage_path(self, sha256: str, artifact_id: str, extension: str) -> Path:
        """Content-addressed path: ~/.aiorch/artifacts/<sha[:2]>/<artifact_id>.<ext>.

        We use a two-char shard prefix to avoid one massive directory.
        The leaf is the artifact UUID (not the sha) because outputs with
        identical content must get distinct paths — one row per artifact
        ID, one file per row.
        """
        shard = sha256[:2]
        leaf = f"{artifact_id}{extension}"
        return self._artifact_dir / shard / leaf

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        return Artifact(
            id=row["id"],
            name=row["name"],
            storage_key=row["storage_path"],
            content_type=row["content_type"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            role=row["role"],
            pipeline_name=row["pipeline_name"],
            run_id=row["run_id"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # put
    # ------------------------------------------------------------------

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
        sha = compute_sha256(content)
        size = len(content)

        # Dedup lookup for inputs only
        if role == "input":
            existing = self._conn.execute(
                "SELECT * FROM artifacts WHERE sha256 = ? AND role = 'input' "
                "AND deleted_at IS NULL LIMIT 1",
                (sha,),
            ).fetchone()
            if existing:
                logger.debug(
                    "LocalArtifactStore dedup hit: %s (%d bytes)", sha[:12], size
                )
                return self._row_to_artifact(existing)

        # New artifact — create row + write file
        artifact_id = str(uuid.uuid4())
        extension = guess_extension(name, content_type)
        storage_path = self._storage_path(sha, artifact_id, extension)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_path.write_bytes(content)

        metadata_json = json.dumps(metadata) if metadata else None
        now = time.time()

        self._conn.execute(
            "INSERT INTO artifacts (id, name, storage_path, content_type, size_bytes, "
            "sha256, role, pipeline_name, run_id, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                artifact_id,
                name,
                str(storage_path),
                content_type,
                size,
                sha,
                role,
                pipeline_name,
                run_id,
                metadata_json,
                now,
            ),
        )
        self._conn.commit()

        return Artifact(
            id=artifact_id,
            name=name,
            storage_key=str(storage_path),
            content_type=content_type,
            size_bytes=size,
            sha256=sha,
            role=role,
            pipeline_name=pipeline_name,
            run_id=run_id,
            metadata=metadata or {},
            created_at=now,
        )

    # ------------------------------------------------------------------
    # get / get_metadata
    # ------------------------------------------------------------------

    def get(self, artifact_id: str) -> tuple[Artifact, bytes]:
        meta = self.get_metadata(artifact_id)
        path = Path(meta.storage_key)
        if not path.exists():
            raise ArtifactNotFound(
                f"Artifact {artifact_id} row exists but content file is missing: {path}"
            )
        return meta, path.read_bytes()

    def get_metadata(self, artifact_id: str) -> Artifact:
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE id = ? AND deleted_at IS NULL",
            (artifact_id,),
        ).fetchone()
        if not row:
            raise ArtifactNotFound(f"Artifact not found: {artifact_id}")
        return self._row_to_artifact(row)

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    def list(
        self,
        *,
        workspace_id: str | None = None,  # unused in local mode
        role: ArtifactRole | None = None,
        content_type_prefix: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Artifact]:
        sql = "SELECT * FROM artifacts WHERE deleted_at IS NULL"
        params: list[Any] = []
        if role is not None:
            sql += " AND role = ?"
            params.append(role)
        if content_type_prefix:
            sql += " AND content_type LIKE ?"
            params.append(content_type_prefix + "%")
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [self._row_to_artifact(r) for r in rows]

    # ------------------------------------------------------------------
    # delete (soft)
    # ------------------------------------------------------------------

    def delete(self, artifact_id: str) -> None:
        # Soft delete — the content file stays on disk. A follow-up
        # cleanup tool can physically remove files whose row is
        # deleted AND not referenced by any run_artifacts row.
        self._conn.execute(
            "UPDATE artifacts SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
            (time.time(), artifact_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # presigned URL (file://)
    # ------------------------------------------------------------------

    def presigned_download_url(
        self,
        artifact_id: str,
        ttl_seconds: int = 600,     # unused for file://
    ) -> str:
        meta = self.get_metadata(artifact_id)
        return Path(meta.storage_key).as_uri()

    # ------------------------------------------------------------------
    # workspace usage
    # ------------------------------------------------------------------

    def workspace_usage_bytes(self, workspace_id: str) -> int:
        # CLI has no workspace concept — return the total for display
        row = self._conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM artifacts "
            "WHERE deleted_at IS NULL"
        ).fetchone()
        return int(row["total"]) if row else 0

    # ------------------------------------------------------------------
    # run binding
    # ------------------------------------------------------------------

    def record_run_binding(
        self,
        *,
        run_id: int,
        artifact_id: str,
        binding_name: str,
        role: ArtifactRole,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_artifacts "
            "(run_id, artifact_id, binding_name, role, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, artifact_id, binding_name, role, time.time()),
        )
        self._conn.commit()
