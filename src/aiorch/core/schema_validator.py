# Copyright 2026 Eresh Gorantla
#
# Licensed under the Apache License, Version 2.0

"""JSON Schema validation for aiorch pipeline YAML.

Layer 1 validation: structural correctness (field names, types, enums).
Layer 2 (Pydantic + DAG) handles semantic correctness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "schemas" / "pipeline.v1.schema.json"
_schema_cache: dict | None = None


def _load_schema() -> dict:
    """Load and cache the pipeline JSON Schema."""
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    if not _SCHEMA_PATH.exists():
        logger.warning("Pipeline schema not found at %s — skipping schema validation", _SCHEMA_PATH)
        return {}

    with open(_SCHEMA_PATH) as f:
        _schema_cache = json.load(f)
    return _schema_cache


def get_schema() -> dict:
    """Return the pipeline JSON Schema dict (for API endpoint)."""
    return _load_schema()


def validate_pipeline_schema(data: dict[str, Any]) -> list[str]:
    """Validate a pipeline dict against the JSON Schema.

    Returns a list of human-readable error messages.
    Returns empty list if valid or if jsonschema is not installed.
    """
    schema = _load_schema()
    if not schema:
        return []

    try:
        import jsonschema
    except ImportError:
        logger.debug("jsonschema not installed — skipping JSON Schema validation")
        return []

    validator = jsonschema.Draft202012Validator(schema)
    errors: list[str] = []

    for error in sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path)):
        path = ".".join(str(p) for p in error.absolute_path)
        if path:
            errors.append(f"{path}: {error.message}")
        else:
            errors.append(error.message)

    return errors
