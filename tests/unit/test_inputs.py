"""Regression tests for input-loader behaviour.

Two bugs these guard against:

1. ``type: file`` was being rejected as a retired type even though the
   parser accepts it and a loader is registered for it. Users hit a
   confusing "use type: artifact" error on a perfectly valid CLI input.

2. ``env:`` block values containing ``${VAR}`` were injected into
   ``os.environ`` *before* variable substitution ran, so downstream
   consumers read the literal ``"${VAR}"`` string.
"""

from __future__ import annotations

import os

import pytest

from aiorch.core.config import load_config, reset_config
from aiorch.inputs import load_input


def test_type_file_input_passes_through_path_as_string():
    # The CLI's file loader is a passthrough — it returns the value /
    # default as-is (the user is expected to read the file themselves
    # in a python: step). The important thing is that it does NOT
    # raise "type has been removed".
    result = load_input({"type": "file", "default": "inputs/sample.json"})
    assert result == "inputs/sample.json"


def test_type_file_input_returns_explicit_value_over_default():
    result = load_input({
        "type": "file",
        "default": "inputs/default.json",
        "value": "inputs/override.json",
    })
    assert result == "inputs/override.json"


def test_retired_types_still_raise():
    # json / csv / env are genuinely retired — make sure those still raise.
    for bad in ("json", "csv", "env"):
        with pytest.raises(ValueError, match="has been removed"):
            load_input({"type": bad})


def test_env_block_resolves_dollar_var_references(tmp_path, monkeypatch):
    # Shell-exported value should flow through an env: block that
    # references it via ${VAR}.
    monkeypatch.setenv("MY_WEBHOOK", "https://hooks.slack.com/services/T/B/X")
    monkeypatch.delenv("FORWARDED_WEBHOOK", raising=False)

    cfg_path = tmp_path / "aiorch.yaml"
    cfg_path.write_text(
        "env:\n"
        "  FORWARDED_WEBHOOK: ${MY_WEBHOOK}\n"
    )

    reset_config()
    load_config(cfg_path)
    assert os.environ["FORWARDED_WEBHOOK"] == "https://hooks.slack.com/services/T/B/X"

    reset_config()


def test_env_block_does_not_override_preexisting_shell_value(tmp_path, monkeypatch):
    # If the var is already in the environment, the env: block must not
    # clobber it — even if the config's resolved value differs.
    monkeypatch.setenv("API_KEY", "from-shell")
    monkeypatch.setenv("OTHER", "from-elsewhere")

    cfg_path = tmp_path / "aiorch.yaml"
    cfg_path.write_text(
        "env:\n"
        "  API_KEY: ${OTHER}\n"
    )

    reset_config()
    load_config(cfg_path)
    assert os.environ["API_KEY"] == "from-shell"

    reset_config()


def test_env_block_accepts_literal_values(tmp_path, monkeypatch):
    monkeypatch.delenv("LITERAL_FLAG", raising=False)

    cfg_path = tmp_path / "aiorch.yaml"
    cfg_path.write_text(
        "env:\n"
        "  LITERAL_FLAG: \"enabled\"\n"
    )

    reset_config()
    load_config(cfg_path)
    assert os.environ["LITERAL_FLAG"] == "enabled"

    reset_config()
