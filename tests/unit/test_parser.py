from pathlib import Path

import pytest

from aiorch.core.parser import Agentfile, parse_file, parse_string


def test_parse_string_happy_path_two_step_chain():
    yaml_src = """
    name: chain
    steps:
      first:
        run: echo hello
        output: msg
      second:
        run: echo "{{msg}}"
        depends: [first]
    """
    af = parse_string(yaml_src)
    assert isinstance(af, Agentfile)
    assert af.name == "chain"
    assert set(af.steps.keys()) == {"first", "second"}
    assert af.steps["second"].depends == ["first"]


def test_parse_string_rejects_step_with_no_primitive():
    yaml_src = """
    name: broken
    steps:
      orphan:
        depends: []
    """
    with pytest.raises(ValueError):
        parse_string(yaml_src)


def test_parse_string_rejects_retired_input_type_env():
    # `type: env` was removed in favor of workspace secrets / configs.
    # The InputField validator raises at parse time so stale YAML
    # surfaces a migration error instead of a generic "unknown type".
    yaml_src = """
    name: legacy
    input:
      secret:
        type: env
    steps:
      show:
        run: echo hi
    """
    with pytest.raises(ValueError, match="env"):
        parse_string(yaml_src)


def test_parse_file_raises_on_missing_path(tmp_path: Path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        parse_file(missing)


def test_parse_file_happy_path(tmp_path: Path):
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(
        "name: one-step\n"
        "steps:\n"
        "  hello:\n"
        "    run: echo hi\n"
    )
    af = parse_file(pipeline)
    assert af.name == "one-step"
    assert list(af.steps) == ["hello"]
