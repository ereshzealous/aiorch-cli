from pathlib import Path

from click.testing import CliRunner

from aiorch.cli import main


def test_cli_help_exits_clean():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_validate_accepts_well_formed_pipeline(tmp_path: Path):
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(
        "name: ok\n"
        "steps:\n"
        "  hello:\n"
        "    run: echo hi\n"
    )
    result = CliRunner().invoke(main, ["validate", str(pipeline)])
    assert result.exit_code == 0, result.output


def test_cli_validate_rejects_missing_file(tmp_path: Path):
    missing = tmp_path / "nope.yaml"
    result = CliRunner().invoke(main, ["validate", str(missing)])
    assert result.exit_code != 0
