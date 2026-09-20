from typer.testing import CliRunner

from maa_duel.cli import app


def test_cli_exposes_pipeline_commands():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "assets",
        "scan",
        "synth",
        "train-vision",
        "extract",
        "review",
        "build-dataset",
        "train-predictor",
        "report",
    ):
        assert command in result.stdout
