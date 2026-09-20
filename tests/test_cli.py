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


def test_training_commands_expose_reproducibility_options():
    runner = CliRunner()
    predictor = runner.invoke(app, ["train-predictor", "--help"])
    vision = runner.invoke(app, ["train-vision", "--help"])
    synthetic = runner.invoke(app, ["synth", "--help"])

    assert predictor.exit_code == 0
    assert "--epochs" in predictor.stdout
    assert "--seed" in predictor.stdout
    assert vision.exit_code == 0
    assert "--device" in vision.stdout
    assert synthetic.exit_code == 0
    assert "--detection-images" in synthetic.stdout
