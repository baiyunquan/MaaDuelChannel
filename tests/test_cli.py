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
        "predict-duel",
        "report",
    ):
        assert command in result.stdout


def test_training_commands_expose_reproducibility_options():
    runner = CliRunner()
    predictor = runner.invoke(app, ["train-predictor", "--help"])
    inference = runner.invoke(app, ["predict-duel", "--help"])
    vision = runner.invoke(app, ["train-vision", "--help"])
    synthetic = runner.invoke(app, ["synth", "--help"])
    assets = runner.invoke(app, ["assets", "--help"])

    assert predictor.exit_code == 0
    assert inference.exit_code == 0
    assert "--epochs" in predictor.stdout
    assert "--seed" in predictor.stdout
    assert vision.exit_code == 0
    assert "--device" in vision.stdout
    assert synthetic.exit_code == 0
    assert "--detection-images" in synthetic.stdout
    assert assets.exit_code == 0
    assert "fetch-prts-combat" in assets.stdout


def test_review_command_exposes_video_dir_and_web_options():
    runner = CliRunner()
    result = runner.invoke(app, ["review", "--help"])

    assert result.exit_code == 0
    assert "--video-dir" in result.stdout
    assert "--web" in result.stdout
    assert "--platform" in result.stdout
    assert "--skip-prep-roster" in result.stdout
    assert "--only-prep-roster" in result.stdout


def test_extract_command_exposes_coarse_scan_and_review_reset_options():
    result = CliRunner().invoke(app, ["extract", "--help"])

    assert result.exit_code == 0
    assert "coarse" in result.stdout.lower()
    assert "--reset-review" in result.stdout
    assert "--phase-debug" in result.stdout


def test_extract_rejects_review_reset_without_full_force(tmp_path):
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    runner = CliRunner()

    missing_force = runner.invoke(
        app,
        [
            "extract",
            "--input-dir",
            str(input_dir),
            "--workspace",
            str(tmp_path / "workspace"),
            "--reset-review",
        ],
    )
    partial = runner.invoke(
        app,
        [
            "extract",
            "--input-dir",
            str(input_dir),
            "--workspace",
            str(tmp_path / "workspace"),
            "--force",
            "--reset-review",
            "--max-videos",
            "1",
        ],
    )

    assert missing_force.exit_code != 0
    assert "--force" in missing_force.output
    assert partial.exit_code != 0
    assert "--max-videos" in partial.output
