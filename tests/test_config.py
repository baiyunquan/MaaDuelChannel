from maa_duel.config import PipelineConfig


def test_workspace_must_not_be_inside_input(tmp_path):
    input_dir = tmp_path / "videos"
    input_dir.mkdir()

    try:
        PipelineConfig(input_dir=input_dir, workspace_dir=input_dir / "workspace")
    except ValueError as exc:
        assert "workspace_dir" in str(exc)
    else:
        raise AssertionError("workspace inside input must be rejected")


def test_workspace_layout_is_created_outside_input(tmp_path):
    config = PipelineConfig(
        input_dir=tmp_path / "videos",
        workspace_dir=tmp_path / "workspace",
    )
    config.ensure_workspace()

    assert config.manifest_dir == tmp_path / "workspace" / "manifests"
    assert config.review_dir.is_dir()
    assert config.model_dir.is_dir()
