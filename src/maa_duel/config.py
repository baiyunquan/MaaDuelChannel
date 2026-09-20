from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class PipelineConfig(BaseModel):
    """Filesystem and sampling configuration shared by every pipeline stage."""

    input_dir: Path
    workspace_dir: Path
    arena: str = "green_vine"
    sample_fps: float = Field(default=10.0, gt=0.0, le=60.0)
    stable_winner_frames: int = Field(default=5, ge=2)
    pipeline_version: str = "0.1.0"

    @model_validator(mode="after")
    def validate_workspace_is_external(self) -> PipelineConfig:
        input_path = self.input_dir.resolve(strict=False)
        workspace_path = self.workspace_dir.resolve(strict=False)
        if workspace_path == input_path or input_path in workspace_path.parents:
            raise ValueError("workspace_dir must be outside input_dir so source videos remain read-only")
        return self

    @property
    def manifest_dir(self) -> Path:
        return self.workspace_dir / "manifests"

    @property
    def frame_dir(self) -> Path:
        return self.workspace_dir / "frames"

    @property
    def review_dir(self) -> Path:
        return self.workspace_dir / "review"

    @property
    def model_dir(self) -> Path:
        return self.workspace_dir / "models"

    @property
    def report_dir(self) -> Path:
        return self.workspace_dir / "reports"

    @property
    def asset_dir(self) -> Path:
        return self.workspace_dir / "assets"

    def ensure_workspace(self) -> None:
        for path in (
            self.manifest_dir,
            self.frame_dir,
            self.review_dir,
            self.model_dir,
            self.report_dir,
            self.asset_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
