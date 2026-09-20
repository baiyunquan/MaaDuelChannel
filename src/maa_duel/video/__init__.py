from pathlib import Path

from maa_duel.config import PipelineConfig
from maa_duel.video.inventory import scan_inventory


def scan_videos(input_dir: Path, workspace: Path):
    config = PipelineConfig(input_dir=input_dir, workspace_dir=workspace)
    config.ensure_workspace()
    return scan_inventory(input_dir, config.manifest_dir / "videos.jsonl")


__all__ = ["scan_videos"]
