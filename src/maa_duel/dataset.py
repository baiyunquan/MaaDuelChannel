from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from maa_duel.review import ReviewStore
from maa_duel.schema import ReviewStatus, RoundSample, Winner
from maa_duel.store import read_jsonl, write_jsonl


class PredictorUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enemy_id: int = Field(ge=1)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class PredictorSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    sample_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arena: Literal["green_vine"] = "green_vine"
    left_units: list[PredictorUnit]
    right_units: list[PredictorUnit]
    winner: Winner


def _predictor_sample(sample: RoundSample) -> PredictorSample:
    validated = RoundSample.model_validate(sample.model_dump(mode="json"))
    if validated.review_status is not ReviewStatus.ACCEPTED or validated.winner is None:
        raise ValueError(f"sample is not accepted: {validated.sample_id}")
    return PredictorSample(
        sample_id=validated.sample_id,
        source_video_sha256=validated.source.video_sha256,
        left_units=[PredictorUnit(enemy_id=unit.enemy_id, x=unit.x, y=unit.y) for unit in validated.left.units],
        right_units=[PredictorUnit(enemy_id=unit.enemy_id, x=unit.x, y=unit.y) for unit in validated.right.units],
        winner=validated.winner,
    )


def build_predictor_dataset(workspace: Path) -> list[PredictorSample]:
    manifest_dir = workspace / "manifests"
    automatic = read_jsonl(manifest_dir / "rounds.auto.jsonl", RoundSample)
    ids = [sample.sample_id for sample in automatic]
    if len(ids) != len(set(ids)):
        raise ValueError("automatic manifest contains duplicate sample ids")

    effective = ReviewStore(workspace / "review" / "corrections.jsonl").overlay(automatic)
    accepted = [sample for sample in effective if sample.review_status is ReviewStatus.ACCEPTED]
    rows = [_predictor_sample(sample) for sample in accepted]
    rows.sort(key=lambda item: item.sample_id)
    output_path = manifest_dir / "predictor.jsonl"
    write_jsonl(output_path, rows)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "accepted_samples": len(accepted),
        "written_samples": len(rows),
        "dataset_sha256": digest,
        "winner_counts": {
            "left": sum(row.winner is Winner.LEFT for row in rows),
            "right": sum(row.winner is Winner.RIGHT for row in rows),
        },
    }
    if metadata["accepted_samples"] != metadata["written_samples"]:
        raise RuntimeError("accepted sample count differs from written training sample count")
    (manifest_dir / "predictor.meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows
