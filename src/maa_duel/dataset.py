from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from maa_duel.calibration import (
    FittedCalibration,
    PositionQuality,
    fit_homography,
    load_calibration,
    transform_point,
)
from maa_duel.combat import DERIVED_FORMULA_VERSION, load_combat_knowledge
from maa_duel.contracts import DatasetVersion, git_commit, sha256_file, write_contract
from maa_duel.extraction_state import active_extraction_paths
from maa_duel.review import ReviewStore
from maa_duel.schema import ReviewStatus, RoundSample, Winner
from maa_duel.store import read_jsonl, write_jsonl
from maa_duel.training.derived import formula_sha256
from maa_duel.training.features import feature_schema_sha256, vocabulary_sha256


class PredictorUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enemy_id: int = Field(ge=1)
    x: float
    y: float
    raw_x: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_y: float | None = Field(default=None, ge=0.0, le=1.0)
    instance_id: str | None = None
    position_quality: PositionQuality = PositionQuality.ESTIMATED
    sprite_offset_applied: bool = False


class BattleState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    sample_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_video_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    arena: Literal["green_vine"] = "green_vine"
    calibration_id: str | None = None
    calibration_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    left_units: list[PredictorUnit]
    right_units: list[PredictorUnit]

    @model_validator(mode="after")
    def validate_sides(self) -> BattleState:
        if not self.left_units or not self.right_units:
            raise ValueError("battle state requires units on both sides")
        return self


class PredictorSample(BattleState):
    winner: Winner


def _calibrated_unit(
    unit,
    *,
    fitted: FittedCalibration,
    side: str,
    index: int,
) -> PredictorUnit:
    calibration = fitted.calibration
    offset = next((value for value in calibration.sprite_offsets if value.enemy_id == unit.enemy_id), None)
    image_x = unit.x * calibration.image_width + (offset.image_dx if offset else 0.0)
    image_y = unit.y * calibration.image_height + (offset.image_dy if offset else 0.0)
    ground_x, ground_y = transform_point(fitted.matrix, image_x, image_y)
    return PredictorUnit(
        enemy_id=unit.enemy_id,
        x=ground_x,
        y=ground_y,
        raw_x=unit.x,
        raw_y=unit.y,
        instance_id=f"{side}-{index}",
        position_quality=PositionQuality.ESTIMATED,
        sprite_offset_applied=offset is not None,
    )


def _predictor_sample(
    sample: RoundSample,
    *,
    fitted: FittedCalibration,
    calibration_digest: str,
) -> PredictorSample:
    validated = RoundSample.model_validate(sample.model_dump(mode="json"))
    if validated.review_status is not ReviewStatus.ACCEPTED or validated.winner is None:
        raise ValueError(f"sample is not accepted: {validated.sample_id}")
    return PredictorSample(
        sample_id=validated.sample_id,
        source_video_sha256=validated.source.video_sha256,
        calibration_id=fitted.calibration.calibration_id,
        calibration_sha256=calibration_digest,
        left_units=[
            _calibrated_unit(unit, fitted=fitted, side="left", index=index)
            for index, unit in enumerate(validated.left.units)
        ],
        right_units=[
            _calibrated_unit(unit, fitted=fitted, side="right", index=index)
            for index, unit in enumerate(validated.right.units)
        ],
        winner=validated.winner,
    )


def _load_video_calibration(workspace: Path, sample_ids: list[str]) -> tuple[Path, FittedCalibration, str]:
    calibration_dir = workspace / "assets" / "calibration"
    candidates = sorted(calibration_dir.glob("*.json")) if calibration_dir.is_dir() else []
    video_calibrations = []
    for path in candidates:
        calibration = load_calibration(path)
        if calibration.arena == "green_vine" and calibration.source_kind == "video":
            video_calibrations.append((path, calibration))
    if len(video_calibrations) != 1:
        joined = ", ".join(sample_ids)
        raise ValueError(
            f"accepted samples [{joined}] require exactly one video calibration for green_vine; "
            f"found {len(video_calibrations)}"
        )
    path, calibration = video_calibrations[0]
    return path, fit_homography(calibration), sha256_file(path)


def build_predictor_dataset(workspace: Path) -> list[PredictorSample]:
    manifest_dir = workspace / "manifests"
    extraction = active_extraction_paths(workspace)
    source_manifest = extraction.rounds_manifest
    if not source_manifest.is_file():
        raise FileNotFoundError(f"round manifest is missing; run extract first: {source_manifest}")
    automatic = read_jsonl(source_manifest, RoundSample)
    ids = [sample.sample_id for sample in automatic]
    if len(ids) != len(set(ids)):
        raise ValueError("automatic manifest contains duplicate sample ids")

    effective = ReviewStore(extraction.corrections).overlay(automatic)
    accepted = [sample for sample in effective if sample.review_status is ReviewStatus.ACCEPTED]
    if accepted:
        calibration_path, fitted, calibration_digest = _load_video_calibration(
            workspace,
            [sample.sample_id for sample in accepted],
        )
        rows = [_predictor_sample(sample, fitted=fitted, calibration_digest=calibration_digest) for sample in accepted]
    else:
        calibration_path = None
        fitted = None
        calibration_digest = None
        rows = []
    rows.sort(key=lambda item: item.sample_id)
    output_path = manifest_dir / "predictor.jsonl"
    write_jsonl(output_path, rows)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    dataset_version = f"predictor-{digest[:12]}"
    metadata = {
        "schema_version": 2,
        "accepted_samples": len(accepted),
        "written_samples": len(rows),
        "dataset_sha256": digest,
        "dataset_version": dataset_version,
        "winner_counts": {
            "left": sum(row.winner is Winner.LEFT for row in rows),
            "right": sum(row.winner is Winner.RIGHT for row in rows),
        },
    }
    if calibration_path is not None and fitted is not None and calibration_digest is not None:
        metadata.update(
            {
                "calibration_id": fitted.calibration.calibration_id,
                "calibration_manifest": calibration_path.relative_to(workspace).as_posix(),
                "calibration_sha256": calibration_digest,
                "formula_version": DERIVED_FORMULA_VERSION,
                "formula_sha256": formula_sha256(),
            }
        )
    knowledge_path = workspace / "assets" / "combat" / "vs2_enemy_combat.json"
    knowledge = load_combat_knowledge(knowledge_path) if knowledge_path.is_file() else None
    if knowledge is not None:
        metadata.update(
            {
                "feature_version": knowledge.feature_version,
                "feature_schema_sha256": feature_schema_sha256(),
                "knowledge_manifest": knowledge_path.relative_to(workspace).as_posix(),
                "knowledge_sha256": sha256_file(knowledge_path),
                "vocabulary_sha256": vocabulary_sha256(knowledge),
            }
        )
    dependency_payload = json.dumps(
        {
            "dataset_sha256": digest,
            "feature_version": metadata.get("feature_version"),
            "feature_schema_sha256": metadata.get("feature_schema_sha256"),
            "formula_version": metadata.get("formula_version"),
            "formula_sha256": metadata.get("formula_sha256"),
            "knowledge_sha256": metadata.get("knowledge_sha256"),
            "calibration_sha256": metadata.get("calibration_sha256"),
            "vocabulary_sha256": metadata.get("vocabulary_sha256"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    dataset_version = f"predictor-{hashlib.sha256(dependency_payload).hexdigest()[:12]}"
    metadata["dataset_version"] = dataset_version
    if metadata["accepted_samples"] != metadata["written_samples"]:
        raise RuntimeError("accepted sample count differs from written training sample count")
    (manifest_dir / "predictor.meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    contract = DatasetVersion(
        dataset_version=dataset_version,
        annotation_manifest=output_path.relative_to(workspace).as_posix(),
        annotation_sha256=sha256_file(output_path),
        source_manifest_sha256=sha256_file(source_manifest),
        task_counts={"predictor": len(rows)},
        split_policy="all-training",
        feature_version=metadata.get("feature_version"),
        feature_schema_sha256=metadata.get("feature_schema_sha256"),
        knowledge_manifest=metadata.get("knowledge_manifest"),
        knowledge_sha256=metadata.get("knowledge_sha256"),
        formula_version=metadata.get("formula_version"),
        formula_sha256=metadata.get("formula_sha256"),
        calibration_id=metadata.get("calibration_id"),
        calibration_manifest=metadata.get("calibration_manifest"),
        calibration_sha256=metadata.get("calibration_sha256"),
        vocabulary_sha256=metadata.get("vocabulary_sha256"),
        git_commit=git_commit(),
    )
    write_contract(workspace / "datasets" / dataset_version / "dataset.json", contract)
    return rows
