from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from maa_duel.schema import BoundingBox

CONTRACT_VERSION = "1.0.0"


class AnnotationTask(StrEnum):
    ROSTER_CLASSIFICATION = "roster_classification"
    BATTLEFIELD_DETECTION = "battlefield_detection"
    OCR_CLASSIFICATION = "ocr_classification"


class AnnotationState(StrEnum):
    PENDING = "pending"
    REVIEWED = "reviewed"


class DataBucket(StrEnum):
    HARD = "hard"
    REPLAY = "replay"
    BASE = "base"


class AnnotationBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_name: str = Field(min_length=1)
    bbox: BoundingBox
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    side: Literal["left", "right"] | None = None


class AnnotationRecord(BaseModel):
    """Stable interchange record between extraction, Platform, sampling, and training."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    contract_version: Literal["1.0.0"] = CONTRACT_VERSION
    annotation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    task: AnnotationTask
    sample_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_image: str = Field(min_length=1)
    exported_image: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    side: Literal["left", "right"] | None = None
    slot: int | None = Field(default=None, ge=0)
    predicted_class: str | None = None
    predicted_text: str | None = None
    predicted_boxes: list[AnnotationBox] = Field(default_factory=list)
    ground_truth_class: str | None = None
    ground_truth_text: str | None = None
    ground_truth_boxes: list[AnnotationBox] = Field(default_factory=list)
    state: AnnotationState = AnnotationState.PENDING
    corrected: bool = False
    bucket: DataBucket | None = None
    model_version: str | None = None
    failure_reasons: list[str] = Field(default_factory=list)


class SamplingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hard_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    replay_fraction: float = Field(default=0.3, ge=0.0, le=1.0)
    base_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    maximum_samples: int | None = Field(default=None, ge=1)
    seed: int = 20260920

    @model_validator(mode="after")
    def validate_fractions(self) -> SamplingPolicy:
        total = self.hard_fraction + self.replay_fraction + self.base_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError("hard, replay, and base fractions must sum to 1.0")
        return self


class DatasetVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    contract_version: Literal["1.0.0"] = CONTRACT_VERSION
    dataset_version: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parent_version: str | None = None
    annotation_manifest: str = Field(min_length=1)
    annotation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_counts: dict[str, int]
    bucket_counts: dict[str, int] = Field(default_factory=dict)
    class_names: dict[str, list[str]] = Field(default_factory=dict)
    sampling_policy: SamplingPolicy | None = None
    split_policy: Literal["annotation-only", "all-training"] = "annotation-only"
    feature_version: str | None = None
    feature_schema_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    formula_version: str | None = None
    formula_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    knowledge_manifest: str | None = None
    knowledge_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    calibration_id: str | None = None
    calibration_manifest: str | None = None
    calibration_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    vocabulary_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    git_commit: str


class ModelVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    contract_version: Literal["1.0.0"] = CONTRACT_VERSION
    model_version: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task: str = Field(min_length=1)
    dataset_version: str | None = None
    parent_model: str | None = None
    base_model: str = Field(min_length=1)
    checkpoint: str = Field(min_length=1)
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    device: str
    precision: Literal["fp32", "fp16", "bf16"]
    training_args: dict[str, object]
    metrics_scope: Literal["training-only"] = "training-only"
    feature_version: str | None = None
    feature_schema_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    formula_version: str | None = None
    formula_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    knowledge_manifest: str | None = None
    knowledge_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    calibration_id: str | None = None
    calibration_manifest: str | None = None
    calibration_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    vocabulary_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    git_commit: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_annotation_id(*parts: object) -> str:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def git_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def write_contract(path: Path, contract: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(contract.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
