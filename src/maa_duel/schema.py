from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Winner(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class AnnotationSource(StrEnum):
    AUTO = "auto"
    MANUAL = "manual"
    SYNTHETIC = "synthetic"


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_order(self) -> BoundingBox:
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("bounding box must have positive width and height")
        return self


class RosterEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enemy_id: int = Field(ge=1)
    count: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    slot: int | None = Field(default=None, ge=0, le=2)


class UnitDetection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enemy_id: int = Field(ge=1)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    source: AnnotationSource = AnnotationSource.AUTO


class SideData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roster: list[RosterEntry] = Field(default_factory=list)
    units: list[UnitDetection] = Field(default_factory=list)

    def roster_counts(self) -> Counter[int]:
        counts: Counter[int] = Counter()
        for entry in self.roster:
            counts[entry.enemy_id] += entry.count
        return counts

    def unit_counts(self) -> Counter[int]:
        return Counter(unit.enemy_id for unit in self.units)


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_relpath: str = Field(min_length=1)
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Timestamps(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prep: float = Field(ge=0.0)
    layout: float = Field(ge=0.0)
    battle_start: float = Field(ge=0.0)
    battle_end: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_order(self) -> Timestamps:
        values = (self.prep, self.layout, self.battle_start, self.battle_end)
        if values != tuple(sorted(values)):
            raise ValueError("round timestamps must be ordered")
        return self


class EvidenceFrames(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prep: str | None = None
    layout: str | None = None
    end: str | None = None


class RoundSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    sample_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source: SourceRef
    arena: Literal["green_vine"] = "green_vine"
    round_index: int = Field(ge=1)
    timestamps: Timestamps
    evidence: EvidenceFrames
    left: SideData
    right: SideData
    winner: Winner | None = None
    winner_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    review_status: ReviewStatus = ReviewStatus.PENDING
    pipeline_version: str = Field(min_length=1)
    failure_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_accepted_sample(self) -> RoundSample:
        if self.review_status is ReviewStatus.ACCEPTED:
            if self.winner is None:
                raise ValueError("accepted sample requires a winner")
            for side_name, side in (("left", self.left), ("right", self.right)):
                if not side.roster:
                    raise ValueError(f"{side_name} requires a non-empty roster")
                if side.roster_counts() != side.unit_counts():
                    raise ValueError(f"{side_name} roster counts must equal detected unit counts")
        return self
