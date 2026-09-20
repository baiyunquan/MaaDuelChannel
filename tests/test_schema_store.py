import json

import pytest
from pydantic import ValidationError

from maa_duel.schema import (
    BoundingBox,
    EvidenceFrames,
    ReviewStatus,
    RosterEntry,
    RoundSample,
    SideData,
    SourceRef,
    Timestamps,
    UnitDetection,
    Winner,
)
from maa_duel.store import read_jsonl, write_jsonl


def make_sample(status=ReviewStatus.ACCEPTED):
    left = SideData(
        roster=[RosterEntry(enemy_id=1, count=1, confidence=0.9)],
        units=[
            UnitDetection(
                enemy_id=1,
                x=0.2,
                y=0.4,
                bbox=BoundingBox(x1=0.1, y1=0.2, x2=0.3, y2=0.4),
                confidence=0.8,
            )
        ],
    )
    right = SideData(
        roster=[RosterEntry(enemy_id=2, count=1, confidence=0.9)],
        units=[
            UnitDetection(
                enemy_id=2,
                x=0.8,
                y=0.4,
                bbox=BoundingBox(x1=0.7, y1=0.2, x2=0.9, y2=0.4),
                confidence=0.8,
            )
        ],
    )
    return RoundSample(
        sample_id="a" * 32,
        source=SourceRef(video_relpath="green/a.mp4", video_sha256="b" * 64),
        round_index=1,
        timestamps=Timestamps(prep=1.0, layout=2.0, battle_start=3.0, battle_end=8.0),
        evidence=EvidenceFrames(prep="frames/prep.jpg", layout="frames/layout.jpg", end="frames/end.jpg"),
        left=left,
        right=right,
        winner=Winner.LEFT,
        review_status=status,
        pipeline_version="0.1.0",
    )


def test_accepted_sample_requires_roster_unit_consistency():
    sample = make_sample(ReviewStatus.PENDING)
    sample.left.units = []

    with pytest.raises(ValidationError, match="roster counts"):
        RoundSample.model_validate(
            {
                **sample.model_dump(mode="json"),
                "review_status": ReviewStatus.ACCEPTED,
            }
        )


def test_coordinates_must_be_normalized():
    with pytest.raises(ValidationError):
        UnitDetection(
            enemy_id=1,
            x=1.1,
            y=0.5,
            bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.2, y2=0.2),
            confidence=0.8,
        )


def test_jsonl_round_trip_is_deterministic(tmp_path):
    path = tmp_path / "rounds.jsonl"
    sample = make_sample()

    write_jsonl(path, [sample])
    loaded = read_jsonl(path, RoundSample)

    assert loaded == [sample]
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["schema_version"] == 1
