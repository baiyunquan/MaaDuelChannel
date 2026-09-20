from datetime import UTC, datetime

from maa_duel.review import ReviewCorrection, ReviewStore, apply_table_edits
from maa_duel.schema import (
    AnnotationSource,
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


def make_sample(sample_id="a" * 32, status=ReviewStatus.ACCEPTED):
    return RoundSample(
        sample_id=sample_id,
        source=SourceRef(video_relpath="green/video.mp4", video_sha256="b" * 64),
        round_index=1,
        timestamps=Timestamps(prep=1, layout=2, battle_start=3, battle_end=4),
        evidence=EvidenceFrames(),
        left=SideData(
            roster=[RosterEntry(enemy_id=1, count=1, confidence=0.9)],
            units=[
                UnitDetection(
                    enemy_id=1,
                    x=0.2,
                    y=0.4,
                    bbox=BoundingBox(x1=0.1, y1=0.2, x2=0.3, y2=0.4),
                    confidence=0.9,
                )
            ],
        ),
        right=SideData(
            roster=[RosterEntry(enemy_id=2, count=1, confidence=0.9)],
            units=[
                UnitDetection(
                    enemy_id=2,
                    x=0.8,
                    y=0.4,
                    bbox=BoundingBox(x1=0.7, y1=0.2, x2=0.9, y2=0.4),
                    confidence=0.9,
                )
            ],
        ),
        winner=Winner.LEFT,
        review_status=status,
        pipeline_version="0.1.0",
    )


def test_review_store_overlays_manual_correction(tmp_path):
    store = ReviewStore(tmp_path / "corrections.jsonl")
    corrected = make_sample()
    corrected.winner = Winner.RIGHT
    store.save(
        ReviewCorrection(
            sample=corrected,
            reviewed_at=datetime.now(UTC),
            note="manual winner check",
        )
    )

    effective = store.overlay([make_sample()])

    assert effective[0].winner is Winner.RIGHT
    assert store.load()[corrected.sample_id].note == "manual winner check"


def test_apply_table_edits_marks_units_manual_and_validates_acceptance():
    sample = make_sample(status=ReviewStatus.PENDING)

    edited = apply_table_edits(
        sample,
        roster_rows=[["left", 1, 1, 1.0], ["right", 2, 1, 1.0]],
        unit_rows=[
            ["left", 1, 0.25, 0.4, 0.15, 0.2, 0.35, 0.4, 1.0],
            ["right", 2, 0.75, 0.4, 0.65, 0.2, 0.85, 0.4, 1.0],
        ],
        winner="right",
        status=ReviewStatus.ACCEPTED,
    )

    assert edited.review_status is ReviewStatus.ACCEPTED
    assert edited.winner is Winner.RIGHT
    assert edited.left.units[0].source is AnnotationSource.MANUAL
