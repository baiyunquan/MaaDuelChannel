from maa_duel.extraction import assemble_round_sample, stable_sample_id
from maa_duel.schema import BoundingBox, ReviewStatus, RosterEntry, SourceRef, Winner
from maa_duel.video.phases import RoundWindow
from maa_duel.vision.layout import RawDetection


def test_assemble_round_sample_auto_accepts_consistent_high_confidence_round():
    window = RoundWindow(
        round_index=2,
        prep_time=1.0,
        layout_time=2.0,
        battle_start=2.5,
        battle_end=7.0,
        complete=True,
        failure_reasons=(),
    )
    source = SourceRef(video_relpath="绿藤/video.mp4", video_sha256="a" * 64)
    rosters = {
        "left": [RosterEntry(enemy_id=1, count=1, confidence=0.95)],
        "right": [RosterEntry(enemy_id=2, count=1, confidence=0.95)],
    }
    detections = [
        RawDetection(
            enemy_id=1,
            bbox=BoundingBox(x1=0.1, y1=0.2, x2=0.2, y2=0.4),
            confidence=0.9,
        ),
        RawDetection(
            enemy_id=2,
            bbox=BoundingBox(x1=0.8, y1=0.2, x2=0.9, y2=0.4),
            confidence=0.9,
        ),
    ]

    sample = assemble_round_sample(
        source=source,
        window=window,
        rosters=rosters,
        detections=detections,
        winner=Winner.LEFT,
        winner_confidence=0.9,
        evidence={"prep": "prep.jpg", "layout": "layout.jpg", "end": "end.jpg"},
    )

    assert sample.review_status is ReviewStatus.ACCEPTED
    assert sample.winner is Winner.LEFT
    assert len(sample.sample_id) == 32


def test_sample_id_does_not_change_when_layout_evidence_time_moves():
    source = SourceRef(video_relpath="green/video.mp4", video_sha256="d" * 64)

    assert stable_sample_id(source, 3, 12.1) == stable_sample_id(source, 3, 12.6)


def test_assemble_round_sample_queues_low_confidence_result_for_review():
    window = RoundWindow(1, 1.0, 2.0, 3.0, 4.0, True, ())
    source = SourceRef(video_relpath="x.mp4", video_sha256="b" * 64)

    sample = assemble_round_sample(
        source=source,
        window=window,
        rosters={
            "left": [RosterEntry(enemy_id=1, count=1, confidence=0.4)],
            "right": [RosterEntry(enemy_id=2, count=1, confidence=0.9)],
        },
        detections=[],
        winner=None,
        winner_confidence=None,
        evidence={},
    )

    assert sample.review_status is ReviewStatus.PENDING
    assert sample.failure_reasons


def test_assemble_round_sample_keeps_empty_rosters_pending():
    window = RoundWindow(1, 1.0, 2.0, 3.0, 4.0, True, ())
    source = SourceRef(video_relpath="x.mp4", video_sha256="c" * 64)

    sample = assemble_round_sample(
        source=source,
        window=window,
        rosters={"left": [], "right": []},
        detections=[],
        winner=Winner.LEFT,
        winner_confidence=0.9,
        evidence={},
    )

    assert sample.review_status is ReviewStatus.PENDING
    assert "left:roster_empty" in sample.failure_reasons
    assert "right:roster_empty" in sample.failure_reasons


def test_runtime_extraction_reports_missing_model_artifacts(tmp_path):
    from maa_duel.extraction import extract_rounds

    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    workspace = tmp_path / "workspace"

    try:
        extract_rounds(input_dir, workspace)
    except FileNotFoundError as exc:
        assert "roster" in str(exc)
    else:
        raise AssertionError("missing trained models must stop extraction")
