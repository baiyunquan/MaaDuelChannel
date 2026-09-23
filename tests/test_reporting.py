import json

from maa_duel.calibration import BattlefieldCalibration, CalibrationPoint, save_calibration
from maa_duel.dataset import build_predictor_dataset
from maa_duel.extraction_state import ActiveExtraction
from maa_duel.reporting import write_report
from maa_duel.review import ReviewCorrection, ReviewStore
from maa_duel.schema import ReviewStatus, Winner
from maa_duel.store import write_jsonl
from maa_duel.video.inventory import Arena, VideoRecord
from tests.review.test_review import make_sample


def test_report_summarizes_inventory_rounds_and_training_dataset(tmp_path):
    workspace = tmp_path / "workspace"
    save_calibration(
        workspace / "assets" / "calibration" / "green-vine-video-v1.json",
        BattlefieldCalibration(
            calibration_id="green-vine-video-v1",
            image_width=1500,
            image_height=1100,
            map_width=15,
            map_height=11,
            fit_points=[
                CalibrationPoint(image_x=x * 100, image_y=y * 100, ground_x=x, ground_y=y)
                for x, y in ((0, 0), (15, 0), (0, 11), (15, 11), (7.5, 0), (7.5, 11), (0, 5.5), (15, 5.5))
            ],
            check_points=[
                CalibrationPoint(image_x=300, image_y=400, ground_x=3, ground_y=4),
                CalibrationPoint(image_x=1200, image_y=900, ground_x=12, ground_y=9),
            ],
        ),
    )
    write_jsonl(
        workspace / "manifests" / "videos.jsonl",
        [
            VideoRecord(
                relative_path="green/a.mp4",
                sha256="a" * 64,
                size_bytes=10,
                mtime_ns=1,
                width=1920,
                height=1080,
                duration=12.5,
                fps=30,
                arena=Arena.GREEN_VINE,
            )
        ],
    )
    write_jsonl(
        workspace / "manifests" / "rounds.auto.jsonl",
        [
            make_sample("b" * 32, ReviewStatus.ACCEPTED),
            make_sample("c" * 32, ReviewStatus.PENDING),
        ],
    )
    build_predictor_dataset(workspace)

    report_path = write_report(workspace)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["videos"]["green_vine"]["count"] == 1
    assert payload["videos"]["green_vine"]["duration_seconds"] == 12.5
    assert payload["rounds"]["accepted"] == 1
    assert payload["rounds"]["pending"] == 1
    assert payload["predictor"]["written_samples"] == 1
    assert (workspace / "reports" / "summary.md").is_file()


def test_report_reads_all_extraction_inputs_from_active_run_and_ignores_legacy(tmp_path):
    workspace = tmp_path / "workspace"
    write_jsonl(
        workspace / "manifests" / "videos.jsonl",
        [
            VideoRecord(
                relative_path="green/a.mp4",
                sha256="a" * 64,
                size_bytes=10,
                mtime_ns=1,
                width=1920,
                height=1080,
                duration=12.5,
                fps=30,
                arena=Arena.GREEN_VINE,
            )
        ],
    )

    legacy_accepted = make_sample("d" * 32, ReviewStatus.ACCEPTED)
    legacy_pending = make_sample("e" * 32, ReviewStatus.PENDING)
    write_jsonl(
        workspace / "manifests" / "rounds.auto.jsonl",
        [legacy_accepted, legacy_pending],
    )
    legacy_corrected = make_sample("e" * 32, ReviewStatus.ACCEPTED)
    ReviewStore(workspace / "review" / "corrections.jsonl").save(
        ReviewCorrection(sample=legacy_corrected, note="legacy correction must be ignored")
    )
    legacy_errors = workspace / "reports" / "extraction-errors.json"
    legacy_errors.parent.mkdir(parents=True, exist_ok=True)
    legacy_errors.write_text(json.dumps([{"kind": "legacy"}] * 5), encoding="utf-8")

    run_id = "20260923T120000000000Z-active01"
    run_root = workspace / "extraction-runs" / run_id
    active_auto = make_sample("f" * 32, ReviewStatus.PENDING)
    write_jsonl(run_root / "manifests" / "rounds.auto.jsonl", [active_auto])
    active_corrected = make_sample("f" * 32, ReviewStatus.ACCEPTED)
    active_corrected.winner = Winner.RIGHT
    ReviewStore(run_root / "review" / "corrections.jsonl").save(
        ReviewCorrection(sample=active_corrected, note="active correction")
    )
    active_errors = run_root / "reports" / "extraction-errors.json"
    active_errors.parent.mkdir(parents=True, exist_ok=True)
    active_errors.write_text(json.dumps([{"kind": "active"}] * 2), encoding="utf-8")
    (workspace / "manifests" / "extraction.active.json").write_text(
        ActiveExtraction(run_id=run_id).model_dump_json(),
        encoding="utf-8",
    )

    report_path = write_report(workspace)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["rounds"]["accepted"] == 1
    assert payload["rounds"]["pending"] == 0
    assert payload["rounds"]["winner_left"] == 0
    assert payload["rounds"]["winner_right"] == 1
    assert payload["extraction_error_count"] == 2
