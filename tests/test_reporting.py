import json

from maa_duel.calibration import BattlefieldCalibration, CalibrationPoint, save_calibration
from maa_duel.dataset import build_predictor_dataset
from maa_duel.reporting import write_report
from maa_duel.schema import ReviewStatus
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
