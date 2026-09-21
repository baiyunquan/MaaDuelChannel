import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from maa_duel.assets import sync_assets
from maa_duel.calibration import BattlefieldCalibration, CalibrationPoint, save_calibration
from maa_duel.combat import CombatKnowledge, CombatStats, EnemyCombatProfile, StageRules, save_combat_knowledge
from maa_duel.dataset import build_predictor_dataset
from maa_duel.reporting import write_report
from maa_duel.schema import ReviewStatus, Winner
from maa_duel.store import write_jsonl
from maa_duel.synthetic import generate_synthetic_dataset
from maa_duel.training.predictor import train_predictor_model
from tests.review.test_review import make_sample


def write_rgba(path: Path, color):
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    image[4:28, 4:28, :3] = color
    image[4:28, 4:28, 3] = 255
    assert cv2.imwrite(str(path), image)


def write_test_calibration(workspace):
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


def test_synthetic_end_to_end_pipeline(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_rgba(source / "portrait.png", (0, 255, 0))
    write_rgba(source / "sprite.png", (0, 0, 255))
    backgrounds = source / "backgrounds"
    backgrounds.mkdir()
    assert cv2.imwrite(str(backgrounds / "arena.png"), np.full((90, 160, 3), 60, dtype=np.uint8))
    catalog = source / "catalog.csv"
    with catalog.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "name", "portrait", "animation"])
        writer.writeheader()
        writer.writerow({"id": 1, "name": "unit", "portrait": "portrait.png", "animation": "sprite.png"})

    workspace = tmp_path / "workspace"
    sync_assets(catalog, workspace, background_dir=backgrounds)
    write_test_calibration(workspace)
    save_combat_knowledge(
        workspace / "assets" / "combat" / "vs2_enemy_combat.json",
        CombatKnowledge(
            stage_id="VS-2",
            stage_title="争锋对决！",
            source_page="https://prts.wiki/w/VS-2",
            source_revision=1,
            fetched_at=datetime(2026, 9, 21, tzinfo=UTC),
            rules=StageRules(raw_text=""),
            enemies=[
                EnemyCombatProfile(
                    enemy_id=enemy_id,
                    display_name=f"unit-{enemy_id}",
                    portrait_name=f"unit-{enemy_id}",
                    page_name=f"unit-{enemy_id}",
                    count_raw="1",
                    stats=CombatStats(
                        hp=1000,
                        attack=100,
                        defense=50,
                        resistance=0,
                        attack_interval=1,
                        weight=1,
                        move_speed=1,
                        attack_radius=1,
                    ),
                )
                for enemy_id in (1, 2)
            ],
        ),
    )
    synthetic = generate_synthetic_dataset(workspace, portrait_variants=1, detection_images=1, seed=1)
    left = make_sample("1" * 32, ReviewStatus.ACCEPTED)
    right = make_sample("2" * 32, ReviewStatus.ACCEPTED)
    right.winner = Winner.RIGHT
    write_jsonl(workspace / "manifests" / "rounds.auto.jsonl", [left, right])
    build_predictor_dataset(workspace)
    training = train_predictor_model(
        workspace,
        all_samples=True,
        epochs=1,
        batch_size=2,
        embedding_dim=8,
        heads=2,
        layers=1,
        device="cpu",
    )
    report = json.loads(write_report(workspace).read_text(encoding="utf-8"))

    assert synthetic.portrait_images == 1
    assert synthetic.detection_images == 1
    assert training["training_samples"] == 2
    assert report["predictor"]["written_samples"] == 2
