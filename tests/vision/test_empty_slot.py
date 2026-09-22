import csv
import json
from pathlib import Path

import cv2
import numpy as np

from maa_duel.assets import find_empty_slot_image, sync_assets
from maa_duel.contracts import AnnotationTask
from maa_duel.sampling import _synthetic_base_records
from maa_duel.synthetic import generate_synthetic_dataset
from maa_duel.vision.roster import (
    Classification,
    RosterFrameRecognizer,
    SlotSpec,
)


def write_test_image(path: Path, shape=(64, 64, 3), color=60):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full(shape, color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_find_empty_slot_image_priority_and_cannotmax_migration(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # 1. When no images exist, returns None
    assert find_empty_slot_image(workspace) is None

    # 2. When CannotMax image exists, finds and copies it to workspace/assets/ui/empty_slot.png
    cannotmax_img = workspace / "CannotMax" / "images" / "empty.png"
    write_test_image(cannotmax_img)

    found = find_empty_slot_image(workspace)
    assert found is not None
    assert found == workspace / "assets" / "ui" / "empty_slot.png"
    assert found.is_file()

    # 3. Custom path takes highest priority
    custom_path = tmp_path / "custom_empty.png"
    write_test_image(custom_path, color=100)
    assert find_empty_slot_image(workspace, custom_path=custom_path) == custom_path.resolve()


def test_synthetic_dataset_with_empty_slot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_test_image(source / "portrait.png", (32, 32, 4), color=120)
    write_test_image(source / "sprite.png", (32, 32, 4), color=200)
    write_test_image(source / "empty.png", (40, 40, 3), color=40)
    backgrounds = source / "backgrounds"
    write_test_image(backgrounds / "bg.png", (90, 160, 3), color=50)

    catalog = source / "catalog.csv"
    with catalog.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "name", "portrait", "animation"])
        writer.writeheader()
        writer.writerow({"id": 1, "name": "slug", "portrait": "portrait.png", "animation": "sprite.png"})

    workspace = tmp_path / "workspace"
    sync_assets(catalog, workspace, background_dir=backgrounds)

    # Put empty slot image into assets/ui
    write_test_image(workspace / "assets" / "ui" / "empty_slot.png", (40, 40, 3), color=40)

    result = generate_synthetic_dataset(
        workspace,
        portrait_variants=3,
        detection_images=1,
        seed=42,
    )

    assert result.empty_slot_images == 3
    # 1 enemy * 3 + 1 empty * 3 = 6
    assert result.portrait_images == 6

    # Verify directory structure
    train_empty_dir = workspace / "synthetic" / "roster" / "train" / "0000"
    train_slug_dir = workspace / "synthetic" / "roster" / "train" / "0001"
    assert train_empty_dir.is_dir()
    assert train_slug_dir.is_dir()
    assert len(list(train_empty_dir.glob("*.jpg"))) == 3

    # Verify class-map.json: 0000 -> 0, 0001 -> 1
    class_map = json.loads((workspace / "synthetic" / "roster" / "class-map.json").read_text(encoding="utf-8"))
    assert class_map == {"0": 0, "1": 1}


def test_sampling_maps_0000_to_empty(tmp_path):
    workspace = tmp_path / "workspace"
    # Create fake synthetic roster train dir with 0000 and 0001
    write_test_image(workspace / "synthetic" / "roster" / "train" / "0000" / "0000-0000.jpg")
    write_test_image(workspace / "synthetic" / "roster" / "train" / "0001" / "0001-0000.jpg")

    records = _synthetic_base_records(workspace)
    assert len(records) == 2
    by_class = {r.ground_truth_class: r for r in records}
    assert "empty" in by_class
    assert "enemy_0001" in by_class
    assert by_class["empty"].task == AnnotationTask.ROSTER_CLASSIFICATION


def test_roster_recognizer_ignores_empty_slot():
    class DummyClassifier:
        def __init__(self, classifications):
            self.classifications = classifications

        def classify_batch(self, images):
            return self.classifications

    class DummyOcr:
        def recognize(self, image, *, detect=True):
            return []

    slots = [
        SlotSpec("left", 0, (0.1, 0.1, 0.2, 0.2), (0.1, 0.2, 0.2, 0.3)),
        SlotSpec("left", 1, (0.3, 0.1, 0.4, 0.2), (0.3, 0.2, 0.4, 0.3)),
    ]
    # slot 0 is empty (enemy_id=0), slot 1 is enemy 5
    recognizer = RosterFrameRecognizer(
        classifier=DummyClassifier(
            [
                Classification(enemy_id=0, confidence=0.95),
                Classification(enemy_id=5, confidence=0.9),
            ]
        ),
        ocr=DummyOcr(),
        slots=slots,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    observations = recognizer.recognize(frame)
    # Neither should produce observation without valid OCR count,
    # but even with count classifier, enemy_id < 1 is skipped immediately.
    assert len(observations) == 0
