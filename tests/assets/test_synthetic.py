import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

from maa_duel.assets import sync_assets
from maa_duel.synthetic import generate_synthetic_dataset


def write_rgba(path: Path, color):
    image = np.zeros((40, 40, 4), dtype=np.uint8)
    image[5:35, 8:32, :3] = color
    image[5:35, 8:32, 3] = 255
    assert cv2.imwrite(str(path), image)


def test_generate_synthetic_dataset_writes_all_train_layouts(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_rgba(source / "portrait.png", (0, 200, 200))
    write_rgba(source / "sprite.png", (0, 100, 255))
    catalog = source / "catalog.csv"
    with catalog.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "name", "original_name", "portrait", "animation"])
        writer.writeheader()
        writer.writerow(
            {"id": 5, "name": "unit", "original_name": "unit", "portrait": "portrait.png", "animation": "sprite.png"}
        )

    workspace = tmp_path / "workspace"
    sync_assets(catalog, workspace)
    background_dir = workspace / "assets" / "backgrounds"
    background_dir.mkdir(parents=True)
    assert cv2.imwrite(str(background_dir / "arena.png"), np.full((180, 320, 3), 80, dtype=np.uint8))

    result = generate_synthetic_dataset(
        workspace,
        portrait_variants=2,
        detection_images=2,
        seed=7,
    )

    assert result.portrait_images == 2
    assert result.detection_images == 2
    assert len(list((workspace / "synthetic" / "roster" / "train" / "0005").glob("*.jpg"))) == 2
    assert len(list((workspace / "synthetic" / "roster" / "val" / "0005").glob("*.jpg"))) == 2
    labels = list((workspace / "synthetic" / "battlefield" / "labels" / "train").glob("*.txt"))
    assert len(labels) == 2
    assert all(label.read_text(encoding="utf-8").strip().startswith("0 ") for label in labels)
    config = yaml.safe_load((workspace / "synthetic" / "battlefield" / "dataset.yaml").read_text(encoding="utf-8"))
    assert config["val"] == "images/train"
    assert config["names"] == {0: "5"}


def test_regeneration_replaces_stale_classes_and_images(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for enemy_id in (1, 2):
        write_rgba(source / f"portrait-{enemy_id}.png", (0, 100 * enemy_id, 200))
        write_rgba(source / f"sprite-{enemy_id}.png", (0, 50 * enemy_id, 255))
    catalog = source / "catalog.csv"

    def write_catalog(enemy_ids):
        with catalog.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "name", "portrait", "animation"])
            writer.writeheader()
            for enemy_id in enemy_ids:
                writer.writerow(
                    {
                        "id": enemy_id,
                        "name": f"unit-{enemy_id}",
                        "portrait": f"portrait-{enemy_id}.png",
                        "animation": f"sprite-{enemy_id}.png",
                    }
                )

    workspace = tmp_path / "workspace"
    write_catalog([1, 2])
    sync_assets(catalog, workspace)
    background_dir = workspace / "assets" / "backgrounds"
    background_dir.mkdir(parents=True)
    assert cv2.imwrite(str(background_dir / "arena.png"), np.full((90, 160, 3), 80, dtype=np.uint8))
    generate_synthetic_dataset(workspace, portrait_variants=1, detection_images=3, seed=1)

    write_catalog([2])
    sync_assets(catalog, workspace)
    generate_synthetic_dataset(workspace, portrait_variants=1, detection_images=1, seed=2)

    roster = workspace / "synthetic" / "roster"
    assert sorted(path.name for path in (roster / "train").iterdir()) == ["0002"]
    assert sorted(path.name for path in (roster / "val").iterdir()) == ["0002"]
    assert len(list((workspace / "synthetic" / "battlefield" / "images" / "train").glob("*.jpg"))) == 1
    assert len(list((workspace / "synthetic" / "battlefield" / "labels" / "train").glob("*.txt"))) == 1


def test_generate_synthetic_dataset_uses_sprites_bank_idle_and_move(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_rgba(source / "portrait.png", (0, 200, 200))
    catalog = source / "catalog.csv"
    with catalog.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "name", "original_name", "portrait"])
        writer.writeheader()
        writer.writerow({"id": 10, "name": "unit-10", "original_name": "unit-10", "portrait": "portrait.png"})
        writer.writerow({"id": 20, "name": "unit-20", "original_name": "unit-20", "portrait": "portrait.png"})

    workspace = tmp_path / "workspace"
    sync_assets(catalog, workspace)
    background_dir = workspace / "assets" / "backgrounds"
    background_dir.mkdir(parents=True)
    assert cv2.imwrite(str(background_dir / "arena.png"), np.full((180, 320, 3), 80, dtype=np.uint8))

    # Populate sprites bank with idle and move
    for eid in (10, 20):
        idle_dir = workspace / "assets" / "sprites" / f"{eid:04d}" / "idle"
        move_dir = workspace / "assets" / "sprites" / f"{eid:04d}" / "move"
        idle_dir.mkdir(parents=True)
        move_dir.mkdir(parents=True)
        write_rgba(idle_dir / "00.png", (50, 50, 200))
        write_rgba(move_dir / "00.png", (200, 50, 50))

    result = generate_synthetic_dataset(
        workspace,
        portrait_variants=1,
        detection_images=4,
        seed=42,
    )

    assert result.detection_images == 4
    assert result.skipped_animations == ()
    config = yaml.safe_load((workspace / "synthetic" / "battlefield" / "dataset.yaml").read_text(encoding="utf-8"))
    assert config["names"] == {0: "10", 1: "20"}
    class_map = yaml.safe_load((workspace / "synthetic" / "battlefield" / "class-map.json").read_text(encoding="utf-8"))
    assert class_map == {"0": 10, "1": 20}


def test_generate_synthetic_dataset_spawns_composite_squad(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_rgba(source / "portrait.png", (0, 200, 200))
    catalog = source / "catalog.csv"
    with catalog.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "name", "original_name", "portrait"])
        writer.writeheader()
        writer.writerow({"id": 58, "name": "trio", "original_name": "trio", "portrait": "portrait.png"})

    workspace = tmp_path / "workspace"
    sync_assets(catalog, workspace)
    background_dir = workspace / "assets" / "backgrounds"
    background_dir.mkdir(parents=True)
    assert cv2.imwrite(str(background_dir / "arena.png"), np.full((180, 320, 3), 80, dtype=np.uint8))

    # Populate 58, 93, 94 sprites
    for eid in (58, 93, 94):
        idle_dir = workspace / "assets" / "sprites" / f"{eid:04d}" / "idle"
        move_dir = workspace / "assets" / "sprites" / f"{eid:04d}" / "move"
        idle_dir.mkdir(parents=True)
        move_dir.mkdir(parents=True)
        write_rgba(idle_dir / "00.png", (50, 50, 200))
        write_rgba(move_dir / "00.png", (200, 50, 50))

    result = generate_synthetic_dataset(
        workspace,
        portrait_variants=1,
        detection_images=2,
        seed=123,
    )

    assert result.detection_images == 2
    # Verify entity_to_contestant.json
    map_text = (workspace / "synthetic" / "battlefield" / "entity_to_contestant.json").read_text(encoding="utf-8")
    e_map = yaml.safe_load(map_text)
    assert e_map["93"] == 58
    assert e_map["94"] == 58
    assert e_map["58"] == 58

    # Each generated label must contain 3 lines (58, 93, 94 spawned together)
    labels = list((workspace / "synthetic" / "battlefield" / "labels" / "train").glob("*.txt"))
    for lbl in labels:
        lines = lbl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3


