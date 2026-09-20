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
    labels = list((workspace / "synthetic" / "battlefield" / "labels" / "train").glob("*.txt"))
    assert len(labels) == 2
    assert all(label.read_text(encoding="utf-8").strip().startswith("0 ") for label in labels)
    config = yaml.safe_load((workspace / "synthetic" / "battlefield" / "dataset.yaml").read_text(encoding="utf-8"))
    assert config["val"] == "images/train"
    assert config["names"] == {0: "5"}
