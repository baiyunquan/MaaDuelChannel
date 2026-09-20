import csv
from pathlib import Path

import cv2
import numpy as np

from maa_duel.assets import load_asset_manifest, sync_assets


def write_image(path: Path, color):
    image = np.full((32, 32, 3), color, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_sync_assets_imports_local_sources_and_reports_missing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_image(source / "portrait.png", (10, 20, 30))
    write_image(source / "sprite.png", (30, 20, 10))
    catalog = source / "catalog.csv"
    with catalog.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "名称", "原始名称", "portrait", "animation"])
        writer.writeheader()
        writer.writerow(
            {"id": 1, "名称": "甲", "原始名称": "A", "portrait": "portrait.png", "animation": "sprite.png"}
        )
        writer.writerow({"id": 2, "名称": "乙", "原始名称": "B", "portrait": "", "animation": ""})

    manifest = sync_assets(catalog, tmp_path / "workspace")
    loaded = load_asset_manifest(tmp_path / "workspace" / "assets" / "catalog.json")

    assert loaded == manifest
    assert len(manifest.enemies) == 2
    assert manifest.enemies[0].portrait is not None
    assert len(manifest.enemies[0].portrait.sha256) == 64
    assert (tmp_path / "workspace" / manifest.enemies[0].portrait.relative_path).is_file()
    assert manifest.enemies[1].missing == ["portrait", "animation"]


def test_sync_assets_imports_background_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_image(source / "portrait.png", (10, 20, 30))
    write_image(source / "sprite.png", (30, 20, 10))
    backgrounds = source / "backgrounds"
    backgrounds.mkdir()
    write_image(backgrounds / "arena.png", (50, 50, 50))
    catalog = source / "catalog.csv"
    with catalog.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "name", "portrait", "animation"])
        writer.writeheader()
        writer.writerow({"id": 1, "name": "unit", "portrait": "portrait.png", "animation": "sprite.png"})

    manifest = sync_assets(catalog, tmp_path / "workspace", background_dir=backgrounds)

    assert len(manifest.backgrounds) == 1
    assert (tmp_path / "workspace" / manifest.backgrounds[0].relative_path).is_file()
