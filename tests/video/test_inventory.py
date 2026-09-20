from pathlib import Path

import cv2
import numpy as np

from maa_duel.video.inventory import Arena, scan_inventory
from maa_duel.store import read_jsonl


def write_video(path: Path, width=320, height=180, frame_count=6):
    path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    assert writer.isOpened()
    for value in range(frame_count):
        writer.write(np.full((height, width, 3), value * 20, dtype=np.uint8))
    writer.release()


def test_scan_inventory_probes_hashes_and_classifies_video(tmp_path):
    input_dir = tmp_path / "training-data"
    video = input_dir / "绿藤城" / "rounds.mp4"
    write_video(video)
    manifest = tmp_path / "workspace" / "manifests" / "videos.jsonl"

    records = scan_inventory(input_dir, manifest)

    assert len(records) == 1
    record = records[0]
    assert record.relative_path == "绿藤城/rounds.mp4"
    assert record.arena is Arena.GREEN_VINE
    assert record.width == 320
    assert record.height == 180
    assert record.duration > 0
    assert record.sha256 == record.sha256.lower()
    assert len(record.sha256) == 64
    assert read_jsonl(manifest, type(record)) == records


def test_scan_inventory_reuses_unchanged_record(tmp_path):
    input_dir = tmp_path / "training-data"
    video = input_dir / "IvyVine" / "rounds.mp4"
    write_video(video)
    manifest = tmp_path / "workspace" / "manifests" / "videos.jsonl"

    first = scan_inventory(input_dir, manifest)
    second = scan_inventory(input_dir, manifest)

    assert second == first
