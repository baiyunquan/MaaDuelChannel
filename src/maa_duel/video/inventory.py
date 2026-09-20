from __future__ import annotations

import hashlib
import json
import subprocess
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from maa_duel.store import read_jsonl, write_jsonl


class Arena(StrEnum):
    GREEN_VINE = "green_vine"
    GREEN_GRASS = "green_grass"
    HONEY_FRUIT = "honey_fruit"
    OTHER = "other"


class VideoRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration: float = Field(gt=0.0)
    fps: float = Field(gt=0.0)
    arena: Arena


def classify_arena(path: Path) -> Arena:
    value = path.as_posix().casefold().replace("-", "").replace("_", "").replace(" ", "")
    if any(token in value for token in ("绿藤", "greenvine", "ivyvine")):
        return Arena.GREEN_VINE
    if any(token in value for token in ("青草", "greengrass")):
        return Arena.GREEN_GRASS
    if any(token in value for token in ("蜜果", "honeyfruit", "honeydew")):
        return Arena.HONEY_FRUIT
    return Arena.OTHER


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_fps(value: str) -> float:
    fraction = Fraction(value)
    result = float(fraction)
    if result <= 0:
        raise ValueError(f"invalid frame rate: {value}")
    return result


def probe_video(path: Path) -> tuple[int, int, float, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    payload = json.loads(completed.stdout)
    if not payload.get("streams"):
        raise ValueError(f"video contains no video stream: {path}")
    stream = payload["streams"][0]
    fps_value = stream.get("avg_frame_rate") or stream["r_frame_rate"]
    return (
        int(stream["width"]),
        int(stream["height"]),
        float(payload["format"]["duration"]),
        _parse_fps(fps_value),
    )


def scan_inventory(input_dir: Path, manifest_path: Path) -> list[VideoRecord]:
    input_dir = input_dir.resolve()
    existing = {row.relative_path: row for row in read_jsonl(manifest_path, VideoRecord)}
    records: list[VideoRecord] = []
    video_paths = sorted(input_dir.rglob("*.mp4"), key=lambda item: item.as_posix().casefold())

    for path in video_paths:
        stat = path.stat()
        relative = path.relative_to(input_dir).as_posix()
        cached = existing.get(relative)
        if cached is not None and cached.size_bytes == stat.st_size and cached.mtime_ns == stat.st_mtime_ns:
            records.append(cached)
            continue

        width, height, duration, fps = probe_video(path)
        records.append(
            VideoRecord(
                relative_path=relative,
                sha256=file_sha256(path),
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                width=width,
                height=height,
                duration=duration,
                fps=fps,
                arena=classify_arena(Path(relative)),
            )
        )

    write_jsonl(manifest_path, records)
    return records
