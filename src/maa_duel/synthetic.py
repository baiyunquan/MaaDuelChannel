from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from maa_duel.assets import EnemyAsset, load_asset_manifest


@dataclass(frozen=True)
class SyntheticResult:
    portrait_images: int
    detection_images: int
    skipped_portraits: tuple[int, ...]
    skipped_animations: tuple[int, ...]


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read image asset: {path}")
    return image


def _rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 4:
        return image
    alpha = np.where(np.max(image, axis=2) > 8, 255, 0).astype(np.uint8)
    return np.dstack((image[:, :, :3], alpha))


def _sprite_frames(path: Path, maximum_frames: int = 24) -> list[np.ndarray]:
    if path.suffix.casefold() not in {".webm", ".mp4", ".mkv", ".avi"}:
        return [_rgba(_read_image(path))]
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open animation asset: {path}")
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = np.linspace(0, total - 1, num=min(maximum_frames, total), dtype=int)
    frames: list[np.ndarray] = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(_rgba(frame))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"animation contains no readable frames: {path}")
    return frames


def _composite(background: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    height, width = sprite.shape[:2]
    alpha = sprite[:, :, 3:4].astype(np.float32) / 255.0
    target = background[y : y + height, x : x + width]
    target[:] = (sprite[:, :, :3] * alpha + target * (1.0 - alpha)).astype(np.uint8)


def _augment_portrait(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rgba = _rgba(image)
    scale = float(rng.uniform(0.75, 1.05))
    size = max(24, round(96 * scale))
    resized = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA)
    canvas = np.full((112, 112, 3), int(rng.integers(15, 70)), dtype=np.uint8)
    x = (112 - size) // 2
    y = (112 - size) // 2
    _composite(canvas, resized, x, y)
    gain = float(rng.uniform(0.8, 1.2))
    canvas = np.clip(canvas.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return canvas


def _available(manifest, kind: str) -> list[tuple[EnemyAsset, Path]]:
    values = []
    for enemy in manifest.enemies:
        asset = getattr(enemy, kind)
        if asset is not None:
            values.append((enemy, Path(asset.relative_path)))
    return values


def generate_synthetic_dataset(
    workspace: Path,
    *,
    portrait_variants: int = 40,
    detection_images: int = 1000,
    seed: int = 20260920,
) -> SyntheticResult:
    manifest = load_asset_manifest(workspace / "assets" / "catalog.json")
    output = workspace / "synthetic"
    rng = np.random.default_rng(seed)
    available_portraits = _available(manifest, "portrait")
    available_animations = _available(manifest, "animation")
    class_ids = sorted(enemy.enemy_id for enemy, _ in available_animations)
    class_index = {enemy_id: index for index, enemy_id in enumerate(class_ids)}

    portrait_count = 0
    roster_map: dict[str, int] = {}
    for class_position, (enemy, relative_path) in enumerate(available_portraits):
        roster_map[str(class_position)] = enemy.enemy_id
        image = _read_image(workspace / relative_path)
        class_dir = output / "roster" / "train" / f"{enemy.enemy_id:04d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for variant in range(portrait_variants):
            augmented = _augment_portrait(image, rng)
            destination = class_dir / f"{enemy.enemy_id:04d}-{variant:04d}.jpg"
            cv2.imencode(".jpg", augmented, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(65, 96))])[1].tofile(destination)
            portrait_count += 1
    (output / "roster").mkdir(parents=True, exist_ok=True)
    (output / "roster" / "class-map.json").write_text(
        json.dumps(roster_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    backgrounds = [
        _read_image(path)[:, :, :3]
        for path in sorted((workspace / "assets" / "backgrounds").glob("*"))
        if path.is_file()
    ]
    if detection_images > 0 and not backgrounds:
        raise ValueError("battlefield synthesis requires at least one image in assets/backgrounds")
    sprite_bank = {
        enemy.enemy_id: _sprite_frames(workspace / relative_path)
        for enemy, relative_path in available_animations
    }
    image_dir = output / "battlefield" / "images" / "train"
    label_dir = output / "battlefield" / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    produced_detection_images = 0
    if detection_images > 0 and sprite_bank:
        enemy_ids = sorted(sprite_bank)
        for image_index in range(detection_images):
            background = backgrounds[int(rng.integers(0, len(backgrounds)))].copy()
            height, width = background.shape[:2]
            labels: list[str] = []
            unit_count = int(rng.integers(1, min(7, len(enemy_ids) + 3)))
            for _ in range(unit_count):
                enemy_id = int(rng.choice(enemy_ids))
                sprite = sprite_bank[enemy_id][int(rng.integers(0, len(sprite_bank[enemy_id])))]
                target_height = int(rng.uniform(0.10, 0.22) * height)
                target_width = max(4, round(sprite.shape[1] * target_height / sprite.shape[0]))
                resized = cv2.resize(sprite, (target_width, target_height), interpolation=cv2.INTER_AREA)
                side_left = bool(rng.integers(0, 2))
                center_x = float(rng.uniform(0.12, 0.43) if side_left else rng.uniform(0.57, 0.88))
                bottom_y = float(rng.uniform(0.25, 0.88))
                x = int(center_x * width - target_width / 2)
                y = int(bottom_y * height - target_height)
                x = max(0, min(width - target_width, x))
                y = max(0, min(height - target_height, y))
                _composite(background, resized, x, y)
                center_x_norm = (x + target_width / 2) / width
                center_y_norm = (y + target_height / 2) / height
                labels.append(
                    f"{class_index[enemy_id]} {center_x_norm:.6f} {center_y_norm:.6f} "
                    f"{target_width / width:.6f} {target_height / height:.6f}"
                )
            image_path = image_dir / f"synthetic-{image_index:06d}.jpg"
            cv2.imencode(".jpg", background, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(65, 96))])[1].tofile(
                image_path
            )
            (label_dir / f"synthetic-{image_index:06d}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
            produced_detection_images += 1

    dataset_config = {
        "path": str((output / "battlefield").resolve()),
        "train": "images/train",
        "val": "images/train",
        "names": {index: str(enemy_id) for enemy_id, index in class_index.items()},
    }
    (output / "battlefield" / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output / "battlefield" / "class-map.json").write_text(
        json.dumps({str(index): enemy_id for enemy_id, index in class_index.items()}, indent=2),
        encoding="utf-8",
    )
    portrait_ids = {enemy.enemy_id for enemy, _ in available_portraits}
    animation_ids = {enemy.enemy_id for enemy, _ in available_animations}
    all_ids = {enemy.enemy_id for enemy in manifest.enemies}
    return SyntheticResult(
        portrait_images=portrait_count,
        detection_images=produced_detection_images,
        skipped_portraits=tuple(sorted(all_ids - portrait_ids)),
        skipped_animations=tuple(sorted(all_ids - animation_ids)),
    )
