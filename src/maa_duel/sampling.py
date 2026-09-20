from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path

import yaml

from maa_duel.contracts import (
    AnnotationBox,
    AnnotationRecord,
    AnnotationState,
    AnnotationTask,
    DataBucket,
    DatasetVersion,
    SamplingPolicy,
    git_commit,
    sha256_file,
    stable_annotation_id,
    write_contract,
)
from maa_duel.schema import BoundingBox
from maa_duel.store import read_jsonl, write_jsonl


def _label(record: AnnotationRecord) -> str:
    if record.task is AnnotationTask.BATTLEFIELD_DETECTION:
        return "+".join(sorted({box.class_name for box in record.ground_truth_boxes})) or "empty"
    return record.ground_truth_class or "unlabeled"


def _balanced_order(records: list[AnnotationRecord], seed: int) -> list[AnnotationRecord]:
    groups: dict[tuple[str, str], list[AnnotationRecord]] = defaultdict(list)
    for record in records:
        groups[(record.task.value, _label(record))].append(record)
    rng = random.Random(seed)
    queues: dict[tuple[str, str], deque[AnnotationRecord]] = {}
    for key, values in groups.items():
        rng.shuffle(values)
        queues[key] = deque(values)
    ordered: list[AnnotationRecord] = []
    keys = sorted(queues)
    while keys:
        next_keys = []
        for key in keys:
            queue = queues[key]
            if queue:
                ordered.append(queue.popleft())
            if queue:
                next_keys.append(key)
        keys = next_keys
    return ordered


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _synthetic_base_records(workspace: Path) -> list[AnnotationRecord]:
    records: list[AnnotationRecord] = []
    roster_root = workspace / "synthetic" / "roster" / "train"
    if roster_root.is_dir():
        for image in sorted(roster_root.glob("*/*")):
            if not image.is_file() or not image.parent.name.isdigit():
                continue
            relative = image.relative_to(workspace).as_posix()
            digest = sha256_file(image)
            annotation_id = stable_annotation_id("synthetic-roster", relative)
            class_name = f"enemy_{int(image.parent.name):04d}"
            records.append(
                AnnotationRecord(
                    annotation_id=annotation_id,
                    task=AnnotationTask.ROSTER_CLASSIFICATION,
                    sample_id=annotation_id,
                    source_image=relative,
                    exported_image=relative,
                    image_sha256=digest,
                    source_video_sha256=digest,
                    predicted_class=class_name,
                    ground_truth_class=class_name,
                    state=AnnotationState.REVIEWED,
                    model_version="synthetic-base",
                )
            )

    detection_root = workspace / "synthetic" / "battlefield"
    class_map_path = detection_root / "class-map.json"
    if class_map_path.is_file():
        class_map = {
            int(index): f"enemy_{int(enemy_id):04d}"
            for index, enemy_id in json.loads(class_map_path.read_text(encoding="utf-8")).items()
        }
        for image in sorted((detection_root / "images" / "train").glob("*")):
            if not image.is_file():
                continue
            label_path = detection_root / "labels" / "train" / f"{image.stem}.txt"
            boxes: list[AnnotationBox] = []
            if label_path.is_file():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    class_id, center_x, center_y, width, height = line.split()[:5]
                    cx, cy, w, h = map(float, (center_x, center_y, width, height))
                    boxes.append(
                        AnnotationBox(
                            class_name=class_map[int(class_id)],
                            bbox=BoundingBox(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2),
                            confidence=1.0,
                            side="left" if cx < 0.5 else "right",
                        )
                    )
            relative = image.relative_to(workspace).as_posix()
            digest = sha256_file(image)
            annotation_id = stable_annotation_id("synthetic-battlefield", relative)
            records.append(
                AnnotationRecord(
                    annotation_id=annotation_id,
                    task=AnnotationTask.BATTLEFIELD_DETECTION,
                    sample_id=annotation_id,
                    source_image=relative,
                    exported_image=relative,
                    image_sha256=digest,
                    source_video_sha256=digest,
                    predicted_boxes=boxes,
                    ground_truth_boxes=boxes,
                    state=AnnotationState.REVIEWED,
                    model_version="synthetic-base",
                )
            )
    return records


def _materialize_classification(
    workspace: Path,
    root: Path,
    task: AnnotationTask,
    records: list[AnnotationRecord],
) -> list[str]:
    names = sorted({record.ground_truth_class for record in records if record.ground_truth_class})
    for split in ("train", "val"):
        for name in names:
            (root / task.value / split / name).mkdir(parents=True, exist_ok=True)
    for record in records:
        if record.ground_truth_class is None:
            continue
        source = workspace / record.exported_image
        if not source.is_file():
            raise FileNotFoundError(f"annotation crop is missing: {source}")
        for split in ("train", "val"):
            _copy_or_link(source, root / task.value / split / record.ground_truth_class / f"{record.annotation_id}.jpg")
    class_map = {
        str(index): int(name.removeprefix("enemy_")) if name.startswith("enemy_") else name
        for index, name in enumerate(names)
    }
    (root / task.value / "class-map.json").write_text(
        json.dumps(class_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return names


def _materialize_detection(workspace: Path, root: Path, records: list[AnnotationRecord]) -> list[str]:
    task_root = root / AnnotationTask.BATTLEFIELD_DETECTION.value
    image_root = task_root / "images" / "train"
    label_root = task_root / "labels" / "train"
    names = sorted({box.class_name for record in records for box in record.ground_truth_boxes})
    class_index = {name: index for index, name in enumerate(names)}
    for record in records:
        source = workspace / record.exported_image
        if not source.is_file():
            raise FileNotFoundError(f"annotation image is missing: {source}")
        _copy_or_link(source, image_root / f"{record.annotation_id}.jpg")
        lines = []
        for box in record.ground_truth_boxes:
            center_x = (box.bbox.x1 + box.bbox.x2) / 2
            center_y = (box.bbox.y1 + box.bbox.y2) / 2
            width = box.bbox.x2 - box.bbox.x1
            height = box.bbox.y2 - box.bbox.y1
            lines.append(f"{class_index[box.class_name]} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}")
        (label_root / f"{record.annotation_id}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    config = {
        "path": str(task_root.resolve()),
        "train": "images/train",
        "val": "images/train",
        "names": {index: name for index, name in enumerate(names)},
    }
    (task_root / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (task_root / "class-map.json").write_text(
        json.dumps({str(index): int(name.removeprefix("enemy_")) for index, name in enumerate(names)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return names


def sample_training_sets(
    workspace: Path,
    annotation_version: str,
    *,
    output_version: str | None = None,
    policy: SamplingPolicy | None = None,
) -> DatasetVersion:
    """Keep every correction, then draw class-balanced replay and base examples without overlap."""

    policy = policy or SamplingPolicy()
    source_root = workspace / "annotations" / "versions" / annotation_version
    source_path = source_root / "annotations.jsonl"
    records = [
        record for record in read_jsonl(source_path, AnnotationRecord) if record.state is AnnotationState.REVIEWED
    ]
    if not records:
        raise ValueError(f"annotation version has no reviewed records: {annotation_version}")

    hard = sorted((record for record in records if record.corrected), key=lambda item: item.annotation_id)
    unchanged = [record for record in records if not record.corrected]
    synthetic_base = _synthetic_base_records(workspace)
    if policy.maximum_samples is not None:
        target = max(len(hard), policy.maximum_samples)
    elif hard and policy.hard_fraction > 0:
        target = min(len(records), max(len(hard), math.ceil(len(hard) / policy.hard_fraction)))
    else:
        target = len(records)
    remaining = max(0, target - len(hard))
    if hard:
        desired_replay = min(remaining, round(target * policy.replay_fraction))
    else:
        non_hard_weight = policy.replay_fraction + policy.base_fraction
        desired_replay = round(remaining * policy.replay_fraction / non_hard_weight) if non_hard_weight else 0
    desired_base = remaining - desired_replay
    replay_count = min(len(unchanged), desired_replay)
    base_count = min(len(synthetic_base), desired_base)
    shortage = remaining - replay_count - base_count
    if shortage > 0:
        extra_replay = min(len(unchanged) - replay_count, shortage)
        replay_count += extra_replay
        shortage -= extra_replay
    if shortage > 0:
        base_count += min(len(synthetic_base) - base_count, shortage)
    replay = _balanced_order(unchanged, policy.seed)[:replay_count]
    base = _balanced_order(synthetic_base, policy.seed + 1)[:base_count]

    selected: list[AnnotationRecord] = []
    for bucket, values in ((DataBucket.HARD, hard), (DataBucket.REPLAY, replay), (DataBucket.BASE, base)):
        for record in values:
            selected.append(record.model_copy(update={"bucket": bucket}))
    selected.sort(key=lambda item: (item.task.value, item.annotation_id))

    selection_digest = hashlib.sha256(
        "\n".join(f"{record.annotation_id}:{record.bucket}" for record in selected).encode("utf-8")
    ).hexdigest()[:12]
    output_version = output_version or f"{annotation_version}-sample-{selection_digest}"
    output_root = workspace / "datasets" / output_version
    if output_root.exists():
        raise FileExistsError(f"dataset version already exists: {output_root}")
    manifest_path = output_root / "annotations.jsonl"
    write_jsonl(manifest_path, selected)

    class_names: dict[str, list[str]] = {}
    for task in (AnnotationTask.ROSTER_CLASSIFICATION, AnnotationTask.OCR_CLASSIFICATION):
        task_records = [record for record in selected if record.task is task]
        if task_records:
            class_names[task.value] = _materialize_classification(workspace, output_root, task, task_records)
    detection_records = [record for record in selected if record.task is AnnotationTask.BATTLEFIELD_DETECTION]
    if detection_records:
        class_names[AnnotationTask.BATTLEFIELD_DETECTION.value] = _materialize_detection(
            workspace, output_root, detection_records
        )

    task_counts = Counter(record.task.value for record in selected)
    bucket_counts = Counter(record.bucket.value for record in selected if record.bucket is not None)
    contract = DatasetVersion(
        dataset_version=output_version,
        parent_version=annotation_version,
        annotation_manifest=manifest_path.relative_to(workspace).as_posix(),
        annotation_sha256=sha256_file(manifest_path),
        source_manifest_sha256=sha256_file(source_path),
        task_counts=dict(sorted(task_counts.items())),
        bucket_counts=dict(sorted(bucket_counts.items())),
        class_names=class_names,
        sampling_policy=policy,
        split_policy="all-training",
        git_commit=git_commit(),
    )
    write_contract(output_root / "dataset.json", contract)
    return contract
