from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from maa_duel.assets import load_asset_manifest
from maa_duel.contracts import (
    AnnotationBox,
    AnnotationRecord,
    AnnotationState,
    AnnotationTask,
    DatasetVersion,
    git_commit,
    sha256_file,
    stable_annotation_id,
    write_contract,
)
from maa_duel.review import ReviewCorrection, ReviewStore
from maa_duel.schema import (
    AnnotationSource,
    BoundingBox,
    ReviewStatus,
    RosterEntry,
    RoundSample,
    SideData,
    UnitDetection,
)
from maa_duel.store import read_jsonl, write_jsonl
from maa_duel.vision.roster import crop_normalized, default_slot_specs

PLATFORM_URL = "https://platform.ultralytics.com/"


@dataclass(frozen=True)
class PlatformExport:
    export_id: str
    directory: Path
    archives: dict[AnnotationTask, Path]
    item_counts: dict[AnnotationTask, int]


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read annotation image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    if min(image.shape[:2]) < 32:
        scale = 32 / min(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError(f"cannot encode annotation image: {path}")
    encoded.tofile(path)


def _enemy_class(enemy_id: int) -> str:
    return f"enemy_{enemy_id:04d}"


def _enemy_id(class_name: str) -> int:
    if not class_name.startswith("enemy_"):
        raise ValueError(f"invalid enemy class name: {class_name}")
    return int(class_name.removeprefix("enemy_"))


def _zip_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source).as_posix()
            if path.is_dir():
                archive.writestr(f"{relative}/", b"")
            else:
                archive.write(path, relative)


def _class_names(workspace: Path, samples: list[RoundSample]) -> list[str]:
    catalog_path = workspace / "assets" / "catalog.json"
    if catalog_path.is_file():
        ids = [enemy.enemy_id for enemy in load_asset_manifest(catalog_path).enemies]
    else:
        ids = sorted(
            {
                entry.enemy_id
                for sample in samples
                for side in (sample.left, sample.right)
                for entry in (*side.roster, *side.units)
            }
        )
    return [_enemy_class(enemy_id) for enemy_id in sorted(ids)]


def export_platform_annotations(workspace: Path, *, export_id: str | None = None) -> PlatformExport:
    """Build three upload-ready Platform datasets and a stable local mapping manifest."""

    automatic = read_jsonl(workspace / "manifests" / "rounds.auto.jsonl", RoundSample)
    if not automatic:
        raise ValueError("automatic round manifest is empty; run extract first")
    samples = ReviewStore(workspace / "review" / "corrections.jsonl").overlay(automatic)
    export_id = export_id or datetime.now(UTC).strftime("platform-%Y%m%dT%H%M%S%fZ")
    root = workspace / "review" / "platform" / export_id
    if root.exists():
        raise FileExistsError(f"Platform export already exists: {root}")

    task_roots = {task: root / task.value for task in AnnotationTask}
    class_names = _class_names(workspace, samples)
    roster_classes = [*class_names, "empty", "unknown"]
    ocr_classes = [*(f"count_{count}" for count in range(1, 100)), "empty", "unreadable"]
    for class_name in roster_classes:
        (task_roots[AnnotationTask.ROSTER_CLASSIFICATION] / "train" / class_name).mkdir(parents=True, exist_ok=True)
    for class_name in ocr_classes:
        (task_roots[AnnotationTask.OCR_CLASSIFICATION] / "train" / class_name).mkdir(parents=True, exist_ok=True)
    detection_root = task_roots[AnnotationTask.BATTLEFIELD_DETECTION]
    (detection_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (detection_root / "labels" / "train").mkdir(parents=True, exist_ok=True)

    records: list[AnnotationRecord] = []
    for sample in samples:
        if sample.evidence.prep:
            prep_path = workspace / sample.evidence.prep
            prep = _read_image(prep_path)
            rosters = {"left": sample.left.roster, "right": sample.right.roster}
            for slot in default_slot_specs():
                roster_entry = rosters[slot.side][slot.index] if slot.index < len(rosters[slot.side]) else None
                roster_class = _enemy_class(roster_entry.enemy_id) if roster_entry else "empty"
                count_class = f"count_{roster_entry.count}" if roster_entry else "empty"
                for task, rect, class_name, text_value in (
                    (AnnotationTask.ROSTER_CLASSIFICATION, slot.icon_rect, roster_class, None),
                    (
                        AnnotationTask.OCR_CLASSIFICATION,
                        slot.count_rect,
                        count_class,
                        str(roster_entry.count) if roster_entry else None,
                    ),
                ):
                    annotation_id = stable_annotation_id(sample.sample_id, task.value, slot.side, slot.index)
                    exported = Path(task.value) / "train" / class_name / f"{annotation_id}.jpg"
                    destination = root / exported
                    _write_image(destination, crop_normalized(prep, rect))
                    records.append(
                        AnnotationRecord(
                            annotation_id=annotation_id,
                            task=task,
                            sample_id=sample.sample_id,
                            source_image=sample.evidence.prep,
                            exported_image=destination.relative_to(workspace).as_posix(),
                            image_sha256=sha256_file(destination),
                            source_video_sha256=sample.source.video_sha256,
                            side=slot.side,
                            slot=slot.index,
                            predicted_class=class_name,
                            predicted_text=text_value,
                            model_version=sample.pipeline_version,
                            failure_reasons=sample.failure_reasons,
                        )
                    )

        if sample.evidence.layout:
            layout_source = workspace / sample.evidence.layout
            annotation_id = stable_annotation_id(sample.sample_id, AnnotationTask.BATTLEFIELD_DETECTION.value)
            exported_image = (
                Path(AnnotationTask.BATTLEFIELD_DETECTION.value) / "images" / "train" / f"{annotation_id}.jpg"
            )
            destination = root / exported_image
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(layout_source, destination)
            boxes: list[AnnotationBox] = []
            for side_name, side in (("left", sample.left), ("right", sample.right)):
                boxes.extend(
                    AnnotationBox(
                        class_name=_enemy_class(unit.enemy_id),
                        bbox=unit.bbox,
                        confidence=unit.confidence,
                        side=side_name,
                    )
                    for unit in side.units
                )
            class_index = {name: index for index, name in enumerate(class_names)}
            label_lines = []
            for box in boxes:
                center_x = (box.bbox.x1 + box.bbox.x2) / 2
                center_y = (box.bbox.y1 + box.bbox.y2) / 2
                width = box.bbox.x2 - box.bbox.x1
                height = box.bbox.y2 - box.bbox.y1
                label_lines.append(
                    f"{class_index[box.class_name]} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}"
                )
            label_path = detection_root / "labels" / "train" / f"{annotation_id}.txt"
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            records.append(
                AnnotationRecord(
                    annotation_id=annotation_id,
                    task=AnnotationTask.BATTLEFIELD_DETECTION,
                    sample_id=sample.sample_id,
                    source_image=sample.evidence.layout,
                    exported_image=destination.relative_to(workspace).as_posix(),
                    image_sha256=sha256_file(destination),
                    source_video_sha256=sample.source.video_sha256,
                    predicted_boxes=boxes,
                    model_version=sample.pipeline_version,
                    failure_reasons=sample.failure_reasons,
                )
            )

    detection_config = {
        "path": ".",
        "train": "images/train",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    (detection_root / "data.yaml").write_text(
        yaml.safe_dump(detection_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    for task, names in (
        (AnnotationTask.ROSTER_CLASSIFICATION, roster_classes),
        (AnnotationTask.OCR_CLASSIFICATION, ocr_classes),
    ):
        (task_roots[task] / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    manifest_path = root / "annotation-manifest.jsonl"
    write_jsonl(manifest_path, records)
    counts = Counter(record.task for record in records)
    metadata = {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "export_id": export_id,
        "created_at": datetime.now(UTC).isoformat(),
        "platform_url": PLATFORM_URL,
        "manifest_sha256": sha256_file(manifest_path),
        "item_counts": {task.value: counts[task] for task in AnnotationTask},
        "class_names": {
            AnnotationTask.ROSTER_CLASSIFICATION.value: roster_classes,
            AnnotationTask.BATTLEFIELD_DETECTION.value: class_names,
            AnnotationTask.OCR_CLASSIFICATION.value: ocr_classes,
        },
    }
    (root / "export.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archives: dict[AnnotationTask, Path] = {}
    for task, task_root in task_roots.items():
        archive = root / f"{task.value}.zip"
        _zip_directory(task_root, archive)
        archives[task] = archive
    return PlatformExport(
        export_id=export_id,
        directory=root,
        archives=archives,
        item_counts={task: counts[task] for task in AnnotationTask},
    )


def _class_name(value: Any, names: dict[int, str]) -> str:
    if isinstance(value, dict):
        value = value.get("class_id", value.get("class", value.get("id")))
    if isinstance(value, list):
        if not value:
            raise ValueError("classification annotation is empty")
        value = value[0]
    if isinstance(value, str) and not value.isdigit():
        return value
    return names[int(value)]


def _classification_name(row: dict[str, Any], names: dict[int, str]) -> str:
    annotations = row.get("annotations") or {}
    for key in ("classification", "classify", "cls", "class"):
        if key in annotations:
            return _class_name(annotations[key], names)
    raise ValueError(f"Platform classification is missing for {row.get('file')}")


def _boxes(row: dict[str, Any], names: dict[int, str]) -> list[AnnotationBox]:
    output: list[AnnotationBox] = []
    for raw in (row.get("annotations") or {}).get("boxes", []):
        class_name = _class_name(raw[0], names)
        center_x, center_y, width, height = map(float, raw[1:5])
        bbox = BoundingBox(
            x1=max(0.0, center_x - width / 2),
            y1=max(0.0, center_y - height / 2),
            x2=min(1.0, center_x + width / 2),
            y2=min(1.0, center_y + height / 2),
        )
        output.append(
            AnnotationBox(
                class_name=class_name,
                bbox=bbox,
                confidence=1.0,
                side="left" if center_x < 0.5 else "right",
            )
        )
    return output


def _box_signature(boxes: list[AnnotationBox]) -> list[tuple[object, ...]]:
    return sorted(
        (
            box.class_name,
            round(box.bbox.x1, 5),
            round(box.bbox.y1, 5),
            round(box.bbox.x2, 5),
            round(box.bbox.y2, 5),
        )
        for box in boxes
    )


def _read_platform_ndjson(path: Path) -> tuple[dict[int, str], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or rows[0].get("type") != "dataset":
        raise ValueError(f"not an Ultralytics Platform NDJSON export: {path}")
    names = {int(index): str(name) for index, name in (rows[0].get("class_names") or {}).items()}
    return names, [row for row in rows[1:] if row.get("type") == "image"]


def import_platform_annotations(
    workspace: Path,
    export_directory: Path,
    platform_exports: dict[AnnotationTask, Path],
    *,
    version: str | None = None,
    parent_version: str | None = None,
) -> DatasetVersion:
    """Import Platform NDJSON snapshots into an immutable local annotation version."""

    manifest_path = export_directory / "annotation-manifest.jsonl"
    records = read_jsonl(manifest_path, AnnotationRecord)
    if not records:
        raise ValueError(f"annotation manifest is empty: {manifest_path}")
    by_id = {record.annotation_id: record for record in records}
    imported: list[AnnotationRecord] = []
    imported_ids: set[str] = set()
    class_names: dict[str, list[str]] = {}
    for task, path in platform_exports.items():
        names, rows = _read_platform_ndjson(path)
        class_names[task.value] = [names[index] for index in sorted(names)]
        for row in rows:
            annotation_id = Path(str(row.get("file", ""))).stem
            original = by_id.get(annotation_id)
            if original is None:
                raise ValueError(f"Platform item is not present in local manifest: {row.get('file')}")
            if original.task is not task:
                raise ValueError(f"task mismatch for annotation {annotation_id}: {original.task} != {task}")
            if annotation_id in imported_ids:
                raise ValueError(f"duplicate Platform annotation: {annotation_id}")
            imported_ids.add(annotation_id)
            payload = original.model_dump(mode="json")
            payload["state"] = AnnotationState.REVIEWED.value
            if task is AnnotationTask.BATTLEFIELD_DETECTION:
                ground_truth_boxes = _boxes(row, names)
                payload["ground_truth_boxes"] = [box.model_dump(mode="json") for box in ground_truth_boxes]
                payload["corrected"] = _box_signature(original.predicted_boxes) != _box_signature(ground_truth_boxes)
            else:
                ground_truth_class = _classification_name(row, names)
                payload["ground_truth_class"] = ground_truth_class
                payload["corrected"] = ground_truth_class != original.predicted_class
                if task is AnnotationTask.OCR_CLASSIFICATION:
                    payload["ground_truth_text"] = (
                        ground_truth_class.removeprefix("count_") if ground_truth_class.startswith("count_") else None
                    )
            imported.append(AnnotationRecord.model_validate(payload))

    if parent_version is not None:
        parent_root = workspace / "annotations" / "versions" / parent_version
        parent_records = read_jsonl(parent_root / "annotations.jsonl", AnnotationRecord)
        if not parent_records:
            raise ValueError(f"parent annotation version is missing or empty: {parent_version}")
        parent_contract = DatasetVersion.model_validate_json((parent_root / "dataset.json").read_text(encoding="utf-8"))
        class_names = {**parent_contract.class_names, **class_names}
        combined = {record.annotation_id: record for record in parent_records}
        combined.update({record.annotation_id: record for record in imported})
        imported = list(combined.values())

    imported.sort(key=lambda item: (item.task.value, item.annotation_id))
    version = version or datetime.now(UTC).strftime("v%Y%m%dT%H%M%S%fZ")
    version_root = workspace / "annotations" / "versions" / version
    if version_root.exists():
        raise FileExistsError(f"annotation version already exists: {version_root}")
    annotation_path = version_root / "annotations.jsonl"
    write_jsonl(annotation_path, imported)
    task_counts = Counter(record.task.value for record in imported)
    contract = DatasetVersion(
        dataset_version=version,
        parent_version=parent_version,
        annotation_manifest=annotation_path.relative_to(workspace).as_posix(),
        annotation_sha256=sha256_file(annotation_path),
        source_manifest_sha256=sha256_file(manifest_path),
        task_counts=dict(sorted(task_counts.items())),
        class_names=class_names,
        git_commit=git_commit(),
    )
    write_contract(version_root / "dataset.json", contract)
    _apply_imported_annotations(workspace, imported, version)
    return contract


def _apply_imported_annotations(workspace: Path, records: list[AnnotationRecord], version: str) -> None:
    automatic = read_jsonl(workspace / "manifests" / "rounds.auto.jsonl", RoundSample)
    store = ReviewStore(workspace / "review" / "corrections.jsonl")
    effective = {sample.sample_id: sample for sample in store.overlay(automatic)}
    grouped: dict[str, list[AnnotationRecord]] = {}
    for record in records:
        grouped.setdefault(record.sample_id, []).append(record)

    corrections: list[ReviewCorrection] = []
    for sample_id, sample_records in grouped.items():
        sample = effective.get(sample_id)
        if sample is None:
            raise ValueError(f"annotation refers to an unknown round sample: {sample_id}")
        roster_slots: dict[str, dict[int, list[int]]] = {
            "left": {index: [entry.enemy_id, entry.count] for index, entry in enumerate(sample.left.roster)},
            "right": {index: [entry.enemy_id, entry.count] for index, entry in enumerate(sample.right.roster)},
        }
        slot_labels: dict[tuple[str, int], dict[str, str]] = {}
        imported_boxes: list[AnnotationBox] | None = None
        for record in sample_records:
            if record.task is AnnotationTask.BATTLEFIELD_DETECTION:
                imported_boxes = record.ground_truth_boxes
                continue
            if record.side is None or record.slot is None or record.ground_truth_class is None:
                continue
            label_kind = "type" if record.task is AnnotationTask.ROSTER_CLASSIFICATION else "count"
            slot_labels.setdefault((record.side, record.slot), {})[label_kind] = record.ground_truth_class

        unresolved_roster = False
        for (side_name, slot), labels in slot_labels.items():
            type_label = labels.get("type")
            count_label = labels.get("count")
            slots = roster_slots[side_name]
            type_is_enemy = type_label is None or type_label.startswith("enemy_")
            count_is_number = count_label is None or count_label.startswith("count_")
            if not type_is_enemy or not count_is_number:
                slots.pop(slot, None)
                if type_label in {"unknown", "unreadable"} or count_label in {"unknown", "unreadable"}:
                    unresolved_roster = True
                if (type_label == "empty" and count_is_number and count_label is not None) or (
                    count_label == "empty" and type_is_enemy and type_label is not None
                ):
                    unresolved_roster = True
                continue
            current = slots.setdefault(slot, [0, 1])
            if type_label is not None:
                current[0] = _enemy_id(type_label)
            if count_label is not None:
                current[1] = int(count_label.removeprefix("count_"))

        sides: dict[str, SideData] = {}
        for side_name, original_side in (("left", sample.left), ("right", sample.right)):
            roster = [
                RosterEntry(enemy_id=values[0], count=values[1], confidence=1.0)
                for _, values in sorted(roster_slots[side_name].items())
                if values[0] > 0 and values[1] > 0
            ]
            if imported_boxes is None:
                units = original_side.units
            else:
                units = [
                    UnitDetection(
                        enemy_id=_enemy_id(box.class_name),
                        x=(box.bbox.x1 + box.bbox.x2) / 2,
                        y=box.bbox.y2,
                        bbox=box.bbox,
                        confidence=1.0,
                        source=AnnotationSource.MANUAL,
                    )
                    for box in imported_boxes
                    if box.side == side_name
                ]
            sides[side_name] = SideData(roster=roster, units=units)

        complete = (
            not unresolved_roster
            and sample.winner is not None
            and bool(sides["left"].roster)
            and bool(sides["right"].roster)
            and sides["left"].roster_counts() == sides["left"].unit_counts()
            and sides["right"].roster_counts() == sides["right"].unit_counts()
        )
        payload = sample.model_dump(mode="json")
        payload["left"] = sides["left"].model_dump(mode="json")
        payload["right"] = sides["right"].model_dump(mode="json")
        payload["review_status"] = ReviewStatus.ACCEPTED.value if complete else ReviewStatus.PENDING.value
        if complete:
            payload["failure_reasons"] = []
            payload["winner_confidence"] = 1.0
        else:
            payload["failure_reasons"] = sorted(
                set([*sample.failure_reasons, "platform_annotations_incomplete_or_inconsistent"])
            )
        corrected = RoundSample.model_validate(payload)
        corrections.append(
            ReviewCorrection(sample=corrected, note=f"Imported from Platform annotation version {version}")
        )
    store.save_many(corrections)


def annotation_enemy_id(class_name: str) -> int:
    return _enemy_id(class_name)
