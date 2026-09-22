from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from maa_duel.contracts import ModelVersion, git_commit, sha256_file, write_contract
from maa_duel.schema import BoundingBox
from maa_duel.vision.layout import RawDetection
from maa_duel.vision.roster import Classification, CountClassification


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def classification_from_result(result: Any, class_map: dict[int, int]) -> Classification:
    class_index = int(result.probs.top1)
    confidence_value = result.probs.top1conf
    confidence = float(confidence_value.item() if hasattr(confidence_value, "item") else confidence_value)
    return Classification(enemy_id=class_map[class_index], confidence=confidence)


def detections_from_result(result: Any, class_map: dict[int, int]) -> list[RawDetection]:
    if result.boxes is None:
        return []
    classes = _array(result.boxes.cls).astype(int)
    confidences = _array(result.boxes.conf).astype(float)
    boxes = _array(result.boxes.xyxyn).astype(float)
    return [
        RawDetection(
            enemy_id=class_map[int(class_index)],
            bbox=BoundingBox(x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3])),
            confidence=float(confidence),
        )
        for class_index, confidence, box in zip(classes, confidences, boxes, strict=True)
    ]


def _load_class_map(path: Path) -> dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(key): int(value) if isinstance(value, int) or str(value).isdigit() else 0 for key, value in payload.items()
    }


def _load_label_map(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in payload.items()}


class YoloPortraitClassifier:
    def __init__(
        self,
        model_path: Path,
        class_map_path: Path,
        *,
        device: str = "0",
        half: bool = True,
        batch_size: int = 32,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed; install the vision extra") from exc
        self.model = YOLO(str(model_path))
        self.class_map = _load_class_map(class_map_path)
        self.device = device
        self.half = half and device != "cpu"
        self.batch_size = batch_size

    def classify(self, image: np.ndarray) -> Classification:
        return self.classify_batch([image])[0]

    def classify_batch(self, images: list[np.ndarray]) -> list[Classification]:
        if not images:
            return []
        results = self.model.predict(
            source=images,
            device=self.device,
            half=self.half,
            batch=self.batch_size,
            verbose=False,
        )
        return [classification_from_result(result, self.class_map) for result in results]


class YoloBattlefieldDetector:
    def __init__(
        self,
        model_path: Path,
        class_map_path: Path,
        *,
        confidence: float = 0.25,
        device: str = "0",
        half: bool = True,
        batch_size: int = 16,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed; install the vision extra") from exc
        self.model = YOLO(str(model_path))
        self.class_map = _load_class_map(class_map_path)
        self.confidence = confidence
        self.device = device
        self.half = half and device != "cpu"
        self.batch_size = batch_size

    def detect(
        self,
        image: np.ndarray,
        *,
        candidate_enemy_ids: Iterable[int] | None = None,
    ) -> list[RawDetection]:
        return self.detect_batch([image], candidate_enemy_ids=candidate_enemy_ids)[0]

    def detect_batch(
        self,
        images: list[np.ndarray],
        *,
        candidate_enemy_ids: Iterable[int] | None = None,
    ) -> list[list[RawDetection]]:
        if not images:
            return []

        classes_filter: list[int] | None = None
        if candidate_enemy_ids is not None:
            classes_filter = [
                idx for idx, eid in self.class_map.items() if eid in candidate_enemy_ids
            ]
            if not classes_filter:
                return [[] for _ in images]

        kwargs: dict[str, Any] = {
            "source": images,
            "conf": self.confidence,
            "device": self.device,
            "half": self.half,
            "batch": self.batch_size,
            "verbose": False,
        }
        if classes_filter is not None:
            kwargs["classes"] = classes_filter

        results = self.model.predict(**kwargs)
        return [detections_from_result(result, self.class_map) for result in results]


class YoloCountClassifier:
    def __init__(
        self,
        model_path: Path,
        class_map_path: Path,
        *,
        device: str = "0",
        half: bool = True,
        batch_size: int = 32,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed; install the vision extra") from exc
        self.model = YOLO(str(model_path))
        self.class_map = _load_label_map(class_map_path)
        self.device = device
        self.half = half and device != "cpu"
        self.batch_size = batch_size

    def classify_batch(self, images: list[np.ndarray]) -> list[CountClassification]:
        if not images:
            return []
        results = self.model.predict(
            source=images,
            device=self.device,
            half=self.half,
            batch=self.batch_size,
            verbose=False,
        )
        output = []
        for result in results:
            class_index = int(result.probs.top1)
            confidence_value = result.probs.top1conf
            confidence = float(confidence_value.item() if hasattr(confidence_value, "item") else confidence_value)
            label = self.class_map[class_index]
            count = int(label.removeprefix("count_")) if label.startswith("count_") else None
            output.append(CountClassification(count=count, confidence=confidence))
        return output


def train_vision_model(
    task: str,
    workspace: Path,
    *,
    all_samples: bool,
    epochs: int = 100,
    image_size: int | None = None,
    device: str = "0",
    base_model: str | None = None,
    dataset_version: str | None = None,
    batch: int | float = -1,
    workers: int = 4,
    cache: str | bool = "disk",
    amp: bool = True,
    deterministic: bool = True,
    seed: int = 20260920,
    patience: int = 50,
) -> Path:
    if not all_samples:
        raise ValueError("vision training requires --all because this project does not create a validation split")
    if task not in {"roster", "battlefield", "ocr"}:
        raise ValueError("task must be roster, battlefield, or ocr")

    if image_size is None or image_size <= 0:
        image_size = 128 if task in {"roster", "ocr"} else 640

    if batch == -1 or batch <= 0:
        batch = 64 if task in {"roster", "ocr"} else 16

    workers = min(workers, 4)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed; install the vision extra") from exc

    if base_model is None:
        base_dir = workspace / "models" / "base"
        base_dir.mkdir(parents=True, exist_ok=True)
        model_name = str(base_dir / ("yolo11n-cls.pt" if task in {"roster", "ocr"} else "yolo11n.pt"))
    else:
        model_name = base_model
    if dataset_version:
        task_directory = {
            "roster": "roster_classification",
            "battlefield": "battlefield_detection",
            "ocr": "ocr_classification",
        }[task]
        task_root = workspace / "datasets" / dataset_version / task_directory
        data = task_root / "data.yaml" if task == "battlefield" else task_root
    elif task == "roster":
        data = workspace / "synthetic" / "roster"
    elif task == "battlefield":
        data = workspace / "synthetic" / "battlefield" / "dataset.yaml"
    else:
        raise ValueError("OCR training requires --dataset-version from reviewed Platform annotations")
    if not data.exists():
        raise FileNotFoundError(f"synthetic dataset is missing: {data}")
    model_version = f"{task}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    task_project = workspace / "models" / "vision" / task
    run_name = model_version
    model = YOLO(model_name)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=image_size,
        device=device,
        val=False,
        project=str(task_project),
        name=run_name,
        exist_ok=True,
        batch=batch,
        workers=workers,
        cache=cache,
        amp=amp,
        deterministic=deterministic,
        seed=seed,
        patience=patience,
        close_mosaic=0 if sys.platform == "win32" or task in {"roster", "ocr"} else 10,
    )
    weight_root = task_project / run_name / "weights"
    checkpoint = weight_root / "best.pt"
    if not checkpoint.is_file():
        checkpoint = weight_root / "last.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"Ultralytics did not produce a checkpoint under {weight_root}")
    contract = ModelVersion(
        model_version=model_version,
        task=task,
        dataset_version=dataset_version,
        base_model=model_name,
        checkpoint=checkpoint.relative_to(workspace).as_posix(),
        checkpoint_sha256=sha256_file(checkpoint),
        device=device,
        precision="fp16" if amp and device != "cpu" else "fp32",
        training_args={
            "epochs": epochs,
            "image_size": image_size,
            "batch": batch,
            "workers": workers,
            "cache": cache,
            "amp": amp,
            "deterministic": deterministic,
            "seed": seed,
            "patience": patience,
            "all_samples": all_samples,
        },
        git_commit=git_commit(),
    )
    write_contract(task_project / run_name / "model.json", contract)
    current_weights = task_project / "weights"
    current_weights.mkdir(parents=True, exist_ok=True)
    if checkpoint.is_file():
        shutil.copy2(checkpoint, current_weights / "best.pt")
    source_class_map = data.parent / "class-map.json" if task == "battlefield" else data / "class-map.json"
    if source_class_map.is_file():
        shutil.copy2(source_class_map, task_project / "class-map.json")
    write_contract(task_project / "model.json", contract)
    return checkpoint
