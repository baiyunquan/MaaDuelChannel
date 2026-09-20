from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from maa_duel.schema import BoundingBox
from maa_duel.vision.layout import RawDetection
from maa_duel.vision.roster import Classification


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
    return {int(key): int(value) for key, value in payload.items()}


class YoloPortraitClassifier:
    def __init__(self, model_path: Path, class_map_path: Path) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed; install the vision extra") from exc
        self.model = YOLO(str(model_path))
        self.class_map = _load_class_map(class_map_path)

    def classify(self, image: np.ndarray) -> Classification:
        results = self.model.predict(source=image, verbose=False)
        if not results:
            return Classification(enemy_id=0, confidence=0.0)
        return classification_from_result(results[0], self.class_map)


class YoloBattlefieldDetector:
    def __init__(self, model_path: Path, class_map_path: Path, *, confidence: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed; install the vision extra") from exc
        self.model = YOLO(str(model_path))
        self.class_map = _load_class_map(class_map_path)
        self.confidence = confidence

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        results = self.model.predict(source=image, conf=self.confidence, verbose=False)
        return detections_from_result(results[0], self.class_map) if results else []


def train_vision_model(
    task: str,
    workspace: Path,
    *,
    all_samples: bool,
    epochs: int = 100,
    image_size: int = 640,
    device: str = "0",
    base_model: str | None = None,
) -> Path:
    if not all_samples:
        raise ValueError("vision training requires --all because this project does not create a validation split")
    if task not in {"roster", "battlefield"}:
        raise ValueError("task must be roster or battlefield")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed; install the vision extra") from exc

    if base_model is None:
        base_dir = workspace / "models" / "base"
        base_dir.mkdir(parents=True, exist_ok=True)
        model_name = str(base_dir / ("yolo11n-cls.pt" if task == "roster" else "yolo11n.pt"))
    else:
        model_name = base_model
    data = (
        workspace / "synthetic" / "roster"
        if task == "roster"
        else workspace / "synthetic" / "battlefield" / "dataset.yaml"
    )
    if not data.exists():
        raise FileNotFoundError(f"synthetic dataset is missing: {data}")
    project = workspace / "models" / "vision"
    run_name = task
    model = YOLO(model_name)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=image_size,
        device=device,
        val=False,
        project=str(project),
        name=run_name,
        exist_ok=True,
    )
    return project / run_name / "weights" / "best.pt"
