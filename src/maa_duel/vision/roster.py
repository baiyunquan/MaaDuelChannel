from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np

from maa_duel.schema import RosterEntry
from maa_duel.vision.ocr import OcrEngine, parse_count

SideName = Literal["left", "right"]
NormalizedRect = tuple[float, float, float, float]


@dataclass(frozen=True)
class RosterObservation:
    side: SideName
    slot: int
    enemy_id: int
    count: int
    type_confidence: float
    count_confidence: float


@dataclass(frozen=True)
class Classification:
    enemy_id: int
    confidence: float


@dataclass(frozen=True)
class CountClassification:
    count: int | None
    confidence: float


class PortraitClassifier(Protocol):
    def classify(self, image: np.ndarray) -> Classification: ...


class CountClassifier(Protocol):
    def classify_batch(self, images: list[np.ndarray]) -> list[CountClassification]: ...


@dataclass(frozen=True)
class SlotSpec:
    side: SideName
    index: int
    icon_rect: NormalizedRect
    count_rect: NormalizedRect


def _rect_around(center_x: float, center_y: float, width: float, height: float) -> NormalizedRect:
    return (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


def calculate_safe_zone(width: int, height: int) -> tuple[int, int, int, int]:
    """
    Returns (offset_x, offset_y, safe_width, safe_height) for a 16:9 safe area
    centered inside an image of arbitrary aspect ratio (e.g. 16:9, 20:9, 4:3).
    """
    target_aspect = 16.0 / 9.0
    aspect = width / max(1, height)
    if aspect >= target_aspect:
        safe_height = height
        safe_width = round(height * target_aspect)
        offset_x = (width - safe_width) // 2
        offset_y = 0
    else:
        safe_width = width
        safe_height = round(width / target_aspect)
        offset_x = 0
        offset_y = (height - safe_height) // 2
    return offset_x, offset_y, safe_width, safe_height


def default_slot_specs(image_shape: tuple[int, ...] | None = None) -> list[SlotSpec]:
    """
    Returns the 6 nominal SlotSpecs mapped into the 16:9 safe zone for the given image shape.
    """
    height, width = image_shape[:2] if image_shape is not None else (1080, 1920)
    offset_x, offset_y, safe_width, safe_height = calculate_safe_zone(width, height)
    r = safe_height * 0.050
    cy = offset_y + safe_height * 0.8944
    specs: list[SlotSpec] = []
    for side, centers in (
        ("left", (0.4013, 0.3404, 0.2794)),
        ("right", (0.5977, 0.6544, 0.7112)),
    ):
        for index, center_x_rel in enumerate(centers):
            cx = offset_x + safe_width * center_x_rel
            icon_rect = (
                max(0.0, (cx - r) / width),
                max(0.0, (cy - r) / height),
                min(1.0, (cx + r) / width),
                min(1.0, (cy + r) / height),
            )
            count_rect = (
                max(0.0, (cx + r * 0.25) / width),
                max(0.0, cy / height),
                min(1.0, (cx + r * 1.25) / width),
                min(1.0, (cy + r * 0.95) / height),
            )
            specs.append(
                SlotSpec(
                    side=side,
                    index=index,
                    icon_rect=icon_rect,
                    count_rect=count_rect,
                )
            )
    return specs


def detect_slot_specs(frame: np.ndarray) -> list[SlotSpec]:
    """
    Detects circular slots in the preparation frame using HoughCircles and rigid geometric fitting.
    Falls back gracefully to default_slot_specs(frame.shape[:2]) if detection fails.
    """
    if frame is None or frame.size == 0 or len(frame.shape) < 2:
        return default_slot_specs()
    height, width = frame.shape[:2]
    base_specs = default_slot_specs((height, width))
    offset_x, offset_y, safe_width, safe_height = calculate_safe_zone(width, height)
    expected_r = safe_height * 0.050

    roi_top = max(0, offset_y + int(safe_height * 0.78))
    roi_bottom = min(height, offset_y + int(safe_height * 0.98))
    roi_left = max(0, offset_x + int(safe_width * 0.20))
    roi_right = min(width, offset_x + int(safe_width * 0.80))
    if roi_bottom <= roi_top or roi_right <= roi_left:
        return base_specs

    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    min_r = max(8, int(expected_r * 0.70))
    max_r = int(expected_r * 1.30)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(expected_r * 1.1),
        param1=45,
        param2=26,
        minRadius=min_r,
        maxRadius=max_r,
    )

    matched: dict[int, tuple[float, float, float]] = {}
    if circles is not None:
        for c in circles[0]:
            cx = float(c[0] + roi_left)
            cy = float(c[1] + roi_top)
            r = float(c[2])
            best_i, best_d = None, float("inf")
            for i, spec in enumerate(base_specs):
                nom_x = (spec.icon_rect[0] + spec.icon_rect[2]) * 0.5 * width
                nom_y = (spec.icon_rect[1] + spec.icon_rect[3]) * 0.5 * height
                dist = float(np.hypot(cx - nom_x, cy - nom_y))
                if dist < expected_r * 0.65 and dist < best_d:
                    best_d = dist
                    best_i = i
            if best_i is not None and best_i not in matched:
                matched[best_i] = (cx, cy, r)

    if not matched:
        return base_specs

    diffs_x = [
        matched[i][0] - (base_specs[i].icon_rect[0] + base_specs[i].icon_rect[2]) * 0.5 * width
        for i in matched
    ]
    diffs_y = [
        matched[i][1] - (base_specs[i].icon_rect[1] + base_specs[i].icon_rect[3]) * 0.5 * height
        for i in matched
    ]
    dx = float(np.median(diffs_x))
    dy = float(np.median(diffs_y))

    if abs(dx) > expected_r * 0.6 or abs(dy) > expected_r * 0.6:
        return base_specs

    final_specs: list[SlotSpec] = []
    for i, spec in enumerate(base_specs):
        if i in matched:
            cx, cy, r = matched[i]
        else:
            nom_x = (spec.icon_rect[0] + spec.icon_rect[2]) * 0.5 * width
            nom_y = (spec.icon_rect[1] + spec.icon_rect[3]) * 0.5 * height
            cx, cy, r = nom_x + dx, nom_y + dy, expected_r
        icon_rect = (
            max(0.0, (cx - r) / width),
            max(0.0, (cy - r) / height),
            min(1.0, (cx + r) / width),
            min(1.0, (cy + r) / height),
        )
        count_rect = (
            max(0.0, (cx + r * 0.25) / width),
            max(0.0, cy / height),
            min(1.0, (cx + r * 1.25) / width),
            min(1.0, (cy + r * 0.95) / height),
        )
        final_specs.append(
            SlotSpec(
                side=spec.side,
                index=spec.index,
                icon_rect=icon_rect,
                count_rect=count_rect,
            )
        )
    return final_specs


def make_portrait_mask(size: int = 64) -> np.ndarray:
    """
    Creates an alpha mask for circular portrait slots:
    - 255 inside inscribed circle (radius ~0.44 * size).
    - 0 outside the circle (eliminating team red/blue square corners).
    - 0 at top-left corner (masking out HUD status symbols).
    - 0 at bottom-right corner (masking out 'x1', 'x2' count text).
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(size * 0.44), 255, -1)
    tl = int(size * 0.28)
    mask[:tl, :tl] = 0
    br = int(size * 0.65)
    mask[br:, br:] = 0
    return mask


def make_circle_mask(size: int = 64) -> np.ndarray:
    """
    Creates an alpha mask for the complete inscribed circle (radius ~0.44 * size).
    Used for measuring slot contrast/std to detect empty slots.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(size * 0.44), 255, -1)
    return mask


class TemplateMatchClassifier:
    """
    Classifies cropped portrait icons by normalized template matching against reference thumbnails,
    using circular geometric and occlusion masks to reject colored backgrounds and HUD elements.
    """

    def __init__(
        self,
        portraits_dir: Path | str,
        empty_slot_path: Path | str | None = None,
        *,
        target_size: int = 64,
        empty_threshold: float = 0.45,
        min_contrast: float = 18.0,
        zoom_crop: float = 0.04,
    ) -> None:
        self.portraits_dir = Path(portraits_dir)
        self.empty_slot_path = Path(empty_slot_path) if empty_slot_path else None
        self.target_size = target_size
        self.empty_threshold = empty_threshold
        self.min_contrast = min_contrast
        self.zoom_crop = zoom_crop
        self._mask = make_portrait_mask(target_size)
        self._circle_mask = make_circle_mask(target_size)
        self._templates: dict[int, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        if not self.portraits_dir.is_dir():
            return
        for p in self.portraits_dir.glob("*/thumbnail.png"):
            try:
                enemy_id = int(p.parent.name)
            except ValueError:
                continue
            if enemy_id <= 0:
                continue
            img = cv2.imread(str(p))
            if img is not None:
                h, w = img.shape[:2]
                if self.zoom_crop > 0:
                    dh, dw = int(h * self.zoom_crop), int(w * self.zoom_crop)
                    cropped = img[dh : max(dh + 1, h - dh), dw : max(dw + 1, w - dw)]
                else:
                    cropped = img
                if cropped.size > 0:
                    resized = cv2.resize(cropped, (self.target_size, self.target_size))
                    self._templates[enemy_id] = resized

        if self.empty_slot_path and self.empty_slot_path.is_file():
            empty_img = cv2.imread(str(self.empty_slot_path))
            if empty_img is not None:
                resized = cv2.resize(empty_img, (self.target_size, self.target_size))
                self._templates[0] = resized

    def classify(self, image: np.ndarray) -> Classification:
        if not self._templates or image is None or image.size == 0:
            return Classification(enemy_id=0, confidence=0.0)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        target = cv2.resize(image, (self.target_size, self.target_size))

        # Fast empty-slot rejection: empty circular slots have low pixel variance (std < 18),
        # whereas actual enemy portraits have rich textures (std >= 27).
        if self.min_contrast > 0:
            circle_pixels = target[self._circle_mask > 0]
            if circle_pixels.size > 0:
                contrast = float(np.std(circle_pixels))
                if contrast < self.min_contrast:
                    return Classification(enemy_id=0, confidence=1.0)

        best_id = 0
        best_score = -1.0
        for enemy_id, tmpl in self._templates.items():
            res = cv2.matchTemplate(target, tmpl, cv2.TM_CCOEFF_NORMED, mask=self._mask)
            score = float(res.max())
            if score > best_score:
                best_score = score
                best_id = enemy_id

        if best_id != 0 and best_score < self.empty_threshold:
            return Classification(enemy_id=0, confidence=round(1.0 - max(0.0, best_score), 4))

        return Classification(enemy_id=best_id, confidence=round(max(0.0, best_score), 4))

    def classify_batch(self, images: list[np.ndarray]) -> list[Classification]:
        return [self.classify(img) for img in images]


class DualEngineClassifier:
    """
    Combines YOLO portrait classifier with OpenCV template matching.
    Implements one-vote veto: when models agree, confidence is boosted;
    when they disagree, confidence is penalized to trigger manual review.
    """

    def __init__(
        self,
        yolo_classifier: PortraitClassifier,
        template_classifier: TemplateMatchClassifier,
        *,
        agreement_boost: float = 0.95,
        veto_confidence: float = 0.15,
        template_veto_threshold: float = 0.65,
    ) -> None:
        self.yolo = yolo_classifier
        self.template = template_classifier
        self.agreement_boost = agreement_boost
        self.veto_confidence = veto_confidence
        self.template_veto_threshold = template_veto_threshold

    def classify(self, image: np.ndarray) -> Classification:
        return self.classify_batch([image])[0]

    def classify_batch(self, images: list[np.ndarray]) -> list[Classification]:
        if not images:
            return []
        yolo_batch = getattr(self.yolo, "classify_batch", None)
        yolo_results = yolo_batch(images) if callable(yolo_batch) else [self.yolo.classify(img) for img in images]
        tmpl_results = self.template.classify_batch(images)

        merged: list[Classification] = []
        for yolo_res, tmpl_res in zip(yolo_results, tmpl_results, strict=True):
            if yolo_res.enemy_id == tmpl_res.enemy_id:
                conf = max(yolo_res.confidence, tmpl_res.confidence, self.agreement_boost)
                merged.append(Classification(enemy_id=yolo_res.enemy_id, confidence=round(min(conf, 1.0), 4)))
            elif tmpl_res.confidence >= self.template_veto_threshold:
                conf = min(self.veto_confidence, yolo_res.confidence, tmpl_res.confidence)
                merged.append(Classification(enemy_id=yolo_res.enemy_id, confidence=round(max(0.05, conf), 4)))
            else:
                merged.append(Classification(enemy_id=yolo_res.enemy_id, confidence=round(yolo_res.confidence, 4)))
        return merged


def crop_normalized(frame: np.ndarray, rect: NormalizedRect) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = rect
    left = max(0, min(width - 1, round(x1 * width)))
    top = max(0, min(height - 1, round(y1 * height)))
    right = max(left + 1, min(width, round(x2 * width)))
    bottom = max(top + 1, min(height, round(y2 * height)))
    return frame[top:bottom, left:right]


class RosterFrameRecognizer:
    def __init__(
        self,
        classifier: PortraitClassifier,
        ocr: OcrEngine,
        *,
        slots: list[SlotSpec] | None = None,
        count_classifier: CountClassifier | None = None,
        minimum_type_confidence: float = 0.25,
        minimum_count_confidence: float = 0.25,
        default_count: int | None = None,
    ) -> None:
        self.classifier = classifier
        self.ocr = ocr
        self.fixed_slots = slots
        self.count_classifier = count_classifier
        self.minimum_type_confidence = minimum_type_confidence
        self.minimum_count_confidence = minimum_count_confidence
        self.default_count = default_count

    @property
    def slots(self) -> list[SlotSpec]:
        return self.fixed_slots if self.fixed_slots is not None else default_slot_specs()

    def recognize(self, frame: np.ndarray) -> list[RosterObservation]:
        return self.recognize_batch([frame])[0]

    def recognize_batch(self, frames: list[np.ndarray]) -> list[list[RosterObservation]]:
        if not frames:
            return []

        frame_slots_list = [
            self.fixed_slots if self.fixed_slots is not None else detect_slot_specs(frame)
            for frame in frames
        ]
        icon_crops = [
            crop_normalized(frame, slot.icon_rect)
            for frame, slots in zip(frames, frame_slots_list, strict=True)
            for slot in slots
        ]
        classify_batch = getattr(self.classifier, "classify_batch", None)
        classifications = (
            classify_batch(icon_crops)
            if callable(classify_batch)
            else [self.classifier.classify(crop) for crop in icon_crops]
        )
        count_crops = [
            crop_normalized(frame, slot.count_rect)
            for frame, slots in zip(frames, frame_slots_list, strict=True)
            for slot in slots
        ]
        count_classifications = (
            self.count_classifier.classify_batch(count_crops)
            if self.count_classifier is not None
            else None
        )
        output: list[list[RosterObservation]] = []
        offset = 0
        for frame, slots in zip(frames, frame_slots_list, strict=True):
            observations: list[RosterObservation] = []
            frame_classifications = classifications[offset : offset + len(slots)]
            offset += len(slots)
            for slot_position, (slot, classification) in enumerate(zip(slots, frame_classifications, strict=True)):
                if classification.enemy_id < 1 or classification.confidence < self.minimum_type_confidence:
                    continue
                if count_classifications is not None:
                    count_result = count_classifications[offset - len(slots) + slot_position]
                    if count_result.count is not None and count_result.confidence >= self.minimum_count_confidence:
                        observations.append(
                            RosterObservation(
                                side=slot.side,
                                slot=slot.index,
                                enemy_id=classification.enemy_id,
                                count=count_result.count,
                                type_confidence=classification.confidence,
                                count_confidence=count_result.confidence,
                            )
                        )
                    continue
                candidates = sorted(
                    self.ocr.recognize(crop_normalized(frame, slot.count_rect), detect=False),
                    key=lambda item: item.confidence,
                    reverse=True,
                )
                detected_count: int | None = None
                detected_count_conf: float = 0.0
                for candidate in candidates:
                    if candidate.confidence < self.minimum_count_confidence:
                        continue
                    try:
                        detected_count = parse_count(candidate.text)
                        detected_count_conf = candidate.confidence
                        break
                    except ValueError:
                        continue
                if detected_count is None and self.default_count is not None:
                    detected_count = self.default_count
                    detected_count_conf = 0.0
                if detected_count is None:
                    continue
                observations.append(
                    RosterObservation(
                        side=slot.side,
                        slot=slot.index,
                        enemy_id=classification.enemy_id,
                        count=max(1, detected_count),
                        type_confidence=classification.confidence,
                        count_confidence=detected_count_conf,
                    )
                )
            output.append(observations)
        return output


def fuse_roster_observations(
    observations: list[RosterObservation],
    *,
    minimum_confidence: float = 0.0,
) -> dict[SideName, list[RosterEntry]]:
    grouped: dict[tuple[SideName, int], list[RosterObservation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.side, observation.slot)].append(observation)

    by_side: dict[SideName, list[RosterEntry]] = {"left": [], "right": []}
    for (side, _slot), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        type_votes: dict[int, float] = defaultdict(float)
        count_votes: dict[int, float] = defaultdict(float)
        for value in values:
            type_votes[value.enemy_id] += value.type_confidence
            count_weight = max(0.5, value.count_confidence)
            count_votes[value.count] += count_weight
        enemy_id = max(type_votes, key=type_votes.get)
        count = max(count_votes, key=count_votes.get)
        total_type = sum(type_votes.values())
        total_count = sum(count_votes.values())
        type_agreement = type_votes[enemy_id] / total_type if total_type > 0 else 1.0
        count_agreement = count_votes[count] / total_count if total_count > 0 else 1.0
        selected_type_confidences = [value.type_confidence for value in values if value.enemy_id == enemy_id]
        selected_count_confidences = [max(0.5, value.count_confidence) for value in values if value.count == count]
        confidence = min(
            type_agreement,
            count_agreement,
            sum(selected_type_confidences) / len(selected_type_confidences),
            sum(selected_count_confidences) / len(selected_count_confidences),
        )
        if confidence >= minimum_confidence:
            by_side[side].append(RosterEntry(enemy_id=enemy_id, count=count, confidence=min(confidence, 1.0)))
    return by_side
