from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Protocol

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


def default_slot_specs() -> list[SlotSpec]:
    specs: list[SlotSpec] = []
    for side, centers in (
        ("left", (0.400, 0.345, 0.290)),
        ("right", (0.600, 0.655, 0.710)),
    ):
        for index, center_x in enumerate(centers):
            specs.append(
                SlotSpec(
                    side=side,
                    index=index,
                    icon_rect=_rect_around(center_x, 0.925, 0.070, 0.105),
                    count_rect=(center_x + 0.015, 0.925, center_x + 0.055, 0.975),
                )
            )
    return specs


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
    ) -> None:
        self.classifier = classifier
        self.ocr = ocr
        self.slots = slots or default_slot_specs()
        self.count_classifier = count_classifier
        self.minimum_type_confidence = minimum_type_confidence
        self.minimum_count_confidence = minimum_count_confidence

    def recognize(self, frame: np.ndarray) -> list[RosterObservation]:
        return self.recognize_batch([frame])[0]

    def recognize_batch(self, frames: list[np.ndarray]) -> list[list[RosterObservation]]:
        if not frames:
            return []
        icon_crops = [crop_normalized(frame, slot.icon_rect) for frame in frames for slot in self.slots]
        classify_batch = getattr(self.classifier, "classify_batch", None)
        classifications = (
            classify_batch(icon_crops)
            if callable(classify_batch)
            else [self.classifier.classify(crop) for crop in icon_crops]
        )
        count_classifications = (
            self.count_classifier.classify_batch(
                [crop_normalized(frame, slot.count_rect) for frame in frames for slot in self.slots]
            )
            if self.count_classifier is not None
            else None
        )
        output: list[list[RosterObservation]] = []
        offset = 0
        for frame in frames:
            observations: list[RosterObservation] = []
            frame_classifications = classifications[offset : offset + len(self.slots)]
            offset += len(self.slots)
            for slot_position, (slot, classification) in enumerate(zip(self.slots, frame_classifications, strict=True)):
                if classification.enemy_id < 1 or classification.confidence < self.minimum_type_confidence:
                    continue
                if count_classifications is not None:
                    count_result = count_classifications[offset - len(self.slots) + slot_position]
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
                for candidate in candidates:
                    if candidate.confidence < self.minimum_count_confidence:
                        continue
                    try:
                        count = parse_count(candidate.text)
                    except ValueError:
                        continue
                    observations.append(
                        RosterObservation(
                            side=slot.side,
                            slot=slot.index,
                            enemy_id=classification.enemy_id,
                            count=count,
                            type_confidence=classification.confidence,
                            count_confidence=candidate.confidence,
                        )
                    )
                    break
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
            count_votes[value.count] += value.count_confidence
        enemy_id = max(type_votes, key=type_votes.get)
        count = max(count_votes, key=count_votes.get)
        type_agreement = type_votes[enemy_id] / sum(type_votes.values())
        count_agreement = count_votes[count] / sum(count_votes.values())
        selected_type_confidences = [value.type_confidence for value in values if value.enemy_id == enemy_id]
        selected_count_confidences = [value.count_confidence for value in values if value.count == count]
        confidence = min(
            type_agreement,
            count_agreement,
            sum(selected_type_confidences) / len(selected_type_confidences),
            sum(selected_count_confidences) / len(selected_count_confidences),
        )
        if confidence >= minimum_confidence:
            by_side[side].append(RosterEntry(enemy_id=enemy_id, count=count, confidence=min(confidence, 1.0)))
    return by_side
