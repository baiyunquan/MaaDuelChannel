from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from maa_duel.schema import Winner
from maa_duel.video.evidence import TimedFrame
from maa_duel.vision.layout import RawDetection


class BattlefieldDetector(Protocol):
    def detect(
        self,
        frame: np.ndarray,
        *,
        candidate_enemy_ids: Iterable[int] | None = None,
    ) -> list[RawDetection]: ...


@dataclass(frozen=True)
class SurvivorWinnerResult:
    winner: Winner | None
    confidence: float | None
    evidence_time: float
    evidence_frame: np.ndarray
    left_survivors: int = 0
    right_survivors: int = 0
    details: str = ""


@dataclass(frozen=True)
class _EvaluatedFrame:
    source: TimedFrame
    left_count: int
    right_count: int

    @property
    def supported_winner(self) -> Winner | None:
        if self.left_count > 0 and self.right_count == 0:
            return Winner.LEFT
        if self.right_count > 0 and self.left_count == 0:
            return Winner.RIGHT
        return None


class BattlefieldSurvivorDetector:
    """Resolve a winner from a phase-validated, chronological end-frame run."""

    def __init__(
        self,
        battlefield_detector: BattlefieldDetector,
        *,
        minimum_confidence: float = 0.25,
        battlefield_roi: tuple[float, float, float, float] = (0.15, 0.08, 0.85, 0.92),
        majority_window: int = 5,
        minimum_supporting_frames: int = 3,
    ) -> None:
        self.battlefield_detector = battlefield_detector
        self.minimum_confidence = minimum_confidence
        self.battlefield_roi = battlefield_roi
        self.majority_window = majority_window
        self.minimum_supporting_frames = minimum_supporting_frames

    def detect_winner(
        self,
        *,
        end_run: Sequence[TimedFrame],
        left_eids: set[int],
        right_eids: set[int],
    ) -> SurvivorWinnerResult:
        if not end_run:
            raise ValueError("validated end frames must not be empty")
        ordered = sorted(end_run, key=lambda item: (item.timestamp, item.frame_index))
        window = ordered[-self.majority_window :]
        detections = self._detect_frames(window, (left_eids | right_eids) - {0})
        evaluated = [
            self._evaluate_frame(frame, frame_detections, left_eids=left_eids, right_eids=right_eids)
            for frame, frame_detections in zip(window, detections, strict=True)
        ]

        left_support = [item for item in evaluated if item.supported_winner is Winner.LEFT]
        right_support = [item for item in evaluated if item.supported_winner is Winner.RIGHT]
        required = self.minimum_supporting_frames
        if len(left_support) >= required and not right_support:
            return self._resolved(Winner.LEFT, left_support[-1], len(left_support), len(evaluated))
        if len(right_support) >= required and not left_support:
            return self._resolved(Winner.RIGHT, right_support[-1], len(right_support), len(evaluated))

        latest = evaluated[-1]
        return SurvivorWinnerResult(
            winner=None,
            confidence=None,
            evidence_time=latest.source.timestamp,
            evidence_frame=latest.source.frame,
            left_survivors=latest.left_count,
            right_survivors=latest.right_count,
            details="unresolved_end_game",
        )

    def _detect_frames(self, frames: list[TimedFrame], candidate_eids: set[int]) -> list[list[RawDetection]]:
        images = [item.frame for item in frames]
        batch_detect = getattr(self.battlefield_detector, "detect_batch", None)
        if callable(batch_detect):
            try:
                result = batch_detect(images, candidate_enemy_ids=candidate_eids or None)
            except TypeError:
                result = batch_detect(images)
            if len(result) != len(images):
                raise RuntimeError("battlefield batch detector returned an unexpected number of rows")
            return list(result)

        result: list[list[RawDetection]] = []
        for image in images:
            try:
                result.append(
                    self.battlefield_detector.detect(image, candidate_enemy_ids=candidate_eids or None)
                )
            except TypeError:
                result.append(self.battlefield_detector.detect(image))
        return result

    def _evaluate_frame(
        self,
        source: TimedFrame,
        detections: list[RawDetection],
        *,
        left_eids: set[int],
        right_eids: set[int],
    ) -> _EvaluatedFrame:
        x1, y1, x2, y2 = self.battlefield_roi
        filtered = [
            detection
            for detection in detections
            if detection.confidence >= self.minimum_confidence
            and x1 <= (detection.bbox.x1 + detection.bbox.x2) / 2.0 <= x2
            and y1 <= (detection.bbox.y1 + detection.bbox.y2) / 2.0 <= y2
        ]
        return _EvaluatedFrame(
            source=source,
            left_count=sum(item.enemy_id in left_eids for item in filtered),
            right_count=sum(item.enemy_id in right_eids for item in filtered),
        )

    @staticmethod
    def _resolved(
        winner: Winner,
        evidence: _EvaluatedFrame,
        supporting: int,
        total: int,
    ) -> SurvivorWinnerResult:
        return SurvivorWinnerResult(
            winner=winner,
            confidence=min(0.95, 0.75 + 0.20 * supporting / max(1, total)),
            evidence_time=evidence.source.timestamp,
            evidence_frame=evidence.source.frame,
            left_survivors=evidence.left_count,
            right_survivors=evidence.right_count,
            details=f"{winner.value}_survivor_majority:{supporting}/{total}",
        )
