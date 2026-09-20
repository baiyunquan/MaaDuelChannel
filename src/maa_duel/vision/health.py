from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from maa_duel.schema import Winner


@dataclass(frozen=True)
class HealthCounts:
    orange: int
    blue: int


class HealthBarDetector:
    def __init__(
        self,
        *,
        minimum_width: int = 8,
        minimum_width_ratio: float = 0.01,
        minimum_height: int = 4,
        maximum_width_ratio: float = 0.25,
        maximum_height_ratio: float = 0.04,
        minimum_aspect: float = 4.0,
        battlefield_roi: tuple[float, float, float, float] = (0.20, 0.08, 0.80, 0.95),
    ) -> None:
        self.minimum_width = minimum_width
        self.minimum_width_ratio = minimum_width_ratio
        self.minimum_height = minimum_height
        self.maximum_width_ratio = maximum_width_ratio
        self.maximum_height_ratio = maximum_height_ratio
        self.minimum_aspect = minimum_aspect
        self.battlefield_roi = battlefield_roi

    def count(self, image: np.ndarray) -> HealthCounts:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(hsv, np.array((0, 100, 100)), np.array((30, 255, 255)))
        blue = cv2.inRange(hsv, np.array((90, 100, 150)), np.array((115, 255, 255)))
        return HealthCounts(
            orange=self._count_mask(orange, image.shape[1], image.shape[0]),
            blue=self._count_mask(blue, image.shape[1], image.shape[0]),
        )

    def _count_mask(self, mask: np.ndarray, frame_width: int, frame_height: int) -> int:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        minimum_width = max(self.minimum_width, int(frame_width * self.minimum_width_ratio))
        maximum_width = int(frame_width * self.maximum_width_ratio)
        maximum_height = max(2, int(frame_height * self.maximum_height_ratio))
        roi_x1, roi_y1, roi_x2, roi_y2 = self.battlefield_roi
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            center_x = (x + width / 2) / frame_width
            center_y = (y + height / 2) / frame_height
            if (
                minimum_width <= width <= maximum_width
                and self.minimum_height <= height <= maximum_height
                and width / height >= self.minimum_aspect
                and roi_x1 <= center_x <= roi_x2
                and roi_y1 <= center_y <= roi_y2
            ):
                count += 1
        return count


class WinnerTracker:
    def __init__(self, *, stable_frames: int = 5) -> None:
        if stable_frames < 2:
            raise ValueError("stable_frames must be at least 2")
        self.stable_frames = stable_frames
        self._seen_orange = False
        self._seen_blue = False
        self._candidate: Winner | None = None
        self._candidate_frames = 0

    @property
    def confidence(self) -> float:
        return min(1.0, self._candidate_frames / self.stable_frames)

    def update(self, *, orange: int, blue: int) -> Winner | None:
        self._seen_orange = self._seen_orange or orange > 0
        self._seen_blue = self._seen_blue or blue > 0
        if not (self._seen_orange and self._seen_blue):
            self._reset_candidate()
            return None

        candidate: Winner | None = None
        if orange > 0 and blue == 0:
            candidate = Winner.LEFT
        elif blue > 0 and orange == 0:
            candidate = Winner.RIGHT

        if candidate is None:
            self._reset_candidate()
            return None
        if candidate is self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = candidate
            self._candidate_frames = 1
        return candidate if self._candidate_frames >= self.stable_frames else None

    def _reset_candidate(self) -> None:
        self._candidate = None
        self._candidate_frames = 0
