from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from maa_duel.video.phases import FrameSignals
from maa_duel.video.viewport import detect_viewport
from maa_duel.vision.ocr import OcrEngine, parse_countdown, parse_round_number

NormalizedRect = tuple[float, float, float, float]


@dataclass(frozen=True)
class _PhaseRois:
    left_rail: NormalizedRect
    right_rail: NormalizedRect
    bottom_panel: NormalizedRect
    left_choice: NormalizedRect
    right_choice: NormalizedRect
    center: NormalizedRect
    corners: tuple[NormalizedRect, NormalizedRect, NormalizedRect, NormalizedRect]


_HD_ROIS = _PhaseRois(
    left_rail=(0.04, 0.06, 0.25, 0.92),
    right_rail=(0.70, 0.06, 0.91, 0.92),
    bottom_panel=(0.18, 0.70, 0.82, 0.97),
    left_choice=(0.00, 0.70, 0.20, 0.94),
    right_choice=(0.80, 0.70, 1.00, 0.94),
    center=(0.29, 0.20, 0.71, 0.62),
    corners=(
        (0.00, 0.00, 0.14, 0.18),
        (0.86, 0.00, 1.00, 0.18),
        (0.00, 0.82, 0.14, 1.00),
        (0.86, 0.82, 1.00, 1.00),
    ),
)

_WIDE_ROIS = _PhaseRois(
    left_rail=(0.08, 0.06, 0.27, 0.92),
    right_rail=(0.63, 0.06, 0.82, 0.92),
    bottom_panel=(0.18, 0.70, 0.82, 0.97),
    left_choice=(0.00, 0.70, 0.20, 0.94),
    right_choice=(0.80, 0.70, 1.00, 0.94),
    center=(0.29, 0.20, 0.71, 0.62),
    corners=(
        (0.00, 0.00, 0.14, 0.18),
        (0.76, 0.00, 0.90, 0.18),
        (0.00, 0.82, 0.14, 1.00),
        (0.76, 0.82, 0.90, 1.00),
    ),
)


class OcrPhaseAnalyzer:
    """Produce independent visual phase scores for one frame.

    Temporal smoothing belongs to :class:`RoundSegmenter`; this class deliberately
    keeps no history so native-rate refinement may analyze frames in any order.
    """

    def __init__(self, ocr: OcrEngine, *, minimum_ocr_confidence: float = 0.5) -> None:
        self.ocr = ocr
        self.minimum_ocr_confidence = minimum_ocr_confidence

    def analyze(self, frame: np.ndarray, timestamp: float) -> FrameSignals:
        return self._analyze(frame, timestamp, include_corner_signature=True)

    def analyze_coarse(self, frame: np.ndarray, timestamp: float) -> FrameSignals:
        return self._analyze(frame, timestamp, include_corner_signature=False)

    def _analyze(
        self,
        frame: np.ndarray,
        timestamp: float,
        *,
        include_corner_signature: bool,
    ) -> FrameSignals:
        if frame is None or frame.size == 0 or len(frame.shape) < 2:
            raise ValueError("phase analyzer requires a non-empty image")
        height, width = frame.shape[:2]
        profile = detect_viewport(width, height).profile
        rois = _WIDE_ROIS if profile == "wide" else _HD_ROIS

        if width > 960:
            scale = 960.0 / width
            visual = cv2.resize(
                frame,
                (960, max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            visual = frame
        hsv = cv2.cvtColor(visual, cv2.COLOR_BGR2HSV)
        left_choice = _crop(hsv, rois.left_choice)
        right_choice = _crop(hsv, rois.right_choice)
        center_ocr = _crop(frame, rois.center)

        battlefield_score = _battlefield_score(hsv, profile)

        bottom_panel_score = _bottom_panel_score(hsv)
        choice_buttons_score = min(
            1.0,
            min(_color_fraction(left_choice, "yellow"), _color_fraction(right_choice, "yellow")) / 0.08,
        )
        countdown_score = _countdown_score(hsv)
        round_banner_score = _round_banner_score(hsv)
        corner_mask_score = _corner_mask_score(visual, rois.corners)
        corner_signature = _corner_signature(visual, rois.corners) if include_corner_signature else ()

        ocr_performed = battlefield_score >= 0.30 and (
            (bottom_panel_score >= 0.25 and choice_buttons_score >= 0.20)
            or (bottom_panel_score >= 0.80 and choice_buttons_score < 0.05)
            or countdown_score >= 0.20
            or round_banner_score >= 0.20
        )
        text = (
            [
                item.text
                for item in self.ocr.recognize(center_ocr, detect=True)
                if item.confidence >= self.minimum_ocr_confidence
            ]
            if ocr_performed
            else []
        )
        countdown_seconds = next(
            (value for value_text in text if (value := parse_countdown(value_text)) is not None),
            None,
        )
        round_number = next(
            (value for value_text in text if (value := parse_round_number(value_text)) is not None),
            None,
        )
        if countdown_seconds is not None:
            countdown_score = 1.0
        if round_number is not None:
            round_banner_score = 1.0

        return FrameSignals(
            timestamp=timestamp,
            battlefield_score=battlefield_score,
            bottom_panel_score=bottom_panel_score,
            choice_buttons_score=choice_buttons_score,
            countdown_score=countdown_score,
            round_banner_score=round_banner_score,
            corner_mask_score=corner_mask_score,
            corner_signature=corner_signature,
            countdown_seconds=countdown_seconds,
            round_number=round_number,
            ocr_performed=ocr_performed,
        )


def _crop(frame: np.ndarray, rect: NormalizedRect) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = rect
    left, top = int(round(x1 * width)), int(round(y1 * height))
    right, bottom = int(round(x2 * width)), int(round(y2 * height))
    return frame[max(0, top) : min(height, bottom), max(0, left) : min(width, right)]


def _hsv_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    if hsv.size == 0:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    if color == "red":
        low = cv2.inRange(hsv, np.array((0, 90, 70)), np.array((12, 255, 255)))
        high = cv2.inRange(hsv, np.array((165, 90, 70)), np.array((179, 255, 255)))
        return cv2.bitwise_or(low, high)
    if color == "cyan":
        return cv2.inRange(hsv, np.array((78, 70, 70)), np.array((108, 255, 255)))
    if color == "yellow":
        return cv2.inRange(hsv, np.array((17, 90, 90)), np.array((40, 255, 255)))
    raise ValueError(f"unsupported phase color: {color}")


def _color_fraction(hsv: np.ndarray, color: str) -> float:
    if hsv.size == 0:
        return 0.0
    return float(np.count_nonzero(_hsv_mask(hsv, color))) / float(hsv.shape[0] * hsv.shape[1])


def _battlefield_score(hsv: np.ndarray, profile: str) -> float:
    if profile == "wide":
        left = ((0.205, 0.03), (0.295, 0.03), (0.205, 0.92), (0.085, 0.92))
        right = ((0.705, 0.03), (0.795, 0.03), (0.915, 0.92), (0.780, 0.92))
    else:
        left = ((0.145, 0.03), (0.235, 0.03), (0.135, 0.92), (0.000, 0.92))
        right = ((0.755, 0.03), (0.845, 0.03), (1.000, 0.92), (0.855, 0.92))
    red_fraction = _polygon_color_fraction(hsv, left, "red")
    cyan_fraction = _polygon_color_fraction(hsv, right, "cyan")
    return min(1.0, min(red_fraction, cyan_fraction) / 0.09)


def _polygon_color_fraction(
    hsv: np.ndarray,
    points: tuple[tuple[float, float], ...],
    color: str,
) -> float:
    height, width = hsv.shape[:2]
    polygon = np.array([(round(x * width), round(y * height)) for x, y in points], dtype=np.int32)
    left, top, box_width, box_height = cv2.boundingRect(polygon)
    clipped = hsv[top : top + box_height, left : left + box_width]
    shifted = polygon - np.array((left, top), dtype=np.int32)
    roi_mask = np.zeros(clipped.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [shifted], 255)
    color_mask = _hsv_mask(clipped, color)
    pixels = int(np.count_nonzero(roi_mask))
    return float(np.count_nonzero(cv2.bitwise_and(color_mask, roi_mask))) / max(1, pixels)


def _bottom_panel_score(hsv: np.ndarray) -> float:
    left = _crop(hsv, (0.18, 0.72, 0.43, 0.97))
    right = _crop(hsv, (0.57, 0.72, 0.82, 0.97))
    badge = _crop(hsv, (0.43, 0.70, 0.57, 0.90))

    def neutral_dark_fraction(crop: np.ndarray) -> float:
        return float(np.mean((crop[..., 1] <= 90) & (crop[..., 2] <= 95)))

    min_dark = min(neutral_dark_fraction(left), neutral_dark_fraction(right))
    dark_score = float(np.clip((min_dark - 0.20) / 0.145, 0.0, 1.0))
    badge_fraction = _color_fraction(badge, "red") + _color_fraction(badge, "yellow")
    badge_score = min(1.0, badge_fraction / 0.06)
    return max(dark_score, badge_score)


def _countdown_score(hsv: np.ndarray) -> float:
    height, width = hsv.shape[:2]
    rect = (0.37, 0.20, 0.63, 0.46)
    crop = _crop(hsv, rect)
    mask = _hsv_mask(crop, "red")
    kernel = np.ones((max(1, round(height * 0.003)), max(3, round(width * 0.012))), dtype=np.uint8)
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    x_offset = int(rect[0] * width)
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        center_x = (x_offset + x + box_width / 2.0) / width
        width_ratio = box_width / width
        height_ratio = box_height / height
        area_ratio = cv2.contourArea(contour) / float(width * height)
        rectangularity = cv2.contourArea(contour) / max(1.0, float(box_width * box_height))
        if (
            0.46 <= center_x <= 0.54
            and 0.045 <= width_ratio <= 0.12
            and 0.075 <= height_ratio <= 0.22
            and area_ratio >= 0.0025
            and 0.45 <= rectangularity <= 0.82
        ):
            best = max(
                best,
                min(1.0, area_ratio / 0.0035, width_ratio / 0.05, height_ratio / 0.09),
            )
    return best


def _round_banner_score(hsv: np.ndarray) -> float:
    height, width = hsv.shape[:2]
    center = _crop(hsv, (0.29, 0.25, 0.71, 0.62))
    if center.size == 0:
        return 0.0
    mask = _hsv_mask(center, "yellow")
    kernel = np.ones((max(1, round(height * 0.003)), max(3, round(width * 0.012))), dtype=np.uint8)
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    x_offset = int(round(0.29 * width))
    y_offset = int(round(0.25 * height))
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        area_ratio = contour_area / float(width * height)
        width_ratio = box_width / width
        height_ratio = box_height / height
        rectangularity = contour_area / max(1.0, float(box_width * box_height))
        center_x = (x_offset + x + box_width / 2.0) / width
        center_y = (y_offset + y + box_height / 2.0) / height
        if (
            area_ratio < 0.006
            or not 0.15 <= width_ratio <= 0.36
            or not 0.045 <= height_ratio <= 0.32
            or width_ratio < height_ratio
            or rectangularity < 0.24
            or not 0.40 <= center_x <= 0.60
            or not 0.34 <= center_y <= 0.60
        ):
            continue
        best = max(
            best,
            min(1.0, area_ratio / 0.0025, width_ratio / 0.15, height_ratio / 0.065),
        )
    return best


def _corner_mask_score(frame: np.ndarray, rects: tuple[NormalizedRect, ...]) -> float:
    dark_corners = 0
    darkness_values: list[float] = []
    for rect in rects:
        crop = _crop(frame, rect)
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark_fraction = float(np.mean(gray < 45))
        darkness_values.append(dark_fraction)
        if dark_fraction >= 0.55:
            dark_corners += 1
    if not darkness_values:
        return 0.0
    agreement = dark_corners / len(darkness_values)
    mean_darkness = float(np.mean(darkness_values))
    return min(1.0, max(agreement, mean_darkness))


def _corner_signature(frame: np.ndarray, rects: tuple[NormalizedRect, ...]) -> tuple[float, ...]:
    values: list[float] = []
    for rect in rects:
        crop = _crop(frame, rect)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thumbnail = cv2.resize(blurred, (8, 8), interpolation=cv2.INTER_AREA)
        values.extend(float(value) / 255.0 for value in thumbnail.flat)
    return tuple(values)
