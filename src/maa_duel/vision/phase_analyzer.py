from __future__ import annotations

import re

import cv2
import numpy as np

from maa_duel.video.phases import FrameSignals
from maa_duel.vision.ocr import OcrEngine, parse_countdown, parse_round_number


class OcrPhaseAnalyzer:
    """Extract timeline signals from the stable center and top-center UI regions."""

    def __init__(self, ocr: OcrEngine, *, minimum_ocr_confidence: float = 0.5) -> None:
        self.ocr = ocr
        self.minimum_ocr_confidence = minimum_ocr_confidence

    def analyze(self, frame: np.ndarray, timestamp: float) -> FrameSignals:
        height, width = frame.shape[:2]
        center = frame[int(height * 0.25) : int(height * 0.70), int(width * 0.25) : int(width * 0.75)]
        top = frame[0 : int(height * 0.16), int(width * 0.35) : int(width * 0.65)]
        center_text = [
            item.text for item in self.ocr.recognize(center, detect=True) if item.confidence >= self.minimum_ocr_confidence
        ]
        top_text = [
            item.text for item in self.ocr.recognize(top, detect=True) if item.confidence >= self.minimum_ocr_confidence
        ]

        countdown = next((value for text in center_text if (value := parse_countdown(text)) is not None), None)
        round_number = next((value for text in center_text if (value := parse_round_number(text)) is not None), None)
        has_player_anchor = any(re.search(r"\d\s*/\s*\d", text) for text in top_text)
        game_visible = countdown is not None or round_number is not None or has_player_anchor
        layout_score = 0.0
        if game_visible and countdown is None and round_number is None:
            gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
            layout_score = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1000.0)

        return FrameSignals(
            timestamp=timestamp,
            game_visible=game_visible,
            countdown_seconds=countdown,
            round_number=round_number,
            layout_score=layout_score,
        )
