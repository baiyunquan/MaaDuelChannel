from __future__ import annotations

import re

import cv2
import numpy as np

from maa_duel.video.phases import FrameSignals
from maa_duel.vision.ocr import OcrEngine, parse_countdown, parse_round_number


class OcrPhaseAnalyzer:
    """Extract timeline signals from the stable center and top-center UI regions."""

    def __init__(
        self,
        ocr: OcrEngine,
        *,
        minimum_ocr_confidence: float = 0.5,
        game_signal_grace_seconds: float = 2.0,
    ) -> None:
        self.ocr = ocr
        self.minimum_ocr_confidence = minimum_ocr_confidence
        self.game_signal_grace_seconds = game_signal_grace_seconds
        self._last_direct_game_signal: float | None = None

    def analyze(self, frame: np.ndarray, timestamp: float) -> FrameSignals:
        height, width = frame.shape[:2]
        center = frame[int(height * 0.32) : int(height * 0.66), int(width * 0.32) : int(width * 0.68)]
        center_text = [
            item.text
            for item in self.ocr.recognize(center, detect=False)
            if item.confidence >= self.minimum_ocr_confidence
        ]
        countdown = next((value for text in center_text if (value := parse_countdown(text)) is not None), None)
        round_number = next((value for text in center_text if (value := parse_round_number(text)) is not None), None)

        has_player_anchor = False
        if countdown is None and round_number is None:
            top = frame[0 : int(height * 0.14), int(width * 0.38) : int(width * 0.62)]
            top_text = [
                item.text
                for item in self.ocr.recognize(top, detect=False)
                if item.confidence >= self.minimum_ocr_confidence
            ]
            has_player_anchor = re.search(r"\d\s*/\s*\d", "".join(top_text)) is not None

        direct_game_signal = countdown is not None or round_number is not None or has_player_anchor
        if direct_game_signal:
            self._last_direct_game_signal = timestamp
        in_grace_window = (
            self._last_direct_game_signal is not None
            and 0.0 <= timestamp - self._last_direct_game_signal <= self.game_signal_grace_seconds
        )
        game_visible = direct_game_signal or in_grace_window
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
