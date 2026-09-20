import numpy as np

from maa_duel.vision.ocr import OcrText
from maa_duel.vision.phase_analyzer import OcrPhaseAnalyzer
from maa_duel.vision.roster import (
    Classification,
    RosterFrameRecognizer,
    SlotSpec,
)


class QueueOcr:
    def __init__(self, batches):
        self.batches = list(batches)

    def recognize(self, image, *, detect=True):
        return self.batches.pop(0)


class ConstantClassifier:
    def __init__(self, enemy_id):
        self.enemy_id = enemy_id

    def classify(self, image):
        return Classification(self.enemy_id, 0.95)


def test_phase_analyzer_parses_countdown_and_game_anchor():
    ocr = QueueOcr(
        [
            [OcrText("00:04", 0.9)],
            [OcrText("5/5", 0.9)],
        ]
    )
    analyzer = OcrPhaseAnalyzer(ocr)

    signal = analyzer.analyze(np.zeros((180, 320, 3), dtype=np.uint8), 1.5)

    assert signal.game_visible
    assert signal.countdown_seconds == 4
    assert signal.round_number is None


def test_phase_analyzer_scores_clear_layout_between_overlays():
    checkerboard = np.indices((180, 320)).sum(axis=0) % 2
    frame = np.repeat((checkerboard * 255).astype(np.uint8)[..., None], 3, axis=2)
    ocr = QueueOcr(
        [
            [],
            [OcrText("5/5", 0.9)],
        ]
    )

    signal = OcrPhaseAnalyzer(ocr).analyze(frame, 2.0)

    assert signal.game_visible
    assert signal.countdown_seconds is None
    assert signal.round_number is None
    assert signal.layout_score > 0.5


def test_roster_recognizer_uses_configured_icon_and_count_slots():
    ocr = QueueOcr([[OcrText("x3", 0.91)], [OcrText("x2", 0.92)]])
    slots = [
        SlotSpec("left", 0, (0.0, 0.0, 0.4, 1.0), (0.0, 0.0, 0.4, 1.0)),
        SlotSpec("right", 0, (0.6, 0.0, 1.0, 1.0), (0.6, 0.0, 1.0, 1.0)),
    ]
    recognizer = RosterFrameRecognizer(ConstantClassifier(7), ocr, slots=slots)

    observations = recognizer.recognize(np.zeros((100, 200, 3), dtype=np.uint8))

    assert [(item.side, item.enemy_id, item.count) for item in observations] == [
        ("left", 7, 3),
        ("right", 7, 2),
    ]
