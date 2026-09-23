import cv2
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


def phase_frame(width: int, height: int, *, prep=False, countdown=False, round_banner=False, masks=False):
    frame = np.full((height, width, 3), 105, dtype=np.uint8)
    wide = width / height > 2.0
    left_x = (0.12, 0.22) if wide else (0.07, 0.18)
    right_x = (0.68, 0.79) if wide else (0.74, 0.87)
    cv2.rectangle(
        frame,
        (int(width * left_x[0]), int(height * 0.10)),
        (int(width * left_x[1]), int(height * 0.88)),
        (20, 20, 230),
        20,
    )
    cv2.rectangle(
        frame,
        (int(width * right_x[0]), int(height * 0.10)),
        (int(width * right_x[1]), int(height * 0.88)),
        (230, 210, 20),
        20,
    )
    if prep:
        cv2.rectangle(frame, (int(width * 0.20), int(height * 0.82)), (int(width * 0.80), height - 1), (25, 25, 25), -1)
        cv2.rectangle(frame, (0, int(height * 0.82)), (int(width * 0.18), height - 1), (0, 220, 255), -1)
        cv2.rectangle(frame, (int(width * 0.82), int(height * 0.82)), (width - 1, height - 1), (0, 220, 255), -1)
        for x in np.linspace(0.28, 0.72, 5):
            cv2.circle(frame, (int(width * x), int(height * 0.90)), int(height * 0.035), (190, 190, 190), 3)
    if countdown:
        cv2.circle(
            frame,
            (int(width * 0.50), int(height * 0.33)),
            int(height * 0.05),
            (20, 20, 230),
            -1,
            cv2.LINE_AA,
        )
    if round_banner:
        cv2.putText(
            frame,
            "ROUND 04",
            (int(width * 0.33), int(height * 0.54)),
            cv2.FONT_HERSHEY_DUPLEX,
            height / 300,
            (0, 225, 255),
            max(3, height // 120),
            cv2.LINE_AA,
        )
    if masks:
        corner_w, corner_h = int(width * 0.14), int(height * 0.18)
        frame[:corner_h, :corner_w] = 2
        frame[:corner_h, -corner_w:] = 2
        frame[-corner_h:, :corner_w] = 2
        frame[-corner_h:, -corner_w:] = 2
    return frame


def test_phase_analyzer_scores_prep_signals_on_both_viewport_profiles():
    for width, height in ((1920, 1080), (1920, 864)):
        signal = OcrPhaseAnalyzer(QueueOcr([[]])).analyze(
            phase_frame(width, height, prep=True, countdown=True, masks=True),
            1.5,
        )

        assert signal.battlefield_score >= 0.55
        assert signal.bottom_panel_score >= 0.55
        assert signal.choice_buttons_score >= 0.55
        assert signal.countdown_score >= 0.55
        assert signal.round_banner_score < 0.55
        assert signal.corner_mask_score >= 0.55


def test_phase_analyzer_scores_clean_layout_without_prep_or_center_overlay():
    signal = OcrPhaseAnalyzer(QueueOcr([[]])).analyze(phase_frame(1920, 1080, masks=True), 2.0)

    assert signal.battlefield_score >= 0.55
    assert signal.bottom_panel_score < 0.55
    assert signal.choice_buttons_score < 0.55
    assert signal.countdown_score < 0.55
    assert signal.round_banner_score < 0.55
    assert signal.corner_mask_score >= 0.55


def test_phase_analyzer_scores_round_banner_without_ocr_text():
    signal = OcrPhaseAnalyzer(QueueOcr([[]])).analyze(
        phase_frame(1920, 1080, round_banner=True, masks=True),
        2.5,
    )

    assert signal.round_banner_score >= 0.55
    assert signal.round_number is None


def test_phase_analyzer_does_not_treat_center_wait_button_as_paired_choices():
    frame = phase_frame(1920, 1080)
    cv2.rectangle(frame, (800, 980), (1120, 1060), (0, 220, 255), -1)

    signal = OcrPhaseAnalyzer(QueueOcr([[]])).analyze(frame, 3.0)

    assert signal.choice_buttons_score < 0.55


def test_phase_analyzer_rejects_off_center_portrait_and_narrow_yellow_combat_flare():
    frame = phase_frame(1920, 1080)
    cv2.rectangle(frame, (int(1920 * 0.40), int(1080 * 0.22)), (int(1920 * 0.46), int(1080 * 0.32)), (20, 20, 230), -1)
    cv2.rectangle(frame, (int(1920 * 0.41), int(1080 * 0.34)), (int(1920 * 0.47), int(1080 * 0.41)), (0, 225, 255), -1)

    signal = OcrPhaseAnalyzer(QueueOcr([[]])).analyze(frame, 3.0)

    assert signal.countdown_score < 0.55
    assert signal.round_banner_score < 0.55


def test_phase_analyzer_rejects_near_center_red_battle_effect():
    frame = phase_frame(1920, 1080)
    cv2.circle(
        frame,
        (int(1920 * 0.445), int(1080 * 0.33)),
        int(1080 * 0.05),
        (20, 20, 230),
        -1,
        cv2.LINE_AA,
    )

    signal = OcrPhaseAnalyzer(QueueOcr([[]])).analyze(frame, 3.0)

    assert signal.countdown_score < 0.55


def test_phase_analyzer_rejects_full_width_yellow_battlefield_fire_grid():
    for width, height in ((1920, 1080), (1920, 864)):
        frame = phase_frame(width, height)
        left = int(width * 0.29)
        right = int(width * 0.71)
        top = int(height * 0.27)
        bottom = int(height * 0.54)
        thickness = max(3, int(height * 0.006))
        for y in np.linspace(top, bottom, 8, dtype=int):
            cv2.line(frame, (left, y), (right, y), (0, 225, 255), thickness)
        cv2.line(frame, (left, top), (left, bottom), (0, 225, 255), thickness)

        signal = OcrPhaseAnalyzer(QueueOcr([[]])).analyze(frame, 3.0)

        assert signal.round_banner_score < 0.55


def test_phase_analyzer_rejects_tall_central_yellow_battle_effect():
    frame = phase_frame(1920, 864)
    cv2.ellipse(
        frame,
        (960, int(864 * 0.48)),
        (int(1920 * 0.09), int(864 * 0.115)),
        0,
        0,
        360,
        (0, 225, 255),
        -1,
        cv2.LINE_AA,
    )

    signal = OcrPhaseAnalyzer(QueueOcr([[]])).analyze(frame, 3.0)

    assert signal.round_banner_score < 0.55


def test_phase_analyzer_uses_detecting_ocr_and_remains_stateless():
    ocr = QueueOcr([[OcrText("00:04", 0.9)], [OcrText("ROUND 05", 0.9)]])
    analyzer = OcrPhaseAnalyzer(ocr)
    frame = phase_frame(1920, 1080, prep=True)

    first = analyzer.analyze(frame, 10.0)
    second = analyzer.analyze(frame, 1.0)

    assert first.countdown_seconds == 4
    assert first.round_number is None
    assert second.countdown_seconds is None
    assert second.round_number == 5


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


class RecordingOcr:
    def __init__(self):
        self.detect_flags = []

    def recognize(self, image, *, detect=True):
        self.detect_flags.append(detect)
        return []


def test_phase_analyzer_uses_text_detection_for_center_region():
    ocr = RecordingOcr()

    OcrPhaseAnalyzer(ocr).analyze(phase_frame(1920, 1080, prep=True), 0.0)

    assert ocr.detect_flags == [True]


def test_phase_analyzer_skips_ocr_without_a_loose_visual_candidate():
    ocr = RecordingOcr()

    signal = OcrPhaseAnalyzer(ocr).analyze(phase_frame(1920, 1080), 0.0)

    assert ocr.detect_flags == []
    assert signal.ocr_performed is False


def test_phase_analyzer_does_not_ocr_generic_yellow_battle_effect():
    frame = phase_frame(1920, 864)
    cv2.ellipse(
        frame,
        (960, int(864 * 0.48)),
        (int(1920 * 0.09), int(864 * 0.115)),
        0,
        0,
        360,
        (0, 225, 255),
        -1,
        cv2.LINE_AA,
    )
    ocr = RecordingOcr()

    signal = OcrPhaseAnalyzer(ocr).analyze(frame, 0.0)

    assert ocr.detect_flags == []
    assert signal.ocr_performed is False


def test_phase_analyzer_coarse_mode_omits_native_corner_thumbnail():
    analyzer = OcrPhaseAnalyzer(QueueOcr([]))
    frame = phase_frame(1920, 1080)

    coarse = analyzer.analyze_coarse(frame, 0.0)
    native = analyzer.analyze(frame, 0.0)

    assert coarse.corner_signature == ()
    assert len(native.corner_signature) == 256


def test_phase_analyzer_ocr_gates_spectator_countdown_without_choice_buttons():
    frame = phase_frame(1920, 1080)
    cv2.rectangle(frame, (int(1920 * 0.18), int(1080 * 0.72)), (int(1920 * 0.82), 1079), (25, 25, 25), -1)
    ocr = QueueOcr([[OcrText("00:03", 0.9)]])

    signal = OcrPhaseAnalyzer(ocr).analyze(frame, 0.0)

    assert signal.choice_buttons_score < 0.2
    assert signal.ocr_performed is True
    assert signal.countdown_seconds == 3
