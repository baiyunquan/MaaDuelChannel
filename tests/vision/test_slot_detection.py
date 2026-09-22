from pathlib import Path

import cv2
import numpy as np

from maa_duel.vision.roster import (
    Classification,
    DualEngineClassifier,
    RosterFrameRecognizer,
    TemplateMatchClassifier,
    calculate_safe_zone,
    default_slot_specs,
    detect_slot_specs,
)


def test_calculate_safe_zone_ratios() -> None:
    # 16:9 exact
    ox, oy, sw, sh = calculate_safe_zone(1920, 1080)
    assert (ox, oy) == (0, 0)
    assert (sw, sh) == (1920, 1080)

    # 20:9 wide (1920x864)
    ox, oy, sw, sh = calculate_safe_zone(1920, 864)
    assert oy == 0
    assert sh == 864
    assert sw == 1536
    assert ox == 192

    # 4:3 tall (1440x1080)
    ox, oy, sw, sh = calculate_safe_zone(1440, 1080)
    assert ox == 0
    assert sw == 1440
    assert sh == 810
    assert oy == 135


def test_default_slot_specs_layout() -> None:
    specs = default_slot_specs((1080, 1920))
    assert len(specs) == 6
    sides = [s.side for s in specs]
    assert sides == ["left", "left", "left", "right", "right", "right"]

    for s in specs:
        x1, y1, x2, y2 = s.icon_rect
        assert 0.0 <= x1 < x2 <= 1.0
        assert 0.0 <= y1 < y2 <= 1.0
        cy = (y1 + y2) / 2
        # Nominal center Y should be ~0.894
        assert abs(cy - 0.8944) < 0.01


def test_detect_slot_specs_synthetic_fallback() -> None:
    # Blank frame: should fall back to default_slot_specs
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    specs = detect_slot_specs(blank)
    assert len(specs) == 6
    default_specs = default_slot_specs((1080, 1920))
    for s, ds in zip(specs, default_specs, strict=True):
        assert np.allclose(s.icon_rect, ds.icon_rect, atol=1e-4)


def test_detect_slot_specs_synthetic_circles() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    base = default_slot_specs((1080, 1920))

    # Draw bright white circles at 3 of the slots
    for idx in [0, 1, 3]:
        spec = base[idx]
        cx = int((spec.icon_rect[0] + spec.icon_rect[2]) * 0.5 * 1920)
        cy = int((spec.icon_rect[1] + spec.icon_rect[3]) * 0.5 * 1080)
        r = int(1080 * 0.050)
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), r - 5, (200, 200, 200), -1)

    specs = detect_slot_specs(frame)
    assert len(specs) == 6
    # Detected slots should be very close to nominal positions
    for s, ds in zip(specs, base, strict=True):
        assert abs(s.icon_rect[0] - ds.icon_rect[0]) < 0.03
        assert abs(s.icon_rect[1] - ds.icon_rect[1]) < 0.03


def test_template_match_classifier(tmp_path: Path) -> None:
    portraits_dir = tmp_path / "portraits"
    portraits_dir.mkdir(parents=True)

    # Create dummy templates for enemy 10 and enemy 20
    t10_dir = portraits_dir / "0010"
    t10_dir.mkdir()
    img10 = np.full((100, 100, 3), 180, dtype=np.uint8)
    cv2.circle(img10, (50, 50), 30, (50, 50, 220), -1)
    cv2.imwrite(str(t10_dir / "thumbnail.png"), img10)

    t20_dir = portraits_dir / "0020"
    t20_dir.mkdir()
    img20 = np.full((100, 100, 3), 180, dtype=np.uint8)
    cv2.rectangle(img20, (20, 20), (80, 80), (220, 50, 50), -1)
    cv2.imwrite(str(t20_dir / "thumbnail.png"), img20)

    classifier = TemplateMatchClassifier(portraits_dir, target_size=64)

    # Classify crop similar to img10
    target10 = cv2.resize(img10, (64, 64))
    res10 = classifier.classify(target10)
    assert res10.enemy_id == 10
    assert res10.confidence > 0.85

    # Target with corner colors (e.g. red/blue team corners) and corner occlusions
    noisy10 = img10.copy()
    noisy10[:20, :20] = (0, 0, 255)  # team corner
    noisy10[-20:, -20:] = (0, 0, 255)  # count text corner
    res10_noisy = classifier.classify(cv2.resize(noisy10, (64, 64)))
    assert res10_noisy.enemy_id == 10
    assert res10_noisy.confidence > 0.80

    # Classify crop similar to img20
    target20 = cv2.resize(img20, (64, 64))
    res20 = classifier.classify(target20)
    assert res20.enemy_id == 20
    assert res20.confidence > 0.80

    # Low contrast flat crop -> correctly recognized as empty slot (enemy_id=0)
    flat_crop = np.full((100, 100, 3), 60, dtype=np.uint8)
    res_flat = classifier.classify(flat_crop)
    assert res_flat.enemy_id == 0
    assert res_flat.confidence == 1.0


def test_dual_engine_classifier_veto() -> None:
    class DummyYolo:
        def __init__(self, pred_id: int, conf: float) -> None:
            self.pred_id = pred_id
            self.conf = conf

        def classify(self, _img: np.ndarray) -> Classification:
            return Classification(enemy_id=self.pred_id, confidence=self.conf)

    class DummyTemplate:
        def __init__(self, pred_id: int, conf: float) -> None:
            self.pred_id = pred_id
            self.conf = conf

        def classify(self, _img: np.ndarray) -> Classification:
            return Classification(enemy_id=self.pred_id, confidence=self.conf)

        def classify_batch(self, images: list[np.ndarray]) -> list[Classification]:
            return [self.classify(img) for img in images]

    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)

    # 1. Agreement: both predict enemy 15 -> confidence boosted
    dual_agree = DualEngineClassifier(
        yolo_classifier=DummyYolo(15, 0.85),
        template_classifier=DummyTemplate(15, 0.70),  # type: ignore[arg-type]
    )
    res_agree = dual_agree.classify(dummy_img)
    assert res_agree.enemy_id == 15
    assert res_agree.confidence >= 0.95

    # 2. Disagreement: YOLO predicts 15, Template predicts 20 -> one-vote veto penalty
    dual_disagree = DualEngineClassifier(
        yolo_classifier=DummyYolo(15, 0.90),
        template_classifier=DummyTemplate(20, 0.80),  # type: ignore[arg-type]
    )
    res_disagree = dual_disagree.classify(dummy_img)
    assert res_disagree.enemy_id == 15
    assert res_disagree.confidence <= 0.15  # Veto downweighting


def test_roster_frame_recognizer_dynamic_slots() -> None:
    class DummyClassifier:
        def classify(self, _img: np.ndarray) -> Classification:
            return Classification(enemy_id=5, confidence=0.9)

        def classify_batch(self, images: list[np.ndarray]) -> list[Classification]:
            return [self.classify(img) for img in images]

    class DummyCountClassifier:
        def classify_batch(self, images: list[np.ndarray]):
            from maa_duel.vision.roster import CountClassification

            return [CountClassification(count=2, confidence=0.95) for _ in images]

    class DummyOcr:
        def recognize(self, _img: np.ndarray, detect: bool = False):
            return []

    recognizer = RosterFrameRecognizer(
        classifier=DummyClassifier(),  # type: ignore[arg-type]
        ocr=DummyOcr(),  # type: ignore[arg-type]
        count_classifier=DummyCountClassifier(),  # type: ignore[arg-type]
    )
    # 6 slots should be dynamically computed from frame
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    observations = recognizer.recognize(frame)
    assert len(observations) == 6
    assert observations[0].enemy_id == 5
    assert observations[0].count == 2

