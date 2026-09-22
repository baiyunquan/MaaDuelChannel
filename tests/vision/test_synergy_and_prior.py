from unittest.mock import MagicMock

import numpy as np

from maa_duel.schema import BoundingBox, RosterEntry
from maa_duel.training.vision import YoloBattlefieldDetector
from maa_duel.vision.layout import (
    SYNERGY_MULTIPLIERS,
    RawDetection,
    _expected_counts,
    get_expected_unit_count,
    reconcile_detections,
)


def test_synergy_multipliers() -> None:
    assert SYNERGY_MULTIPLIERS[58] == 3  # 侠客三人行 (刘关张)
    assert SYNERGY_MULTIPLIERS[57] == 2  # 并驾骑士

    assert get_expected_unit_count(58, 1) == 3
    assert get_expected_unit_count(58, 2) == 6
    assert get_expected_unit_count(57, 1) == 2
    assert get_expected_unit_count(25, 4) == 4  # Normal monster


def test_expected_counts_with_synergy() -> None:
    roster = [
        RosterEntry(enemy_id=58, count=1, confidence=0.9),  # 3 units
        RosterEntry(enemy_id=25, count=2, confidence=0.9),  # 2 units
    ]
    counts = _expected_counts(roster)
    assert counts[58] == 3
    assert counts[25] == 2


def test_reconcile_detections_with_synergy() -> None:
    # Left has ID 58 (count 1 -> expects 3 units)
    left_roster = [RosterEntry(enemy_id=58, count=1, confidence=0.9)]
    # Right has ID 34 (count 2 -> expects 2 units)
    right_roster = [RosterEntry(enemy_id=34, count=2, confidence=0.9)]

    # Detections:
    # 3 units of ID 58 on left side (x < 0.5)
    # 2 units of ID 34 on right side (x >= 0.5)
    detections = [
        RawDetection(enemy_id=58, bbox=BoundingBox(x1=0.1, y1=0.2, x2=0.2, y2=0.4), confidence=0.8),
        RawDetection(enemy_id=58, bbox=BoundingBox(x1=0.2, y1=0.2, x2=0.3, y2=0.4), confidence=0.85),
        RawDetection(enemy_id=58, bbox=BoundingBox(x1=0.3, y1=0.2, x2=0.4, y2=0.4), confidence=0.9),
        RawDetection(enemy_id=34, bbox=BoundingBox(x1=0.6, y1=0.2, x2=0.7, y2=0.4), confidence=0.8),
        RawDetection(enemy_id=34, bbox=BoundingBox(x1=0.7, y1=0.2, x2=0.8, y2=0.4), confidence=0.85),
    ]

    result = reconcile_detections(
        left_roster=left_roster,
        right_roster=right_roster,
        detections=detections,
    )
    assert result.accepted is True
    assert len(result.left_units) == 3
    assert len(result.right_units) == 2


def test_reconcile_detections_half_field_exclusivity() -> None:
    left_roster = [RosterEntry(enemy_id=25, count=1, confidence=0.9)]
    right_roster = [RosterEntry(enemy_id=34, count=1, confidence=0.9)]

    # Detections:
    # Left has 1 valid ID 25, and 1 FALSE ID 34 on the left side (x < 0.5)
    # Right has 1 valid ID 34
    detections = [
        RawDetection(enemy_id=25, bbox=BoundingBox(x1=0.2, y1=0.3, x2=0.3, y2=0.5), confidence=0.9),
        RawDetection(enemy_id=34, bbox=BoundingBox(x1=0.35, y1=0.3, x2=0.45, y2=0.5), confidence=0.7),  # noise!
        RawDetection(enemy_id=34, bbox=BoundingBox(x1=0.7, y1=0.3, x2=0.8, y2=0.5), confidence=0.9),
    ]

    result = reconcile_detections(
        left_roster=left_roster,
        right_roster=right_roster,
        detections=detections,
    )
    # The false ID 34 on the left should be excluded by half-field exclusivity!
    assert result.accepted is True
    assert len(result.left_units) == 1
    assert result.left_units[0].enemy_id == 25
    assert len(result.right_units) == 1
    assert result.right_units[0].enemy_id == 34


def test_yolo_battlefield_detector_classes_filter(tmp_path) -> None:
    class_map_file = tmp_path / "class-map.json"
    # Class map: model idx 0 -> enemy 25, model idx 1 -> enemy 34, model idx 2 -> enemy 58
    class_map_file.write_text('{"0": 25, "1": 34, "2": 58}', encoding="utf-8")

    dummy_model_file = tmp_path / "model.pt"
    dummy_model_file.touch()

    # Mock YOLO
    mock_yolo_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = None
    mock_yolo_instance.predict.return_value = [mock_result]

    mock_yolo_cls = MagicMock(return_value=mock_yolo_instance)

    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
        detector = YoloBattlefieldDetector(
            model_path=dummy_model_file,
            class_map_path=class_map_file,
        )

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Detect with candidate_enemy_ids=[34, 58]
        detector.detect(dummy_img, candidate_enemy_ids=[34, 58])

        mock_yolo_instance.predict.assert_called_once()
        call_kwargs = mock_yolo_instance.predict.call_args.kwargs
        assert "classes" in call_kwargs
        # Classes should be [1, 2] corresponding to IDs 34 and 58
        assert sorted(call_kwargs["classes"]) == [1, 2]
