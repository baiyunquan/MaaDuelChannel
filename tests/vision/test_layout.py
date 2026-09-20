from maa_duel.schema import BoundingBox, RosterEntry
from maa_duel.vision.layout import RawDetection, reconcile_detections


def detection(enemy_id, x, confidence=0.9):
    return RawDetection(
        enemy_id=enemy_id,
        bbox=BoundingBox(x1=x - 0.02, y1=0.3, x2=x + 0.02, y2=0.4),
        confidence=confidence,
    )


def test_reconciliation_assigns_sides_and_uses_bottom_center_position():
    result = reconcile_detections(
        left_roster=[RosterEntry(enemy_id=1, count=2, confidence=0.9)],
        right_roster=[RosterEntry(enemy_id=2, count=1, confidence=0.9)],
        detections=[detection(1, 0.2), detection(1, 0.3), detection(2, 0.8)],
    )

    assert result.accepted
    assert [unit.x for unit in result.left_units] == [0.2, 0.3]
    assert result.left_units[0].y == 0.4
    assert [unit.enemy_id for unit in result.right_units] == [2]


def test_reconciliation_rejects_missing_and_unexpected_units():
    result = reconcile_detections(
        left_roster=[RosterEntry(enemy_id=1, count=2, confidence=0.9)],
        right_roster=[RosterEntry(enemy_id=2, count=1, confidence=0.9)],
        detections=[detection(1, 0.2), detection(9, 0.8)],
    )

    assert not result.accepted
    assert any(reason.startswith("left:count_mismatch") for reason in result.reasons)
    assert any(reason.startswith("right:unexpected_enemy") for reason in result.reasons)
