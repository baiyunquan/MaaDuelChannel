from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from maa_duel.schema import BoundingBox, RosterEntry, UnitDetection


@dataclass(frozen=True)
class RawDetection:
    enemy_id: int
    bbox: BoundingBox
    confidence: float


@dataclass(frozen=True)
class ReconciliationResult:
    left_units: list[UnitDetection] = field(default_factory=list)
    right_units: list[UnitDetection] = field(default_factory=list)
    accepted: bool = False
    reasons: tuple[str, ...] = ()


SYNERGY_MULTIPLIERS: dict[int, int] = {
    57: 2,  # 并驾骑士（腐败骑士 + 凋零骑士）
    58: 3,  # 侠客三人行（俗称“刘关张” / 桃园三结义）
}


def get_expected_unit_count(enemy_id: int, roster_count: int) -> int:
    """Returns the expected battlefield unit count for a given enemy_id taking synergy into account."""
    multiplier = SYNERGY_MULTIPLIERS.get(enemy_id, 1)
    return roster_count * multiplier


def _expected_counts(roster: list[RosterEntry]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for entry in roster:
        counts[entry.enemy_id] += get_expected_unit_count(entry.enemy_id, entry.count)
    return counts


def reconcile_detections(
    *,
    left_roster: list[RosterEntry],
    right_roster: list[RosterEntry],
    detections: list[RawDetection],
    minimum_confidence: float = 0.25,
) -> ReconciliationResult:
    left_allowed = {entry.enemy_id for entry in left_roster if entry.enemy_id > 0}
    right_allowed = {entry.enemy_id for entry in right_roster if entry.enemy_id > 0}

    # Cross-half leaks: only filter out detections strictly belonging to the opposite roster
    left_excluded = right_allowed - left_allowed
    right_excluded = left_allowed - right_allowed

    raw_by_side = {
        "left": [
            item
            for item in detections
            if item.confidence >= minimum_confidence
            and (item.bbox.x1 + item.bbox.x2) / 2 < 0.5
            and item.enemy_id not in left_excluded
        ],
        "right": [
            item
            for item in detections
            if item.confidence >= minimum_confidence
            and (item.bbox.x1 + item.bbox.x2) / 2 >= 0.5
            and item.enemy_id not in right_excluded
        ],
    }
    rosters = {"left": left_roster, "right": right_roster}
    selected_by_side: dict[str, list[UnitDetection]] = {"left": [], "right": []}
    reasons: list[str] = []

    for side_name in ("left", "right"):
        if not rosters[side_name]:
            reasons.append(f"{side_name}:roster_empty")
        expected = _expected_counts(rosters[side_name])
        actual = Counter(item.enemy_id for item in raw_by_side[side_name])
        for enemy_id in sorted(actual.keys() - expected.keys()):
            reasons.append(f"{side_name}:unexpected_enemy:{enemy_id}")
        for enemy_id in sorted(expected):
            if actual[enemy_id] != expected[enemy_id]:
                reasons.append(
                    f"{side_name}:count_mismatch:{enemy_id}:expected={expected[enemy_id]}:actual={actual[enemy_id]}"
                )
            candidates = sorted(
                (item for item in raw_by_side[side_name] if item.enemy_id == enemy_id),
                key=lambda item: item.confidence,
                reverse=True,
            )[: expected[enemy_id]]
            for item in candidates:
                selected_by_side[side_name].append(
                    UnitDetection(
                        enemy_id=item.enemy_id,
                        x=(item.bbox.x1 + item.bbox.x2) / 2,
                        y=item.bbox.y2,
                        bbox=item.bbox,
                        confidence=item.confidence,
                    )
                )
        selected_by_side[side_name].sort(key=lambda item: (item.enemy_id, item.x, item.y))

    return ReconciliationResult(
        left_units=selected_by_side["left"],
        right_units=selected_by_side["right"],
        accepted=not reasons,
        reasons=tuple(reasons),
    )
