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


def _expected_counts(roster: list[RosterEntry]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for entry in roster:
        counts[entry.enemy_id] += entry.count
    return counts


def reconcile_detections(
    *,
    left_roster: list[RosterEntry],
    right_roster: list[RosterEntry],
    detections: list[RawDetection],
    minimum_confidence: float = 0.25,
) -> ReconciliationResult:
    raw_by_side = {
        "left": [
            item
            for item in detections
            if item.confidence >= minimum_confidence and (item.bbox.x1 + item.bbox.x2) / 2 < 0.5
        ],
        "right": [
            item
            for item in detections
            if item.confidence >= minimum_confidence and (item.bbox.x1 + item.bbox.x2) / 2 >= 0.5
        ],
    }
    rosters = {"left": left_roster, "right": right_roster}
    selected_by_side: dict[str, list[UnitDetection]] = {"left": [], "right": []}
    reasons: list[str] = []

    for side_name in ("left", "right"):
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
