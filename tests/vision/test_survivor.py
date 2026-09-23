from __future__ import annotations

import numpy as np
import pytest

from maa_duel.schema import BoundingBox, Winner
from maa_duel.video.evidence import TimedFrame
from maa_duel.video.phases import FrameSignals
from maa_duel.vision.layout import RawDetection
from maa_duel.vision.survivor import BattlefieldSurvivorDetector


def detection(enemy_id: int, *, x: float = 0.5) -> RawDetection:
    return RawDetection(
        enemy_id,
        BoundingBox(x1=x - 0.04, y1=0.4, x2=x + 0.04, y2=0.6),
        0.9,
    )


def end_frame(index: int) -> TimedFrame:
    return TimedFrame(
        timestamp=float(index),
        frame_index=index,
        frame=np.full((32, 48, 3), index, dtype=np.uint8),
        signals=FrameSignals(timestamp=float(index), battlefield_score=1.0),
    )


class SequenceDetector:
    def __init__(self, detections: dict[int, list[RawDetection]]) -> None:
        self.detections = detections

    def detect_batch(self, frames, *, candidate_enemy_ids=None):
        result = []
        for frame in frames:
            items = self.detections.get(int(frame[0, 0, 0]), [])
            if candidate_enemy_ids is not None:
                items = [item for item in items if item.enemy_id in candidate_enemy_ids]
            result.append(items)
        return result


def test_survivor_requires_three_of_last_five_supporting_frames() -> None:
    detector = SequenceDetector(
        {
            1: [detection(1)],
            2: [],
            3: [detection(1)],
            4: [detection(1)],
            5: [],
        }
    )

    result = BattlefieldSurvivorDetector(detector).detect_winner(
        end_run=tuple(end_frame(index) for index in range(1, 6)),
        left_eids={1},
        right_eids={2},
    )

    assert result.winner is Winner.LEFT
    assert result.evidence_time == 4.0
    assert result.left_survivors == 1
    assert result.right_survivors == 0


def test_single_frame_survivor_false_positive_remains_unresolved() -> None:
    detector = SequenceDetector({1: [detection(1)]})

    result = BattlefieldSurvivorDetector(detector).detect_winner(
        end_run=tuple(end_frame(index) for index in range(1, 6)),
        left_eids={1},
        right_eids={2},
    )

    assert result.winner is None
    assert result.evidence_time == 5.0
    assert result.details == "unresolved_end_game"


def test_two_legal_end_frames_are_not_enough_to_resolve_winner() -> None:
    detector = SequenceDetector({1: [detection(1)], 2: [detection(1)]})

    result = BattlefieldSurvivorDetector(detector).detect_winner(
        end_run=(end_frame(1), end_frame(2)),
        left_eids={1},
        right_eids={2},
    )

    assert result.winner is None
    assert result.evidence_time == 2.0


def test_contradictory_survivor_evidence_remains_unresolved() -> None:
    detector = SequenceDetector(
        {
            1: [detection(1)],
            2: [detection(1)],
            3: [detection(1)],
            4: [detection(2)],
            5: [],
        }
    )

    result = BattlefieldSurvivorDetector(detector).detect_winner(
        end_run=tuple(end_frame(index) for index in range(1, 6)),
        left_eids={1},
        right_eids={2},
    )

    assert result.winner is None
    assert result.evidence_time == 5.0


def test_survivor_ignores_detections_outside_battlefield_roi() -> None:
    detector = SequenceDetector({index: [detection(1, x=0.05)] for index in range(1, 6)})

    result = BattlefieldSurvivorDetector(detector).detect_winner(
        end_run=tuple(end_frame(index) for index in range(1, 6)),
        left_eids={1},
        right_eids={2},
    )

    assert result.winner is None


def test_survivor_rejects_empty_validated_end_run() -> None:
    with pytest.raises(ValueError, match="validated end frames"):
        BattlefieldSurvivorDetector(SequenceDetector({})).detect_winner(
            end_run=(),
            left_eids={1},
            right_eids={2},
        )
