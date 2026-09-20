from maa_duel.schema import Winner
from maa_duel.vision.health import HealthBarDetector, WinnerTracker
from maa_duel.vision.roster import RosterObservation, fuse_roster_observations

import cv2
import numpy as np


def test_roster_fusion_uses_weighted_type_vote_and_count_mode():
    observations = [
        RosterObservation("left", 0, enemy_id=7, count=3, type_confidence=0.9, count_confidence=0.8),
        RosterObservation("left", 0, enemy_id=7, count=3, type_confidence=0.8, count_confidence=0.9),
        RosterObservation("left", 0, enemy_id=9, count=2, type_confidence=0.4, count_confidence=0.5),
    ]

    fused = fuse_roster_observations(observations)

    assert len(fused["left"]) == 1
    assert fused["left"][0].enemy_id == 7
    assert fused["left"][0].count == 3
    assert fused["left"][0].confidence > 0.7


def test_health_detector_counts_synthetic_orange_and_blue_bars():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 50), (80, 56), (0, 140, 255), thickness=-1)
    cv2.rectangle(image, (180, 80), (250, 86), (255, 140, 0), thickness=-1)

    counts = HealthBarDetector().count(image)

    assert counts.orange == 1
    assert counts.blue == 1


def test_winner_tracker_requires_both_sides_before_stable_elimination():
    tracker = WinnerTracker(stable_frames=3)

    assert tracker.update(orange=2, blue=0) is None
    assert tracker.update(orange=2, blue=2) is None
    assert tracker.update(orange=2, blue=0) is None
    assert tracker.update(orange=1, blue=0) is None
    assert tracker.update(orange=1, blue=0) is Winner.LEFT


def test_winner_tracker_resets_candidate_when_both_sides_return():
    tracker = WinnerTracker(stable_frames=2)
    tracker.update(orange=1, blue=1)
    assert tracker.update(orange=0, blue=1) is None
    assert tracker.update(orange=1, blue=1) is None
    assert tracker.update(orange=0, blue=1) is None
    assert tracker.update(orange=0, blue=1) is Winner.RIGHT
