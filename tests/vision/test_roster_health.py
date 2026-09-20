import cv2
import numpy as np

from maa_duel.schema import Winner
from maa_duel.vision.health import HealthBarDetector, WinnerTracker
from maa_duel.vision.roster import RosterObservation, fuse_roster_observations


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


def test_roster_fusion_does_not_promote_one_low_confidence_observation():
    observations = [
        RosterObservation("left", 0, enemy_id=7, count=3, type_confidence=0.26, count_confidence=0.26),
    ]

    fused = fuse_roster_observations(observations)

    assert fused["left"][0].confidence == 0.26


def test_health_detector_counts_synthetic_orange_and_blue_bars():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(image, (60, 50), (120, 56), (0, 140, 255), thickness=-1)
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


def test_health_detector_rejects_long_ui_lines_and_side_grid():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(image, (70, 80), (100, 84), (0, 140, 255), thickness=-1)
    cv2.rectangle(image, (190, 100), (225, 104), (255, 140, 0), thickness=-1)
    cv2.rectangle(image, (30, 20), (270, 24), (255, 140, 0), thickness=-1)
    cv2.rectangle(image, (10, 20), (14, 180), (255, 140, 0), thickness=-1)
    cv2.rectangle(image, (5, 140), (55, 144), (0, 140, 255), thickness=-1)

    counts = HealthBarDetector().count(image)

    assert counts.orange == 1
    assert counts.blue == 1


def test_health_detector_rejects_dark_desaturated_blue_floor_highlight():
    hsv = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(hsv, (80, 60), (130, 66), (104, 95, 120), thickness=-1)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    counts = HealthBarDetector().count(image)

    assert counts.blue == 0
