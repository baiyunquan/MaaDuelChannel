from pathlib import Path

import cv2
import numpy as np

from maa_duel.extraction import VideoExtractor
from maa_duel.schema import BoundingBox, ReviewStatus, SourceRef, Winner
from maa_duel.video.phases import FrameSignals
from maa_duel.vision.health import HealthCounts
from maa_duel.vision.layout import RawDetection
from maa_duel.vision.roster import RosterObservation


def write_video(path: Path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
    assert writer.isOpened()
    for index in range(50):
        writer.write(np.full((180, 320, 3), index, dtype=np.uint8))
    writer.release()


class FakePhaseAnalyzer:
    def analyze(self, frame, timestamp):
        if timestamp < 0.2:
            return FrameSignals(timestamp, True, countdown_seconds=2)
        if timestamp < 0.4:
            return FrameSignals(timestamp, True, countdown_seconds=1)
        if timestamp < 0.6:
            return FrameSignals(timestamp, True, countdown_seconds=0)
        if timestamp < 1.0:
            return FrameSignals(timestamp, True, layout_score=timestamp)
        if timestamp < 1.2:
            return FrameSignals(timestamp, True, round_number=1)
        return FrameSignals(timestamp, True)


class FakeRosterRecognizer:
    def recognize(self, frame):
        return [
            RosterObservation("left", 0, 1, 1, 0.95, 0.95),
            RosterObservation("right", 0, 2, 1, 0.95, 0.95),
        ]


class FakeBattlefieldDetector:
    def detect(self, frame):
        return [
            RawDetection(1, BoundingBox(x1=0.1, y1=0.2, x2=0.2, y2=0.4), 0.95),
            RawDetection(2, BoundingBox(x1=0.8, y1=0.2, x2=0.9, y2=0.4), 0.95),
        ]


class FakeHealthDetector:
    def __init__(self):
        self.calls = 0

    def count(self, frame):
        self.calls += 1
        if self.calls <= 3:
            return HealthCounts(orange=1, blue=1)
        return HealthCounts(orange=1, blue=0)


def test_video_extractor_runs_two_pass_pipeline_and_writes_evidence(tmp_path):
    video = tmp_path / "source.mp4"
    write_video(video)
    source = SourceRef(video_relpath="绿藤/source.mp4", video_sha256="a" * 64)
    extractor = VideoExtractor(
        phase_analyzer=FakePhaseAnalyzer(),
        roster_recognizer=FakeRosterRecognizer(),
        battlefield_detector=FakeBattlefieldDetector(),
        health_detector=FakeHealthDetector(),
        scan_fps=5.0,
        stable_winner_frames=3,
    )

    samples = extractor.extract_video(video, source, tmp_path / "workspace")

    assert len(samples) == 1
    assert samples[0].winner is Winner.LEFT
    assert samples[0].review_status is ReviewStatus.ACCEPTED
    assert Path(tmp_path / "workspace" / samples[0].evidence.prep).is_file()
    assert Path(tmp_path / "workspace" / samples[0].evidence.layout).is_file()
    assert Path(tmp_path / "workspace" / samples[0].evidence.end).is_file()
