from pathlib import Path

import cv2
import numpy as np
import pytest

from maa_duel.extraction import VideoExtractor
from maa_duel.schema import BoundingBox, ReviewStatus, SourceRef, Winner
from maa_duel.video.phases import FrameSignals
from maa_duel.vision.layout import RawDetection
from maa_duel.vision.roster import RosterObservation

FPS = 30.0
PREP = 20
LAYOUT = 60
BANNER = 100
BATTLE = 150


def write_stage_video(path: Path, values: list[int]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 180))
    assert writer.isOpened()
    for value in values:
        writer.write(np.full((180, 320, 3), value, dtype=np.uint8))
    writer.release()


def round_frames(*, with_layout: bool = True, battle_frames: int = 75) -> list[int]:
    values = [PREP] * 31
    if with_layout:
        values.extend([LAYOUT] * 2)
    values.extend([BANNER] * 12)
    values.extend([BATTLE] * battle_frames)
    return values


class MarkerPhaseAnalyzer:
    def analyze(self, frame, timestamp):
        value = float(np.mean(frame))
        if value > 220:
            return FrameSignals(timestamp)
        if value < 40:
            return FrameSignals(
                timestamp,
                battlefield_score=1.0,
                bottom_panel_score=1.0,
                choice_buttons_score=1.0,
                countdown_score=1.0,
                corner_mask_score=1.0,
                countdown_seconds=2,
            )
        if value < 80:
            return FrameSignals(timestamp, battlefield_score=1.0, corner_mask_score=1.0)
        if value < 125:
            return FrameSignals(
                timestamp,
                battlefield_score=1.0,
                round_banner_score=1.0,
                corner_mask_score=1.0,
                round_number=1,
            )
        return FrameSignals(timestamp, battlefield_score=1.0)


class FakeRosterRecognizer:
    def recognize(self, frame):
        return [
            RosterObservation("left", 0, 1, 1, 0.95, 0.95),
            RosterObservation("right", 0, 2, 1, 0.95, 0.95),
        ]


class MarkerBattlefieldDetector:
    def detect(self, frame, *, candidate_enemy_ids=None):
        value = float(np.mean(frame))
        items = [RawDetection(1, BoundingBox(x1=0.3, y1=0.2, x2=0.4, y2=0.5), 0.95)]
        if value < 125:
            items.append(RawDetection(2, BoundingBox(x1=0.6, y1=0.2, x2=0.7, y2=0.5), 0.95))
        if candidate_enemy_ids is not None:
            items = [item for item in items if item.enemy_id in candidate_enemy_ids]
        return items


def make_extractor(*, evidence_prefix: Path = Path("frames")) -> VideoExtractor:
    return VideoExtractor(
        phase_analyzer=MarkerPhaseAnalyzer(),
        roster_recognizer=FakeRosterRecognizer(),
        battlefield_detector=MarkerBattlefieldDetector(),
        scan_fps=10.0,
        evidence_prefix=evidence_prefix,
    )


def source(digest: str = "a") -> SourceRef:
    return SourceRef(video_relpath="绿藤/source.mp4", video_sha256=digest * 64)


def test_native_rate_refinement_finds_two_frame_layout_gap_missed_by_coarse_scan(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    values = round_frames()
    write_stage_video(video, values)
    extractor = make_extractor()

    samples = extractor.extract_video(
        video,
        source(),
        tmp_path / "workspace",
        duration=len(values) / FPS,
        source_fps=FPS,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.timestamps.layout == pytest.approx(32 / FPS)
    assert sample.timestamps.battle_start == pytest.approx(45 / FPS)
    assert sample.timestamps.battle_end == pytest.approx((len(values) - 1) / FPS)
    assert sample.winner is Winner.LEFT
    assert sample.review_status is ReviewStatus.ACCEPTED
    diagnostics = extractor.phase_diagnostics
    assert diagnostics["resolution"] == {"width": 320, "height": 180}
    assert diagnostics["fine_frames"] > 0
    assert diagnostics["ocr_calls"] <= diagnostics["coarse_frames"] + diagnostics["fine_frames"]
    assert diagnostics["selected_frames"][0]["layout"]["frame_index"] == 32
    assert diagnostics["selected_frames"][0]["end"]["timestamp"] == sample.timestamps.battle_end


def test_layout_unresolved_never_falls_back_to_prep_or_banner(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    values = round_frames(with_layout=False)
    write_stage_video(video, values)
    extractor = make_extractor()

    workspace = tmp_path / "workspace"
    samples = extractor.extract_video(
        video,
        source("b"),
        workspace,
        duration=len(values) / FPS,
        source_fps=FPS,
    )

    assert samples == []
    assert any(issue.kind == "layout_unresolved" for issue in extractor.issues)
    contact_sheets = extractor.phase_diagnostics["contact_sheets"]
    assert len(contact_sheets) == 1
    assert (workspace / contact_sheets[0]["path"]).is_file()


def test_native_refinement_recovers_prep_when_coarse_scan_only_sees_round(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    values = [240, PREP, PREP, LAYOUT, LAYOUT, *([BANNER] * 12), *([BATTLE] * 75)]
    write_stage_video(video, values)
    extractor = make_extractor()

    samples = extractor.extract_video(
        video,
        source("f"),
        tmp_path / "workspace",
        duration=len(values) / FPS,
        source_fps=FPS,
    )

    assert len(samples) == 1
    assert samples[0].timestamps.prep == pytest.approx(2 / FPS)
    assert samples[0].timestamps.layout == pytest.approx(4 / FPS)


def test_previous_round_end_frame_stays_before_next_round_prep(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    first = round_frames()
    second = round_frames()
    values = first + second
    write_stage_video(video, values)
    workspace = tmp_path / "workspace"

    samples = make_extractor().extract_video(
        video,
        source("c"),
        workspace,
        duration=len(values) / FPS,
        source_fps=FPS,
    )

    assert len(samples) == 2
    assert samples[0].timestamps.battle_end < len(first) / FPS
    first_end = cv2.imread(str(workspace / samples[0].evidence.end))
    assert first_end is not None
    assert float(np.mean(first_end)) > 130.0


def test_unresolved_later_round_does_not_discard_earlier_success(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    first = round_frames()
    second = round_frames(with_layout=False)
    values = first + second
    write_stage_video(video, values)
    extractor = make_extractor()

    samples = extractor.extract_video(
        video,
        source("d"),
        tmp_path / "workspace",
        duration=len(values) / FPS,
        source_fps=FPS,
    )

    assert [sample.round_index for sample in samples] == [1]
    assert any(issue.round_index == 2 and issue.kind == "layout_unresolved" for issue in extractor.issues)


def test_evidence_prefix_can_target_a_staged_extraction_run(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    values = round_frames()
    write_stage_video(video, values)
    prefix = Path("extraction-runs") / "run-1" / "frames"

    sample = make_extractor(evidence_prefix=prefix).extract_video(
        video,
        source("e"),
        tmp_path / "staging",
        duration=len(values) / FPS,
        source_fps=FPS,
    )[0]

    assert sample.evidence.layout is not None
    assert Path(sample.evidence.layout).is_relative_to(prefix)
    assert (tmp_path / "staging" / sample.evidence.layout).is_file()
