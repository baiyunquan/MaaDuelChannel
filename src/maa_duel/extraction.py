from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from maa_duel.extraction_state import (
    StagedExtraction,
    active_extraction_paths,
    create_staged_extraction,
    publish_staged_extraction,
)
from maa_duel.schema import (
    EvidenceFrames,
    ReviewStatus,
    RosterEntry,
    RoundSample,
    SideData,
    SourceRef,
    Timestamps,
    Winner,
)
from maa_duel.video.evidence import (
    TimedFrame,
    select_battle_start,
    select_end_run,
    select_layout_frame,
    select_prep_frame,
)
from maa_duel.video.phases import FrameSignals, RoundCandidate, RoundSegmenter, RoundWindow, TimeSpan
from maa_duel.vision.health import HealthCounts, WinnerTracker
from maa_duel.vision.layout import RawDetection, reconcile_detections
from maa_duel.vision.roster import RosterObservation, fuse_roster_observations
from maa_duel.vision.survivor import BattlefieldSurvivorDetector


class PhaseAnalyzer(Protocol):
    def analyze(self, frame: np.ndarray, timestamp: float) -> FrameSignals: ...


class RosterRecognizer(Protocol):
    def recognize(self, frame: np.ndarray) -> list[RosterObservation]: ...


class BattlefieldDetector(Protocol):
    def detect(
        self,
        frame: np.ndarray,
        *,
        candidate_enemy_ids: Iterable[int] | None = None,
    ) -> list[RawDetection]: ...


class HealthDetector(Protocol):
    def count(self, frame: np.ndarray) -> HealthCounts: ...


@dataclass(frozen=True)
class ExtractionIssue:
    round_index: int
    kind: str
    detail: str
    observed_round_number: int | None = None
    start: float | None = None
    end: float | None = None


@dataclass(frozen=True)
class RefinedRound:
    window: RoundWindow
    prep: TimedFrame
    layout: TimedFrame
    battle_start: TimedFrame
    end_run: tuple[TimedFrame, ...]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path, default: object) -> object:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_review_store(source: Path, destination: Path, *, reset: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not reset and source.is_file():
        shutil.copy2(source, destination)
    else:
        destination.write_text("", encoding="utf-8")


def _carry_forward_samples(
    workspace: Path,
    staged: StagedExtraction,
    samples: list[RoundSample],
) -> list[RoundSample]:
    """Copy an active run into a new immutable run without sharing evidence paths."""

    carried: list[RoundSample] = []
    for sample in samples:
        evidence: dict[str, str] = {}
        for label in ("prep", "layout", "end"):
            raw_path = getattr(sample.evidence, label)
            if raw_path is None:
                raise ValueError(
                    f"cannot carry sample {sample.sample_id} with missing {label} evidence; use --force"
                )
            source_path = workspace / raw_path
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"cannot carry sample {sample.sample_id}; evidence is missing: {source_path}"
                )
            suffix = source_path.suffix or ".jpg"
            relative = (
                staged.evidence_prefix
                / sample.source.video_sha256[:12]
                / f"{sample.sample_id}-{label}{suffix}"
            )
            destination = staged.staging_workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_path, destination)
            except OSError:
                shutil.copy2(source_path, destination)
            evidence[label] = relative.as_posix()
        carried.append(
            sample.model_copy(
                update={"evidence": sample.evidence.model_copy(update=evidence)},
            )
        )
    return carried


def stable_sample_id(source: SourceRef, round_index: int, _layout_time: float) -> str:
    value = f"{source.video_sha256}:{round_index}".encode()
    return hashlib.blake2b(value, digest_size=16).hexdigest()


def assemble_round_sample(
    *,
    source: SourceRef,
    window: RoundWindow,
    rosters: dict[str, list[RosterEntry]],
    detections: list[RawDetection],
    winner: Winner | None,
    winner_confidence: float | None,
    evidence: dict[str, str],
    acceptance_confidence: float = 0.8,
    pipeline_version: str = "0.2.0",
    extra_reasons: tuple[str, ...] = (),
) -> RoundSample:
    if None in (window.prep_time, window.layout_time, window.battle_start):
        raise ValueError("round window must contain prep, layout, and battle timestamps")

    reconciled = reconcile_detections(
        left_roster=rosters.get("left", []),
        right_roster=rosters.get("right", []),
        detections=detections,
    )
    reasons = list(window.failure_reasons) + list(reconciled.reasons) + list(extra_reasons)
    if winner is None:
        reasons.append("winner_unresolved")
    confidences = [
        *(item.confidence for item in rosters.get("left", [])),
        *(item.confidence for item in rosters.get("right", [])),
        *(item.confidence for item in reconciled.left_units),
        *(item.confidence for item in reconciled.right_units),
    ]
    if winner_confidence is not None:
        confidences.append(winner_confidence)
    high_confidence = bool(confidences) and min(confidences) >= acceptance_confidence
    accepted = (
        window.complete
        and reconciled.accepted
        and winner is not None
        and high_confidence
        and not extra_reasons
    )
    if not high_confidence:
        reasons.append("low_confidence")

    return RoundSample(
        sample_id=stable_sample_id(source, window.round_index, float(window.layout_time)),
        source=source,
        round_index=window.round_index,
        timestamps=Timestamps(
            prep=float(window.prep_time),
            layout=float(window.layout_time),
            battle_start=float(window.battle_start),
            battle_end=window.battle_end,
        ),
        evidence=EvidenceFrames(**evidence),
        left=SideData(roster=rosters.get("left", []), units=reconciled.left_units),
        right=SideData(roster=rosters.get("right", []), units=reconciled.right_units),
        winner=winner,
        winner_confidence=winner_confidence,
        review_status=ReviewStatus.ACCEPTED if accepted else ReviewStatus.PENDING,
        pipeline_version=pipeline_version,
        failure_reasons=sorted(set(reasons)),
    )


class VideoExtractor:
    """Two-pass video extractor: find windows first, then run expensive models only on evidence frames."""

    def __init__(
        self,
        *,
        phase_analyzer: PhaseAnalyzer,
        roster_recognizer: RosterRecognizer,
        battlefield_detector: BattlefieldDetector,
        health_detector: HealthDetector | None = None,
        survivor_detector: BattlefieldSurvivorDetector | None = None,
        scan_fps: float = 10.0,
        stable_winner_frames: int = 5,
        evidence_prefix: Path = Path("frames"),
        phase_debug: bool = False,
        contact_sheet_prefix: Path = Path("reports/phase-contact-sheets"),
    ) -> None:
        self.phase_analyzer = phase_analyzer
        self.roster_recognizer = roster_recognizer
        self.battlefield_detector = battlefield_detector
        self.health_detector = health_detector
        self.survivor_detector = survivor_detector or BattlefieldSurvivorDetector(battlefield_detector)
        self.scan_fps = scan_fps
        self.stable_winner_frames = stable_winner_frames
        if evidence_prefix.is_absolute() or ".." in evidence_prefix.parts:
            raise ValueError("evidence_prefix must be a safe workspace-relative path")
        if contact_sheet_prefix.is_absolute() or ".." in contact_sheet_prefix.parts:
            raise ValueError("contact_sheet_prefix must be a safe workspace-relative path")
        self.evidence_prefix = evidence_prefix
        self.phase_debug = phase_debug
        self.contact_sheet_prefix = contact_sheet_prefix
        self.issues: list[ExtractionIssue] = []
        self.phase_diagnostics: dict[str, object] = {}
        self._fine_frame_count = 0
        self._ocr_call_count = 0

    def extract_video(
        self,
        video_path: Path,
        source: SourceRef,
        workspace: Path,
        *,
        duration: float | None = None,
        source_fps: float | None = None,
    ) -> list[RoundSample]:
        self.issues = []
        self._fine_frame_count = 0
        self._ocr_call_count = 0
        if duration is None or source_fps is None:
            detected_duration, detected_fps = self._video_properties(video_path)
            duration = duration if duration is not None else detected_duration
            source_fps = source_fps if source_fps is not None else detected_fps
        width, height = self._video_resolution(video_path)

        sample_gen = self._sample_video(video_path, start=0.0, end=duration)
        coarse_analyze = getattr(self.phase_analyzer, "analyze_coarse", self.phase_analyzer.analyze)
        try:
            signals = [coarse_analyze(frame, timestamp) for timestamp, frame in sample_gen]
        finally:
            with contextlib.suppress(Exception):
                sample_gen.close()
        self._ocr_call_count += sum(signal.ocr_performed for signal in signals)
        candidates = RoundSegmenter().segment(signals, video_end=duration)
        self.phase_diagnostics = {
            "video": source.video_relpath,
            "video_sha256": source.video_sha256,
            "duration": duration,
            "source_fps": source_fps,
            "resolution": {"width": width, "height": height},
            "coarse_frames": len(signals),
            "candidates": [asdict(candidate) for candidate in candidates],
            "state_intervals": [
                {
                    "round_index": candidate.round_index,
                    "prep": asdict(candidate.prep_span) if candidate.prep_span is not None else None,
                    "layout_gap_search": asdict(candidate.layout_search),
                    "round": asdict(candidate.round_span) if candidate.round_span is not None else None,
                    "battle_search": asdict(candidate.battle_start_search),
                    "end_search": asdict(candidate.end_search),
                }
                for candidate in candidates
            ],
        }
        samples: list[RoundSample] = []
        selected_frames: list[dict[str, object]] = []
        native_capture = cv2.VideoCapture(str(video_path)) if candidates else None
        if native_capture is not None and not native_capture.isOpened():
            native_capture.release()
            raise ValueError(f"cannot open video for native-rate refinement: {video_path}")
        try:
            for candidate in candidates:
                refined = self._refine_candidate(
                    video_path,
                    candidate,
                    source_fps,
                    capture=native_capture,
                )
                if refined is None:
                    continue
                try:
                    sample = self._extract_window(video_path, source, workspace, refined)
                    samples.append(sample)
                    end = min(
                        refined.end_run,
                        key=lambda frame: abs(frame.timestamp - sample.timestamps.battle_end),
                    )
                    selected_frames.append(
                        {
                            "round_index": candidate.round_index,
                            "prep": self._frame_diagnostic(refined.prep),
                            "layout": self._frame_diagnostic(refined.layout),
                            "battle_start": self._frame_diagnostic(refined.battle_start),
                            "end": self._frame_diagnostic(end),
                        }
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    self._record_issue(
                        candidate,
                        kind="window_error",
                        detail=str(exc),
                    )
        finally:
            if native_capture is not None:
                native_capture.release()
        issue_rounds = {issue.round_index for issue in self.issues}
        contact_sheets: list[dict[str, object]] = []
        for candidate in candidates:
            if not self.phase_debug and candidate.round_index not in issue_rounds:
                continue
            relative = (
                self.contact_sheet_prefix
                / f"{source.video_sha256[:12]}-round-{candidate.round_index:03d}.jpg"
            )
            try:
                self._write_contact_sheet(
                    video_path,
                    workspace / relative,
                    candidate,
                    source_fps,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                contact_sheets.append(
                    {
                        "round_index": candidate.round_index,
                        "error": str(exc),
                    }
                )
            else:
                contact_sheets.append(
                    {
                        "round_index": candidate.round_index,
                        "path": relative.as_posix(),
                    }
                )
        self.phase_diagnostics["issues"] = [asdict(issue) for issue in self.issues]
        self.phase_diagnostics["rejection_reasons"] = [issue.kind for issue in self.issues]
        self.phase_diagnostics["fine_frames"] = self._fine_frame_count
        self.phase_diagnostics["ocr_calls"] = self._ocr_call_count
        self.phase_diagnostics["selected_frames"] = selected_frames
        self.phase_diagnostics["contact_sheets"] = contact_sheets
        self.phase_diagnostics["output_rounds"] = len(samples)
        return samples

    @staticmethod
    def _frame_diagnostic(frame: TimedFrame) -> dict[str, float | int]:
        return {"frame_index": frame.frame_index, "timestamp": frame.timestamp}

    def _write_contact_sheet(
        self,
        video_path: Path,
        output_path: Path,
        candidate: RoundCandidate,
        source_fps: float,
    ) -> None:
        start = candidate.layout_search.start
        end = max(start, candidate.next_phase_start - 1.0 / source_fps)
        timestamps = np.linspace(start, end, num=12) if end > start else np.array([start])
        tiles: list[np.ndarray] = []
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video for contact sheet: {video_path}")
        try:
            for timestamp in timestamps:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    continue
                tile = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                cv2.rectangle(tile, (0, 0), (319, 25), (0, 0, 0), thickness=-1)
                cv2.putText(
                    tile,
                    f"round {candidate.round_index}  t={timestamp:.3f}s",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                tiles.append(tile)
        finally:
            capture.release()
        if not tiles:
            raise ValueError(f"cannot read contact-sheet frames from {video_path}")
        columns = 4
        rows = (len(tiles) + columns - 1) // columns
        sheet = np.zeros((rows * 180, columns * 320, 3), dtype=np.uint8)
        for index, tile in enumerate(tiles):
            row, column = divmod(index, columns)
            sheet[row * 180 : (row + 1) * 180, column * 320 : (column + 1) * 320] = tile
        self._write_image(output_path, sheet)

    def _refine_candidate(
        self,
        video_path: Path,
        candidate: RoundCandidate,
        source_fps: float,
        *,
        capture: cv2.VideoCapture | None = None,
    ) -> RefinedRound | None:
        guard = max(2.0 / source_fps, 0.2)
        if candidate.prep_span is not None:
            prep_search = TimeSpan(
                max(0.0, candidate.prep_span.start - guard),
                min(candidate.layout_search.end, candidate.prep_span.end + guard),
            )
        else:
            prep_end = (
                candidate.round_span.start
                if candidate.round_span is not None
                else candidate.layout_search.end
            )
            prep_search = TimeSpan(candidate.layout_search.start, max(candidate.layout_search.start, prep_end))
        prep_frames = self._analyze_interval(
            video_path,
            prep_search,
            source_fps,
            capture=capture,
        )
        prep = select_prep_frame(prep_frames)
        if prep is None:
            self._record_issue(
                candidate,
                kind="prep_unresolved",
                detail="no stable paired preparation UI at native frame rate",
                span=prep_search,
            )
            return None
        prep = self._materialize_frame(prep)
        del prep_frames

        transition_frames = self._analyze_interval(
            video_path,
            candidate.layout_search,
            source_fps,
            capture=capture,
        )
        layout = select_layout_frame(
            transition_frames,
            require_corner_transition=candidate.round_span is None,
        )
        if layout is None:
            self._record_issue(
                candidate,
                kind="layout_unresolved",
                detail="no two consecutive battlefield frames between countdown and ROUND",
                span=candidate.layout_search,
            )
            return None
        layout = self._materialize_frame(layout)
        del transition_frames

        battle_frames = self._analyze_interval(
            video_path,
            candidate.battle_start_search,
            source_fps,
            capture=capture,
        )
        battle_start = select_battle_start(battle_frames, reference=layout)
        if battle_start is None:
            self._record_issue(
                candidate,
                kind="battle_start_unresolved",
                detail="ROUND or corner transition never reached stable battle state",
                span=candidate.battle_start_search,
            )
            return None
        battle_start = self._materialize_frame(battle_start)
        del battle_frames

        end_frames = [
            frame
            for frame in self._analyze_interval(
                video_path,
                candidate.end_search,
                source_fps,
                include_end=False,
                capture=capture,
            )
            if frame.timestamp >= battle_start.timestamp
        ]
        end_run = select_end_run(end_frames)
        if not end_run and candidate.end_search.start > battle_start.timestamp:
            full_end_search = TimeSpan(battle_start.timestamp, candidate.next_phase_start)
            end_frames = self._analyze_interval(
                video_path,
                full_end_search,
                source_fps,
                include_end=False,
                capture=capture,
            )
            end_run = select_end_run(end_frames)
        if not end_run:
            self._record_issue(
                candidate,
                kind="end_unresolved",
                detail="no legal battlefield run before the next phase boundary",
                span=candidate.end_search,
            )
            return None
        end_run = tuple(self._materialize_frame(frame) for frame in end_run)

        reasons = tuple(reason for reason in candidate.failure_reasons if reason != "missing_prep")
        window = RoundWindow(
            round_index=candidate.round_index,
            prep_time=prep.timestamp,
            layout_time=layout.timestamp,
            battle_start=battle_start.timestamp,
            battle_end=end_run[-1].timestamp,
            complete=not reasons,
            failure_reasons=reasons,
        )
        return RefinedRound(
            window=window,
            prep=prep,
            layout=layout,
            battle_start=battle_start,
            end_run=end_run,
        )

    @staticmethod
    def _materialize_frame(frame: TimedFrame) -> TimedFrame:
        if not frame.encoded:
            return frame
        decoded = cv2.imdecode(frame.frame, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"cannot decode buffered source frame {frame.frame_index}")
        return replace(frame, frame=decoded, encoded=False)

    def _analyze_interval(
        self,
        video_path: Path,
        span: TimeSpan,
        source_fps: float,
        *,
        include_end: bool = True,
        capture: cv2.VideoCapture | None = None,
    ) -> list[TimedFrame]:
        return self._analyze_intervals(
            video_path,
            {"only": (span, include_end)},
            source_fps,
            capture=capture,
        )["only"]

    def _analyze_intervals(
        self,
        video_path: Path,
        intervals: dict[str, tuple[TimeSpan, bool]],
        source_fps: float,
        *,
        capture: cv2.VideoCapture | None = None,
    ) -> dict[str, list[TimedFrame]]:
        owns_capture = capture is None
        active_capture = capture or cv2.VideoCapture(str(video_path))
        if not active_capture.isOpened():
            if owns_capture:
                active_capture.release()
            raise ValueError(f"cannot open video: {video_path}")
        result: dict[str, list[TimedFrame]] = {}
        try:
            for name, (span, include_end) in intervals.items():
                frames: list[TimedFrame] = []
                for frame_index, timestamp, frame in self._decode_capture_interval(
                    active_capture,
                    span,
                    source_fps,
                    include_end=include_end,
                ):
                    signals = self.phase_analyzer.analyze(frame, timestamp)
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )
                    if not ok:
                        raise ValueError(f"cannot buffer source frame {frame_index}")
                    frames.append(
                        TimedFrame(
                            timestamp=timestamp,
                            frame_index=frame_index,
                            frame=encoded,
                            signals=signals,
                            encoded=True,
                        )
                    )
                self._fine_frame_count += len(frames)
                self._ocr_call_count += sum(frame.signals.ocr_performed for frame in frames)
                result[name] = frames
        finally:
            if owns_capture:
                active_capture.release()
        return result

    @staticmethod
    def _decode_native_interval(
        video_path: Path,
        span: TimeSpan,
        source_fps: float,
        *,
        include_end: bool,
    ):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        try:
            yield from VideoExtractor._decode_capture_interval(
                capture,
                span,
                source_fps,
                include_end=include_end,
            )
        finally:
            capture.release()

    @staticmethod
    def _decode_capture_interval(
        capture: cv2.VideoCapture,
        span: TimeSpan,
        source_fps: float,
        *,
        include_end: bool,
    ):
        start_frame = max(0, int(np.floor(span.start * source_fps + 1e-6)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame
        epsilon = 0.5 / source_fps
        while True:
            timestamp = frame_index / source_fps
            if timestamp > span.end + epsilon or (not include_end and timestamp >= span.end - epsilon):
                break
            ok, frame = capture.read()
            if not ok:
                break
            if timestamp >= span.start - epsilon:
                yield frame_index, timestamp, frame
            frame_index += 1

    def _record_issue(
        self,
        candidate: RoundCandidate,
        *,
        kind: str,
        detail: str,
        span: TimeSpan | None = None,
    ) -> None:
        self.issues.append(
            ExtractionIssue(
                round_index=candidate.round_index,
                observed_round_number=candidate.observed_round_number,
                kind=kind,
                detail=detail,
                start=span.start if span else None,
                end=span.end if span else None,
            )
        )

    def _extract_window(
        self,
        video_path: Path,
        source: SourceRef,
        workspace: Path,
        refined: RefinedRound,
    ) -> RoundSample:
        window = refined.window
        assert window.prep_time is not None
        assert window.layout_time is not None
        assert window.battle_start is not None

        prep_frame = refined.prep.frame
        prep_frames = [prep_frame]
        recognize_batch = getattr(self.roster_recognizer, "recognize_batch", None)
        if callable(recognize_batch):
            observations = [item for frame_items in recognize_batch(prep_frames) for item in frame_items]
        else:
            observations = [item for frame in prep_frames for item in self.roster_recognizer.recognize(frame)]
        rosters = fuse_roster_observations(observations)
        candidate_enemy_ids = {
            entry.enemy_id
            for side_roster in rosters.values()
            for entry in side_roster
            if entry.enemy_id > 0
        }

        layout_frame = refined.layout.frame
        try:
            detections = self.battlefield_detector.detect(
                layout_frame,
                candidate_enemy_ids=candidate_enemy_ids or None,
            )
        except TypeError:
            detections = self.battlefield_detector.detect(layout_frame)

        left_eids = {entry.enemy_id for entry in rosters.get("left", []) if entry.enemy_id > 0}
        right_eids = {entry.enemy_id for entry in rosters.get("right", []) if entry.enemy_id > 0}

        survivor_res = self.survivor_detector.detect_winner(
            end_run=refined.end_run,
            left_eids=left_eids,
            right_eids=right_eids,
        )
        winner: Winner | None = survivor_res.winner
        winner_confidence: float | None = survivor_res.confidence
        end_frame: np.ndarray = survivor_res.evidence_frame
        window = replace(window, battle_end=survivor_res.evidence_time)

        if winner is None and self.health_detector is not None:
            tracker = WinnerTracker(stable_frames=self.stable_winner_frames)
            battle_gen = self._sample_video(video_path, start=window.battle_start, end=window.battle_end)
            try:
                for _, frame in battle_gen:
                    counts = self.health_detector.count(frame)
                    resolved = tracker.update(orange=counts.orange, blue=counts.blue)
                    if resolved is not None:
                        winner = resolved
                        winner_confidence = tracker.confidence
                        break
            finally:
                with contextlib.suppress(Exception):
                    battle_gen.close()

        sample_id = stable_sample_id(source, window.round_index, window.layout_time)
        relative_dir = self.evidence_prefix / source.video_sha256[:12]
        filenames = {
            "prep": relative_dir / f"{sample_id}-prep.jpg",
            "layout": relative_dir / f"{sample_id}-layout.jpg",
            "end": relative_dir / f"{sample_id}-end.jpg",
        }
        self._write_image(workspace / filenames["prep"], prep_frame)
        self._write_image(workspace / filenames["layout"], layout_frame)
        self._write_image(workspace / filenames["end"], end_frame)

        return assemble_round_sample(
            source=source,
            window=window,
            rosters=rosters,
            detections=detections,
            winner=winner,
            winner_confidence=winner_confidence,
            evidence={key: value.as_posix() for key, value in filenames.items()},
        )

    @staticmethod
    def _video_properties(video_path: Path) -> tuple[float, float]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if fps <= 0.0 or frames <= 0.0:
            raise ValueError(f"video reports invalid timing metadata: {video_path}")
        return frames / fps, fps

    @staticmethod
    def _video_resolution(video_path: Path) -> tuple[int, int]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        try:
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:
            capture.release()
        if width <= 0 or height <= 0:
            raise ValueError(f"video reports invalid resolution: {video_path}")
        return width, height

    def _sample_video_nvdec(
        self,
        video_path: Path,
        *,
        start: float,
        end: float | None,
    ):
        dim_proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height",
                "-of",
                "csv=p=0:s=x",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
        )
        parts = dim_proc.stdout.strip().split("x")
        if len(parts) != 3 or parts[0] != "h264":
            raise ValueError(f"video is not h264 for NVDEC: {dim_proc.stdout}")
        width, height = int(parts[1]), int(parts[2])

        cmd = ["ffmpeg", "-hwaccel", "cuda", "-c:v", "h264_cuvid"]
        if start > 0.0:
            cmd.extend(["-ss", f"{start:.3f}"])
        if end is not None:
            cmd.extend(["-to", f"{end:.3f}"])
        cmd.extend(
            [
                "-i",
                str(video_path),
                "-vf",
                f"fps={self.scan_fps}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ]
        )

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=width * height * 3 * 10,
        )
        frame_bytes = width * height * 3
        frame_index = 0
        try:
            assert proc.stdout is not None
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
                timestamp = start + frame_index / self.scan_fps
                if end is not None and timestamp > end:
                    break
                yield timestamp, frame
                frame_index += 1
        finally:
            if proc.stdout is not None:
                with contextlib.suppress(Exception):
                    proc.stdout.close()
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=0.5)

    def _sample_video(
        self,
        video_path: Path,
        *,
        start: float,
        end: float | None,
    ):
        nvdec_ok = False
        nvdec_gen = None
        try:
            nvdec_gen = self._sample_video_nvdec(video_path, start=start, end=end)
            for item in nvdec_gen:
                nvdec_ok = True
                yield item
            if nvdec_ok:
                return
        except Exception:
            pass
        finally:
            if nvdec_gen is not None:
                with contextlib.suppress(Exception):
                    nvdec_gen.close()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            capture.release()
            raise ValueError(f"video reports invalid FPS: {video_path}")
        step = max(1, round(fps / self.scan_fps))
        start_frame = max(0, round(start * fps))
        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / fps
                if end is not None and timestamp > end:
                    break
                yield timestamp, frame
                for _ in range(step - 1):
                    if not capture.grab():
                        break
                frame_index += step
        finally:
            capture.release()

    @staticmethod
    def _read_frame(video_path: Path, timestamp: float) -> np.ndarray:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"cannot read {video_path} at {timestamp:.3f}s")
            return frame
        finally:
            capture.release()

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError(f"cannot encode evidence image: {path}")
        encoded.tofile(path)


def extract_rounds(
    input_dir: Path,
    workspace: Path,
    *,
    device: str = "0",
    half: bool = True,
    batch_size: int = 32,
    scan_fps: float | None = None,
    max_videos: int | None = None,
    force: bool = False,
    reset_review: bool = False,
    phase_debug: bool = False,
    workers: int = 8,
) -> list[RoundSample]:
    """Extract Green Vine videos into an immutable run and atomically promote it."""

    if reset_review and not force:
        raise ValueError("--reset-review requires --force")
    if reset_review and max_videos is not None:
        raise ValueError("--reset-review cannot be combined with --max-videos")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    import typer

    from maa_duel.config import PipelineConfig
    from maa_duel.store import read_jsonl, write_jsonl
    from maa_duel.training.vision import YoloBattlefieldDetector, YoloCountClassifier, YoloPortraitClassifier
    from maa_duel.video.inventory import Arena, VideoRecord, scan_inventory
    from maa_duel.vision.health import HealthBarDetector
    from maa_duel.vision.ocr import RapidOcrEngine
    from maa_duel.vision.phase_analyzer import OcrPhaseAnalyzer
    from maa_duel.vision.roster import DualEngineClassifier, RosterFrameRecognizer, TemplateMatchClassifier

    roster_model = workspace / "models" / "vision" / "roster" / "weights" / "best.pt"
    battlefield_model = workspace / "models" / "vision" / "battlefield" / "weights" / "best.pt"
    ocr_model = workspace / "models" / "vision" / "ocr" / "weights" / "best.pt"
    ocr_class_map = workspace / "models" / "vision" / "ocr" / "class-map.json"
    roster_class_map = workspace / "models" / "vision" / "roster" / "class-map.json"
    battlefield_class_map = workspace / "models" / "vision" / "battlefield" / "class-map.json"
    if not roster_class_map.is_file():
        roster_class_map = workspace / "synthetic" / "roster" / "class-map.json"
    if not battlefield_class_map.is_file():
        battlefield_class_map = workspace / "synthetic" / "battlefield" / "class-map.json"
    required = (
        ("roster model", roster_model),
        ("roster class map", roster_class_map),
        ("battlefield model", battlefield_model),
        ("battlefield class map", battlefield_class_map),
    )
    for label, path in required:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    config = PipelineConfig(input_dir=input_dir, workspace_dir=workspace)
    config.ensure_workspace()
    video_manifest = config.manifest_dir / "videos.jsonl"
    records = (
        read_jsonl(video_manifest, VideoRecord)
        if video_manifest.exists()
        else scan_inventory(input_dir, video_manifest)
    )
    fps = scan_fps if scan_fps is not None else config.sample_fps
    portraits_dir = workspace / "assets" / "portraits"
    empty_slot_path = workspace / "assets" / "ui" / "empty_slot.png"
    active = active_extraction_paths(workspace)
    staged = create_staged_extraction(workspace)

    def make_extractor() -> VideoExtractor:
        ocr_local = RapidOcrEngine(use_cuda=device != "cpu")
        count_classifier_local = (
            YoloCountClassifier(
                ocr_model,
                ocr_class_map,
                device=device,
                half=half,
                batch_size=batch_size,
            )
            if ocr_model.is_file() and ocr_class_map.is_file()
            else None
        )
        yolo_portrait_local = YoloPortraitClassifier(
            roster_model,
            roster_class_map,
            device=device,
            half=half,
            batch_size=batch_size,
        )
        if portraits_dir.is_dir():
            template_classifier_local = TemplateMatchClassifier(
                portraits_dir,
                empty_slot_path=empty_slot_path if empty_slot_path.is_file() else None,
            )
            portrait_classifier_local = DualEngineClassifier(yolo_portrait_local, template_classifier_local)
        else:
            portrait_classifier_local = yolo_portrait_local

        phase_analyzer_local = OcrPhaseAnalyzer(ocr_local)
        battlefield_detector_local = YoloBattlefieldDetector(
            battlefield_model,
            battlefield_class_map,
            device=device,
            half=half,
            batch_size=batch_size,
        )
        survivor_detector_local = BattlefieldSurvivorDetector(battlefield_detector_local)

        return VideoExtractor(
            phase_analyzer=phase_analyzer_local,
            roster_recognizer=RosterFrameRecognizer(
                portrait_classifier_local,
                ocr_local,
                count_classifier=count_classifier_local,
                default_count=1,
            ),
            battlefield_detector=battlefield_detector_local,
            survivor_detector=survivor_detector_local,
            health_detector=HealthBarDetector(),
            scan_fps=fps,
            stable_winner_frames=config.stable_winner_frames,
            evidence_prefix=staged.evidence_prefix,
            phase_debug=phase_debug,
            contact_sheet_prefix=(
                Path("extraction-runs") / staged.run_id / "reports" / "phase-contact-sheets"
            ),
        )

    samples: list[RoundSample] = []
    already_extracted_videos: set[str] = set()
    errors: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    if not force and active.rounds_manifest.is_file():
        previous = read_jsonl(active.rounds_manifest, RoundSample)
        samples = _carry_forward_samples(workspace, staged, previous)
        already_extracted_videos = {item.source.video_relpath for item in samples}
        previous_errors = _load_json(active.extraction_errors, [])
        if not isinstance(previous_errors, list):
            raise ValueError(f"invalid extraction errors report: {active.extraction_errors}")
        errors.extend(item for item in previous_errors if isinstance(item, dict))
        previous_diagnostics = _load_json(active.phase_diagnostics, {})
        if isinstance(previous_diagnostics, dict):
            previous_videos = previous_diagnostics.get("videos", [])
            if isinstance(previous_videos, list):
                diagnostics.extend(item for item in previous_videos if isinstance(item, dict))

    _copy_review_store(active.corrections, staged.paths.corrections, reset=reset_review)
    write_jsonl(staged.paths.rounds_manifest, samples)

    gv_records = [record for record in records if record.arena is Arena.GREEN_VINE]
    gv_records.sort(key=lambda item: (item.duration > 300, item.duration, item.relative_path.casefold()))
    if max_videos is not None:
        gv_records = gv_records[:max_videos]

    pending_records = [
        (idx, record)
        for idx, record in enumerate(gv_records, start=1)
        if record.relative_path not in already_extracted_videos
    ]

    typer.echo(
        f"Extracting {len(gv_records)} Green Vine videos at {fps:.1f} fps using {workers} worker(s) "
        f"(already completed: {len(already_extracted_videos)}, pending: {len(pending_records)})..."
    )

    tls = threading.local()

    def get_thread_extractor() -> VideoExtractor:
        if not hasattr(tls, "extractor"):
            tls.extractor = make_extractor()
        return tls.extractor

    def process_record(
        record_item: tuple[int, VideoRecord],
    ) -> tuple[list[RoundSample], list[dict[str, object]], dict[str, object]]:
        index, record = record_item
        extractor = get_thread_extractor()
        path = input_dir / Path(record.relative_path)
        source = SourceRef(video_relpath=record.relative_path, video_sha256=record.sha256)
        typer.echo(f"[{index}/{len(gv_records)}] Starting {record.relative_path} ({record.duration:.1f}s)...")
        try:
            extracted = extractor.extract_video(
                path,
                source,
                staged.staging_workspace,
                duration=record.duration,
                source_fps=record.fps,
            )
            typer.echo(
                f"[{index}/{len(gv_records)}] Done {record.relative_path} -> "
                f"{len(extracted)} rounds (issues: {len(extractor.issues)})"
            )
            local_errors = [
                {
                    "video": record.relative_path,
                    "round_index": str(issue.round_index),
                    "kind": issue.kind,
                    "error": issue.detail,
                    "observed_round_number": issue.observed_round_number,
                    "start": issue.start,
                    "end": issue.end,
                }
                for issue in extractor.issues
            ]
            return extracted, local_errors, dict(extractor.phase_diagnostics)
        except (OSError, ValueError, RuntimeError) as exc:
            typer.echo(f"[{index}/{len(gv_records)}] Error {record.relative_path}: {exc}", err=True)
            diagnostic = {
                "video": record.relative_path,
                "video_sha256": record.sha256,
                "duration": record.duration,
                "source_fps": record.fps,
                "fatal_error": str(exc),
            }
            return [], [{"video": record.relative_path, "kind": "video_error", "error": str(exc)}], diagnostic

    fatal_errors: list[dict[str, object]] = []

    def collect(
        extracted: list[RoundSample],
        local_errors: list[dict[str, object]],
        diagnostic: dict[str, object],
    ) -> None:
        samples.extend(extracted)
        errors.extend(local_errors)
        diagnostics.append(diagnostic)
        fatal_errors.extend(error for error in local_errors if error.get("kind") == "video_error")
        samples.sort(key=lambda s: (s.source.video_relpath.casefold(), s.round_index, s.timestamps.layout))
        write_jsonl(staged.paths.rounds_manifest, samples)

    if workers <= 1:
        for item in pending_records:
            collect(*process_record(item))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_record, item) for item in pending_records]
            for future in concurrent.futures.as_completed(futures):
                collect(*future.result())

    samples.sort(key=lambda item: (item.source.video_relpath.casefold(), item.round_index, item.timestamps.layout))
    errors.sort(key=lambda item: (str(item.get("video", "")).casefold(), str(item.get("round_index", ""))))
    diagnostics.sort(key=lambda item: str(item.get("video", "")).casefold())
    write_jsonl(staged.paths.rounds_manifest, samples)
    _write_json(staged.paths.extraction_errors, errors)
    _write_json(
        staged.paths.phase_diagnostics,
        {
            "schema_version": 1,
            "run_id": staged.run_id,
            "coarse_scan_fps": fps,
            "videos": diagnostics,
        },
    )
    if fatal_errors:
        raise RuntimeError(
            f"{len(fatal_errors)} video(s) failed; staged extraction was retained and the active run was not changed"
        )
    publish_staged_extraction(workspace, staged)
    typer.echo(f"Promoted extraction run {staged.run_id} with {len(samples)} round(s).")
    return samples
