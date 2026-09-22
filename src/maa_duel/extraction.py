from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

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
from maa_duel.video.phases import FrameSignals, RoundSegmenter, RoundWindow
from maa_duel.vision.health import HealthCounts, WinnerTracker
from maa_duel.vision.layout import RawDetection, reconcile_detections
from maa_duel.vision.roster import RosterObservation, fuse_roster_observations


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
    pipeline_version: str = "0.1.0",
) -> RoundSample:
    if None in (window.prep_time, window.layout_time, window.battle_start):
        raise ValueError("round window must contain prep, layout, and battle timestamps")

    reconciled = reconcile_detections(
        left_roster=rosters.get("left", []),
        right_roster=rosters.get("right", []),
        detections=detections,
    )
    reasons = list(window.failure_reasons) + list(reconciled.reasons)
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
    accepted = window.complete and reconciled.accepted and winner is not None and high_confidence
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
        health_detector: HealthDetector,
        scan_fps: float = 10.0,
        stable_winner_frames: int = 5,
    ) -> None:
        self.phase_analyzer = phase_analyzer
        self.roster_recognizer = roster_recognizer
        self.battlefield_detector = battlefield_detector
        self.health_detector = health_detector
        self.scan_fps = scan_fps
        self.stable_winner_frames = stable_winner_frames
        self.issues: list[ExtractionIssue] = []

    def extract_video(self, video_path: Path, source: SourceRef, workspace: Path) -> list[RoundSample]:
        self.issues = []
        signals = [
            self.phase_analyzer.analyze(frame, timestamp)
            for timestamp, frame in self._sample_video(video_path, start=0.0, end=None)
        ]
        windows = RoundSegmenter().segment(signals)
        samples: list[RoundSample] = []
        for window in windows:
            if not window.complete:
                self.issues.append(
                    ExtractionIssue(
                        round_index=window.round_index,
                        kind="incomplete_window",
                        detail=",".join(window.failure_reasons),
                    )
                )
                continue
            try:
                samples.append(self._extract_window(video_path, source, workspace, window))
            except (OSError, ValueError, RuntimeError) as exc:
                self.issues.append(
                    ExtractionIssue(
                        round_index=window.round_index,
                        kind="window_error",
                        detail=str(exc),
                    )
                )
        return samples

    def _extract_window(
        self,
        video_path: Path,
        source: SourceRef,
        workspace: Path,
        window: RoundWindow,
    ) -> RoundSample:
        assert window.prep_time is not None
        assert window.layout_time is not None
        assert window.battle_start is not None

        prep_times = sorted(
            {
                max(0.0, window.prep_time - 0.2),
                window.prep_time,
                min(window.layout_time - 0.05, window.prep_time + 0.2),
            }
        )
        prep_frames = [self._read_frame(video_path, timestamp) for timestamp in prep_times]
        prep_frame = prep_frames[-1]
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

        layout_frame = self._read_frame(video_path, window.layout_time)
        try:
            detections = self.battlefield_detector.detect(
                layout_frame,
                candidate_enemy_ids=candidate_enemy_ids or None,
            )
        except TypeError:
            detections = self.battlefield_detector.detect(layout_frame)

        tracker = WinnerTracker(stable_frames=self.stable_winner_frames)
        winner: Winner | None = None
        end_frame = self._read_frame(video_path, window.battle_end)
        for _, frame in self._sample_video(video_path, start=window.battle_start, end=window.battle_end):
            counts = self.health_detector.count(frame)
            resolved = tracker.update(orange=counts.orange, blue=counts.blue)
            if resolved is not None:
                winner = resolved
                end_frame = frame
                break

        sample_id = stable_sample_id(source, window.round_index, window.layout_time)
        relative_dir = Path("frames") / source.video_sha256[:12]
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
            winner_confidence=tracker.confidence if winner is not None else None,
            evidence={key: value.as_posix() for key, value in filenames.items()},
        )

    def _sample_video(
        self,
        video_path: Path,
        *,
        start: float,
        end: float | None,
    ):
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
) -> list[RoundSample]:
    """Run extraction for every Green Vine video and replace the automatic manifest."""

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
    ocr = RapidOcrEngine(use_cuda=device != "cpu")
    count_classifier = (
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
    yolo_portrait_classifier = YoloPortraitClassifier(
        roster_model,
        roster_class_map,
        device=device,
        half=half,
        batch_size=batch_size,
    )
    portraits_dir = workspace / "assets" / "portraits"
    empty_slot_path = workspace / "assets" / "ui" / "empty_slot.png"
    if portraits_dir.is_dir():
        template_classifier = TemplateMatchClassifier(
            portraits_dir,
            empty_slot_path=empty_slot_path if empty_slot_path.is_file() else None,
        )
        portrait_classifier = DualEngineClassifier(yolo_portrait_classifier, template_classifier)
    else:
        portrait_classifier = yolo_portrait_classifier

    fps = scan_fps if scan_fps is not None else config.sample_fps
    extractor = VideoExtractor(
        phase_analyzer=OcrPhaseAnalyzer(ocr),
        roster_recognizer=RosterFrameRecognizer(
            portrait_classifier,
            ocr,
            count_classifier=count_classifier,
            default_count=1,
        ),
        battlefield_detector=YoloBattlefieldDetector(
            battlefield_model,
            battlefield_class_map,
            device=device,
            half=half,
            batch_size=batch_size,
        ),
        health_detector=HealthBarDetector(),
        scan_fps=fps,
        stable_winner_frames=config.stable_winner_frames,
    )
    rounds_manifest = config.manifest_dir / "rounds.auto.jsonl"
    samples: list[RoundSample] = []
    already_extracted_videos: set[str] = set()
    if rounds_manifest.is_file():
        try:
            samples = read_jsonl(rounds_manifest, RoundSample)
            already_extracted_videos = {item.source.video_relpath for item in samples}
        except Exception:
            samples = []

    errors: list[dict[str, str]] = []
    gv_records = [record for record in records if record.arena is Arena.GREEN_VINE]
    gv_records.sort(key=lambda item: (item.duration > 300, item.duration, item.relative_path.casefold()))
    if max_videos is not None:
        gv_records = gv_records[:max_videos]

    typer.echo(
        f"Extracting {len(gv_records)} Green Vine videos at {fps:.1f} fps "
        f"(already completed: {len(already_extracted_videos)})..."
    )
    for index, record in enumerate(gv_records, start=1):
        if record.relative_path in already_extracted_videos:
            continue
        path = input_dir / Path(record.relative_path)
        source = SourceRef(video_relpath=record.relative_path, video_sha256=record.sha256)
        typer.echo(f"[{index}/{len(gv_records)}] {record.relative_path} ({record.duration:.1f}s)...")
        try:
            extracted = extractor.extract_video(path, source, workspace)
            samples.extend(extracted)
            typer.echo(f"  -> Extracted {len(extracted)} rounds (issues: {len(extractor.issues)})")
            if extracted:
                samples.sort(
                    key=lambda item: (item.source.video_relpath.casefold(), item.round_index, item.timestamps.layout)
                )
                write_jsonl(config.manifest_dir / "rounds.auto.jsonl", samples)
            errors.extend(
                {
                    "video": record.relative_path,
                    "round_index": str(issue.round_index),
                    "kind": issue.kind,
                    "error": issue.detail,
                }
                for issue in extractor.issues
            )
        except (OSError, ValueError, RuntimeError) as exc:
            typer.echo(f"  -> Video error: {exc}", err=True)
            errors.append({"video": record.relative_path, "kind": "video_error", "error": str(exc)})

    samples.sort(key=lambda item: (item.source.video_relpath.casefold(), item.round_index, item.timestamps.layout))
    write_jsonl(config.manifest_dir / "rounds.auto.jsonl", samples)
    error_path = config.report_dir / "extraction-errors.json"
    import json

    error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    return samples
