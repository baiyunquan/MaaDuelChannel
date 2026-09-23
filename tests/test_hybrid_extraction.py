import json
import time
from pathlib import Path

import pytest

from maa_duel.extraction_state import create_staged_extraction, publish_staged_extraction
from maa_duel.schema import EvidenceFrames, RoundSample, SideData, SourceRef, Timestamps
from maa_duel.store import write_jsonl
from maa_duel.video.inventory import Arena, VideoRecord


def _prepare_workspace(tmp_path: Path, durations: list[float]) -> tuple[Path, Path]:
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    workspace = tmp_path / "workspace"
    manifest = workspace / "manifests" / "videos.jsonl"
    manifest.parent.mkdir(parents=True)
    write_jsonl(
        manifest,
        [
            VideoRecord(
                relative_path=f"video-{index:02}.mp4",
                sha256=f"{index:064x}",
                size_bytes=1,
                mtime_ns=1,
                width=1920,
                height=864,
                duration=duration,
                fps=30.0,
                arena=Arena.GREEN_VINE,
            )
            for index, duration in enumerate(durations, start=1)
        ],
    )
    for name in ("roster", "battlefield"):
        root = workspace / "models" / "vision" / name
        (root / "weights").mkdir(parents=True)
        (root / "weights" / "best.pt").touch()
        (root / "class-map.json").write_text("{}")
    return input_dir, workspace


def test_extract_routes_longest_video_to_one_cuda_ocr_worker(tmp_path, monkeypatch):
    from maa_duel import extraction as extraction_module
    from maa_duel.training import vision as vision_training
    from maa_duel.vision import ocr as ocr_module

    input_dir, workspace = _prepare_workspace(tmp_path, [10, 100, 20, 80, 30, 50, 40, 60, 70])
    ocr_instances: list[tuple[bool, int]] = []

    class FakeOcr:
        def __init__(self, *, use_cuda: bool, cpu_threads: int = 2) -> None:
            ocr_instances.append((use_cuda, cpu_threads))
            self.use_cuda = use_cuda
            self._providers = (
                ("CUDAExecutionProvider", "CPUExecutionProvider") if use_cuda else ("CPUExecutionProvider",)
            )

    class FakeExtractor:
        def __init__(self, *, phase_analyzer, **kwargs) -> None:
            self.phase_analyzer = phase_analyzer
            self.backend = "cuda" if phase_analyzer.ocr.use_cuda else "cpu"
            self.issues = []

        def extract_video(self, video_path, source, workspace, *, duration, source_fps):
            time.sleep(0.02)
            self.phase_diagnostics = {"video": source.video_relpath, "ocr_calls": 3}
            return []

    monkeypatch.setattr(ocr_module, "RapidOcrEngine", FakeOcr)
    monkeypatch.setattr(extraction_module, "VideoExtractor", FakeExtractor)
    monkeypatch.setattr(vision_training, "YoloPortraitClassifier", lambda *args, **kwargs: object())
    monkeypatch.setattr(vision_training, "YoloBattlefieldDetector", lambda *args, **kwargs: object())

    samples = extraction_module.extract_rounds(
        input_dir,
        workspace,
        force=True,
        reset_review=True,
        workers=3,
    )

    assert samples == []
    assert sum(use_cuda for use_cuda, _ in ocr_instances) == 1
    assert all(threads == 2 for use_cuda, threads in ocr_instances if not use_cuda)
    pointer = json.loads((workspace / "manifests" / "extraction.active.json").read_text())
    diagnostics_path = workspace / "extraction-runs" / pointer["run_id"] / "reports" / "phase-diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())["videos"]
    assert len(diagnostics) == 9
    by_video = {item["video"]: item for item in diagnostics}
    assert by_video["video-02.mp4"]["ocr_backend"] == "cuda"
    assert any(item["ocr_backend"] == "cpu" for item in diagnostics)
    assert all(item["ocr_providers"] for item in diagnostics)
    assert all(item["elapsed_seconds"] >= 0 for item in diagnostics)

    # The existing --max-videos smoke behavior selects the shortest source first.
    extraction_module.extract_rounds(input_dir, workspace, force=True, max_videos=1, workers=3)
    pointer = json.loads((workspace / "manifests" / "extraction.active.json").read_text())
    diagnostics_path = workspace / "extraction-runs" / pointer["run_id"] / "reports" / "phase-diagnostics.json"
    limited = json.loads(diagnostics_path.read_text())["videos"]
    assert [(item["video"], item["ocr_backend"]) for item in limited] == [("video-01.mp4", "cuda")]

    previous_gpu_count = sum(use_cuda for use_cuda, _ in ocr_instances)
    extraction_module.extract_rounds(input_dir, workspace, device="cpu", force=True, max_videos=2, workers=3)
    pointer = json.loads((workspace / "manifests" / "extraction.active.json").read_text())
    diagnostics_path = workspace / "extraction-runs" / pointer["run_id"] / "reports" / "phase-diagnostics.json"
    cpu_only = json.loads(diagnostics_path.read_text())["videos"]
    assert len(cpu_only) == 2
    assert all(item["ocr_backend"] == "cpu" for item in cpu_only)
    assert sum(use_cuda for use_cuda, _ in ocr_instances) == previous_gpu_count


def test_cuda_ocr_initialization_failure_does_not_promote_run(tmp_path, monkeypatch):
    from maa_duel import extraction as extraction_module
    from maa_duel.vision import ocr as ocr_module

    input_dir, workspace = _prepare_workspace(tmp_path, [10, 20])
    attempts = []

    class FailingOcr:
        def __init__(self, *, use_cuda: bool, cpu_threads: int = 2) -> None:
            attempts.append(use_cuda)
            if use_cuda:
                raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(ocr_module, "RapidOcrEngine", FailingOcr)

    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        extraction_module.extract_rounds(input_dir, workspace, force=True, reset_review=True, workers=2)

    assert attempts == [True]
    assert not (workspace / "manifests" / "extraction.active.json").exists()


def test_resume_only_schedules_videos_without_carried_samples(tmp_path, monkeypatch):
    from maa_duel import extraction as extraction_module
    from maa_duel.training import vision as vision_training
    from maa_duel.vision import ocr as ocr_module

    input_dir, workspace = _prepare_workspace(tmp_path, [10, 100])
    staged = create_staged_extraction(workspace, run_id="seed-run")
    prefix = staged.evidence_prefix / "seed"
    evidence = EvidenceFrames(**{label: (prefix / f"{label}.jpg").as_posix() for label in ("prep", "layout", "end")})
    carried = RoundSample(
        sample_id="a" * 32,
        source=SourceRef(video_relpath="video-02.mp4", video_sha256=f"{2:064x}"),
        round_index=1,
        timestamps=Timestamps(prep=1.0, layout=2.0, battle_start=3.0, battle_end=4.0),
        evidence=evidence,
        left=SideData(),
        right=SideData(),
        pipeline_version="0.2.0",
    )
    write_jsonl(staged.paths.rounds_manifest, [carried])
    staged.paths.corrections.write_text("")
    for relative in (evidence.prep, evidence.layout, evidence.end):
        assert relative is not None
        path = staged.staging_workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    publish_staged_extraction(workspace, staged)

    class FakeOcr:
        def __init__(self, *, use_cuda: bool, cpu_threads: int = 2) -> None:
            self._providers = (
                ("CUDAExecutionProvider", "CPUExecutionProvider") if use_cuda else ("CPUExecutionProvider",)
            )

    class FakeExtractor:
        def __init__(self, *, phase_analyzer, **kwargs) -> None:
            self.phase_analyzer = phase_analyzer
            self.issues = []

        def extract_video(self, video_path, source, workspace, *, duration, source_fps):
            self.phase_diagnostics = {"video": source.video_relpath, "ocr_calls": 1}
            return []

    monkeypatch.setattr(ocr_module, "RapidOcrEngine", FakeOcr)
    monkeypatch.setattr(extraction_module, "VideoExtractor", FakeExtractor)
    monkeypatch.setattr(vision_training, "YoloPortraitClassifier", lambda *args, **kwargs: object())
    monkeypatch.setattr(vision_training, "YoloBattlefieldDetector", lambda *args, **kwargs: object())

    samples = extraction_module.extract_rounds(input_dir, workspace, workers=2)

    assert [sample.source.video_relpath for sample in samples] == ["video-02.mp4"]
    pointer = json.loads((workspace / "manifests" / "extraction.active.json").read_text())
    diagnostics_path = workspace / "extraction-runs" / pointer["run_id"] / "reports" / "phase-diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())["videos"]
    assert [item["video"] for item in diagnostics] == ["video-01.mp4"]
    assert diagnostics[0]["ocr_backend"] == "cuda"
