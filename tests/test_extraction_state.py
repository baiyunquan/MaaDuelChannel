from pathlib import Path

import pytest

from maa_duel.extraction_state import (
    active_extraction_paths,
    create_staged_extraction,
    publish_staged_extraction,
)
from maa_duel.schema import EvidenceFrames, RoundSample, SideData, SourceRef, Timestamps
from maa_duel.store import write_jsonl


def sample(run_id: str, *, with_missing_evidence: bool = False) -> RoundSample:
    prefix = Path("extraction-runs") / run_id / "frames" / "abc"
    return RoundSample(
        sample_id="a" * 32,
        source=SourceRef(video_relpath="green/source.mp4", video_sha256="b" * 64),
        round_index=1,
        timestamps=Timestamps(prep=1.0, layout=2.0, battle_start=3.0, battle_end=4.0),
        evidence=EvidenceFrames(
            prep=(prefix / "prep.jpg").as_posix(),
            layout=(prefix / "layout.jpg").as_posix(),
            end=(prefix / ("missing.jpg" if with_missing_evidence else "end.jpg")).as_posix(),
        ),
        left=SideData(),
        right=SideData(),
        pipeline_version="0.2.0",
    )


def write_staged_sample(workspace: Path, run_id: str, *, missing: bool = False):
    staged = create_staged_extraction(workspace, run_id=run_id)
    row = sample(run_id, with_missing_evidence=missing)
    write_jsonl(staged.paths.rounds_manifest, [row])
    write_jsonl(staged.paths.corrections, [])
    for relative in (row.evidence.prep, row.evidence.layout, row.evidence.end):
        if missing and relative and relative.endswith("missing.jpg"):
            continue
        assert relative is not None
        path = staged.staging_workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    return staged


def test_active_paths_fall_back_to_legacy_workspace_without_pointer(tmp_path) -> None:
    paths = active_extraction_paths(tmp_path)

    assert paths.run_id is None
    assert paths.rounds_manifest == tmp_path / "manifests" / "rounds.auto.jsonl"
    assert paths.corrections == tmp_path / "review" / "corrections.jsonl"
    assert paths.extraction_errors == tmp_path / "reports" / "extraction-errors.json"


def test_publish_switches_all_active_paths_with_one_pointer(tmp_path) -> None:
    legacy_corrections = tmp_path / "review" / "corrections.jsonl"
    legacy_corrections.parent.mkdir(parents=True)
    legacy_corrections.write_text("old-review\n", encoding="utf-8")
    staged = write_staged_sample(tmp_path, "20260923T120000000000Z-a1b2c3d4")

    published = publish_staged_extraction(tmp_path, staged)
    active = active_extraction_paths(tmp_path)

    assert active == published
    assert active.run_id == staged.run_id
    assert active.rounds_manifest.is_file()
    assert active.corrections.is_file()
    assert (tmp_path / "extraction-runs" / staged.run_id / "frames" / "abc" / "end.jpg").is_file()
    assert legacy_corrections.read_text(encoding="utf-8") == "old-review\n"


def test_unpublished_staging_does_not_change_active_legacy_paths(tmp_path) -> None:
    staged = write_staged_sample(tmp_path, "20260923T120000000000Z-deadbeef")

    active = active_extraction_paths(tmp_path)

    assert active.run_id is None
    assert staged.staged_run_root.is_dir()
    assert not staged.final_run_root.exists()


def test_publish_rejects_manifest_with_missing_evidence_before_switch(tmp_path) -> None:
    staged = write_staged_sample(tmp_path, "20260923T120000000000Z-bad0bad0", missing=True)

    with pytest.raises(ValueError, match="missing evidence"):
        publish_staged_extraction(tmp_path, staged)

    assert active_extraction_paths(tmp_path).run_id is None
    assert not staged.final_run_root.exists()


def test_invalid_active_pointer_never_silently_falls_back_to_legacy(tmp_path) -> None:
    pointer = tmp_path / "manifests" / "extraction.active.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text('{"schema_version":1,"run_id":"missing-run"}', encoding="utf-8")

    with pytest.raises(ValueError, match="does not exist"):
        active_extraction_paths(tmp_path)
