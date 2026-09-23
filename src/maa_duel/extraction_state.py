from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from maa_duel.schema import RoundSample
from maa_duel.store import read_jsonl


class ActiveExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class ExtractionPaths:
    run_id: str | None
    root: Path
    rounds_manifest: Path
    corrections: Path
    extraction_errors: Path
    phase_diagnostics: Path


@dataclass(frozen=True)
class StagedExtraction:
    run_id: str
    staging_workspace: Path
    staged_run_root: Path
    final_run_root: Path
    evidence_prefix: Path
    paths: ExtractionPaths


def active_extraction_paths(workspace: Path) -> ExtractionPaths:
    workspace = Path(workspace)
    pointer = workspace / "manifests" / "extraction.active.json"
    if not pointer.exists():
        return ExtractionPaths(
            run_id=None,
            root=workspace,
            rounds_manifest=workspace / "manifests" / "rounds.auto.jsonl",
            corrections=workspace / "review" / "corrections.jsonl",
            extraction_errors=workspace / "reports" / "extraction-errors.json",
            phase_diagnostics=workspace / "reports" / "phase-diagnostics.json",
        )
    try:
        active = ActiveExtraction.model_validate_json(pointer.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"invalid active extraction pointer: {pointer}") from exc
    paths = _run_paths(workspace, active.run_id)
    if not paths.root.is_dir():
        raise ValueError(f"active extraction run does not exist: {paths.root}")
    return paths


def create_staged_extraction(workspace: Path, run_id: str | None = None) -> StagedExtraction:
    workspace = Path(workspace)
    if run_id is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{timestamp}-{secrets.token_hex(4)}"
    run_id = ActiveExtraction(run_id=run_id).run_id
    staging_workspace = workspace / ".staging" / "extract" / run_id
    staged_run_root = staging_workspace / "extraction-runs" / run_id
    final_run_root = workspace / "extraction-runs" / run_id
    if staging_workspace.exists() or final_run_root.exists():
        raise FileExistsError(f"extraction run already exists: {run_id}")
    paths = _paths_for_root(run_id, staged_run_root)
    for directory in (
        paths.rounds_manifest.parent,
        paths.corrections.parent,
        paths.extraction_errors.parent,
        staged_run_root / "frames",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return StagedExtraction(
        run_id=run_id,
        staging_workspace=staging_workspace,
        staged_run_root=staged_run_root,
        final_run_root=final_run_root,
        evidence_prefix=Path("extraction-runs") / run_id / "frames",
        paths=paths,
    )


def publish_staged_extraction(workspace: Path, staged: StagedExtraction) -> ExtractionPaths:
    workspace = Path(workspace)
    if staged.final_run_root.exists():
        raise FileExistsError(f"published extraction run already exists: {staged.final_run_root}")
    _validate_staged_extraction(staged)

    staged.final_run_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged.staged_run_root, staged.final_run_root)

    pointer = workspace / "manifests" / "extraction.active.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_suffix(pointer.suffix + ".tmp")
    temporary.write_text(ActiveExtraction(run_id=staged.run_id).model_dump_json(), encoding="utf-8")
    os.replace(temporary, pointer)
    return _run_paths(workspace, staged.run_id)


def _validate_staged_extraction(staged: StagedExtraction) -> None:
    samples = read_jsonl(staged.paths.rounds_manifest, RoundSample)
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("staged extraction contains duplicate sample IDs")
    expected_prefix = Path("extraction-runs") / staged.run_id / "frames"
    for sample in samples:
        for label, raw_path in (
            ("prep", sample.evidence.prep),
            ("layout", sample.evidence.layout),
            ("end", sample.evidence.end),
        ):
            if raw_path is None:
                raise ValueError(f"staged sample {sample.sample_id} has missing evidence field: {label}")
            relative = Path(raw_path)
            if relative.is_absolute() or ".." in relative.parts or not relative.is_relative_to(expected_prefix):
                raise ValueError(f"staged sample {sample.sample_id} has evidence outside its run: {raw_path}")
            physical = staged.staging_workspace / relative
            if not physical.is_file():
                raise ValueError(f"staged sample {sample.sample_id} references missing evidence: {raw_path}")
    if not staged.paths.corrections.exists():
        raise ValueError("staged extraction is missing its review store")


def _run_paths(workspace: Path, run_id: str) -> ExtractionPaths:
    return _paths_for_root(run_id, workspace / "extraction-runs" / run_id)


def _paths_for_root(run_id: str, root: Path) -> ExtractionPaths:
    return ExtractionPaths(
        run_id=run_id,
        root=root,
        rounds_manifest=root / "manifests" / "rounds.auto.jsonl",
        corrections=root / "review" / "corrections.jsonl",
        extraction_errors=root / "reports" / "extraction-errors.json",
        phase_diagnostics=root / "reports" / "phase-diagnostics.json",
    )
