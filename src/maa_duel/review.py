from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from maa_duel.schema import RoundSample
from maa_duel.store import read_jsonl, write_jsonl


class ReviewCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    sample: RoundSample
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    note: str = ""

    @model_validator(mode="after")
    def revalidate_sample(self) -> ReviewCorrection:
        self.sample = RoundSample.model_validate(self.sample.model_dump(mode="json"))
        return self


class ReviewStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, ReviewCorrection]:
        rows = read_jsonl(self.path, ReviewCorrection)
        return {row.sample.sample_id: row for row in rows}

    def save(self, correction: ReviewCorrection) -> None:
        self.save_many([correction])

    def save_many(self, values: list[ReviewCorrection]) -> None:
        corrections = self.load()
        for correction in values:
            corrections[correction.sample.sample_id] = correction
        write_jsonl(
            self.path,
            (corrections[key] for key in sorted(corrections)),
        )

    def overlay(self, automatic: list[RoundSample]) -> list[RoundSample]:
        corrections = self.load()
        effective = [
            corrections[sample.sample_id].sample if sample.sample_id in corrections else sample for sample in automatic
        ]
        automatic_ids = {sample.sample_id for sample in automatic}
        effective.extend(corrections[sample_id].sample for sample_id in sorted(corrections.keys() - automatic_ids))
        return effective


def launch_review(workspace: Path):
    """Prepare Platform datasets and open the hosted visual annotation editor."""

    import webbrowser

    from maa_duel.annotations import PLATFORM_URL, export_platform_annotations

    exported = export_platform_annotations(workspace)
    webbrowser.open(PLATFORM_URL)
    return exported


def launch_local_review_qt(
    workspace: Path,
    video_dir: Path | None = None,
    skip_prep_roster: bool = False,
    only_prep_roster: bool = False,
) -> None:
    """Launch desktop PyQt6 review application with optional two-stage flow."""
    from maa_duel.gui import launch_local_review_qt as _launch_qt

    _launch_qt(
        workspace,
        video_dir=video_dir,
        skip_prep_roster=skip_prep_roster,
        only_prep_roster=only_prep_roster,
    )

