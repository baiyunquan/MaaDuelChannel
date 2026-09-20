from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from maa_duel.schema import (
    AnnotationSource,
    BoundingBox,
    ReviewStatus,
    RosterEntry,
    RoundSample,
    SideData,
    UnitDetection,
    Winner,
)
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
        corrections = self.load()
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


def apply_table_edits(
    sample: RoundSample,
    *,
    roster_rows: object,
    unit_rows: object,
    winner: str | None,
    status: ReviewStatus,
) -> RoundSample:
    rosters: dict[str, list[RosterEntry]] = {"left": [], "right": []}
    units: dict[str, list[UnitDetection]] = {"left": [], "right": []}
    for row in _table_rows(roster_rows):
        if not row or row[0] not in rosters:
            continue
        side = str(row[0])
        rosters[side].append(RosterEntry(enemy_id=int(row[1]), count=int(row[2]), confidence=float(row[3])))
    for row in _table_rows(unit_rows):
        if not row or row[0] not in units:
            continue
        side = str(row[0])
        units[side].append(
            UnitDetection(
                enemy_id=int(row[1]),
                x=float(row[2]),
                y=float(row[3]),
                bbox=BoundingBox(x1=float(row[4]), y1=float(row[5]), x2=float(row[6]), y2=float(row[7])),
                confidence=float(row[8]),
                source=AnnotationSource.MANUAL,
            )
        )

    payload = sample.model_dump(mode="json")
    payload["left"] = SideData(roster=rosters["left"], units=units["left"]).model_dump(mode="json")
    payload["right"] = SideData(roster=rosters["right"], units=units["right"]).model_dump(mode="json")
    payload["winner"] = Winner(winner).value if winner else None
    payload["review_status"] = status.value
    if status is ReviewStatus.ACCEPTED:
        payload["failure_reasons"] = []
        payload["winner_confidence"] = 1.0
    return RoundSample.model_validate(payload)


def _table_rows(value: object) -> list[list[object]]:
    if value is None:
        return []
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [list(row) for row in value]


def _roster_rows(sample: RoundSample) -> list[list[object]]:
    return [
        [side_name, entry.enemy_id, entry.count, entry.confidence]
        for side_name, side in (("left", sample.left), ("right", sample.right))
        for entry in side.roster
    ]


def _unit_rows(sample: RoundSample) -> list[list[object]]:
    return [
        [
            side_name,
            unit.enemy_id,
            unit.x,
            unit.y,
            unit.bbox.x1,
            unit.bbox.y1,
            unit.bbox.x2,
            unit.bbox.y2,
            unit.confidence,
        ]
        for side_name, side in (("left", sample.left), ("right", sample.right))
        for unit in side.units
    ]


def _annotated_layout(workspace: Path, sample: RoundSample):
    if not sample.evidence.layout:
        return None
    path = workspace / sample.evidence.layout
    image = cv2.imdecode(np.fromfile(path, dtype="uint8"), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    for side_name, side, color in (
        ("L", sample.left, (0, 140, 255)),
        ("R", sample.right, (255, 140, 0)),
    ):
        for unit in side.units:
            x1, y1 = int(unit.bbox.x1 * width), int(unit.bbox.y1 * height)
            x2, y2 = int(unit.bbox.x2 * width), int(unit.bbox.y2 * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"{side_name}:{unit.enemy_id}",
                (x1, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def launch_review(workspace: Path) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is not installed; install the review extra") from exc

    automatic = read_jsonl(workspace / "manifests" / "rounds.auto.jsonl", RoundSample)
    if not automatic:
        raise ValueError("automatic round manifest is empty; run extract first")
    store = ReviewStore(workspace / "review" / "corrections.jsonl")
    samples = store.overlay(automatic)

    def load(index: int):
        index = max(0, min(len(samples) - 1, int(index)))
        sample = samples[index]
        prep = str(workspace / sample.evidence.prep) if sample.evidence.prep else None
        end = str(workspace / sample.evidence.end) if sample.evidence.end else None
        return (
            index,
            f"{index + 1}/{len(samples)} · {sample.sample_id} · {sample.review_status.value}",
            prep,
            _annotated_layout(workspace, sample),
            end,
            _roster_rows(sample),
            _unit_rows(sample),
            sample.winner.value if sample.winner else None,
            "\n".join(sample.failure_reasons),
        )

    def save(index, roster_rows, unit_rows, winner, note, accepted):
        index = int(index)
        status = ReviewStatus.ACCEPTED if accepted else ReviewStatus.REJECTED
        edited = apply_table_edits(
            samples[index],
            roster_rows=roster_rows,
            unit_rows=unit_rows,
            winner=winner,
            status=status,
        )
        samples[index] = edited
        store.save(ReviewCorrection(sample=edited, note=note or ""))
        next_index = min(index + 1, len(samples) - 1)
        return load(next_index)

    with gr.Blocks(title="MAA Duel Channel Review") as demo:
        index = gr.State(0)
        title = gr.Markdown()
        with gr.Row():
            prep_image = gr.Image(label="Preparation", interactive=False)
            layout_image = gr.Image(label="Initial layout", interactive=False)
            end_image = gr.Image(label="Winner evidence", interactive=False)
        roster = gr.Dataframe(
            headers=["side", "enemy_id", "count", "confidence"],
            datatype=["str", "number", "number", "number"],
            type="array",
            label="Roster",
        )
        units = gr.Dataframe(
            headers=["side", "enemy_id", "x", "y", "x1", "y1", "x2", "y2", "confidence"],
            datatype=["str", "number", "number", "number", "number", "number", "number", "number", "number"],
            type="array",
            label="Units and boxes",
        )
        winner = gr.Dropdown(["left", "right"], label="Winner")
        note = gr.Textbox(label="Review note")
        with gr.Row():
            previous = gr.Button("Previous")
            accept = gr.Button("Accept", variant="primary")
            reject = gr.Button("Reject")
            following = gr.Button("Next")

        outputs = [index, title, prep_image, layout_image, end_image, roster, units, winner, note]
        demo.load(load, inputs=[index], outputs=outputs)
        previous.click(lambda value: load(int(value) - 1), inputs=[index], outputs=outputs)
        following.click(lambda value: load(int(value) + 1), inputs=[index], outputs=outputs)
        accept.click(
            lambda idx, r, u, w, n: save(idx, r, u, w, n, True),
            inputs=[index, roster, units, winner, note],
            outputs=outputs,
        )
        reject.click(
            lambda idx, r, u, w, n: save(idx, r, u, w, n, False),
            inputs=[index, roster, units, winner, note],
            outputs=outputs,
        )
    demo.launch(inbrowser=True)
