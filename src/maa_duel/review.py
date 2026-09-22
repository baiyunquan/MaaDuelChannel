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


def _load_image(path: Path | None):
    if not path or not path.exists():
        return None
    image = cv2.imdecode(np.fromfile(path, dtype="uint8"), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _annotated_layout(workspace: Path, sample: RoundSample):
    if not sample.evidence.layout:
        return None
    path = workspace / sample.evidence.layout
    if not path.exists():
        return None
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


def launch_local_review(workspace: Path, port: int = 7860) -> None:
    """Launch an offline local Gradio web UI to review round samples."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is not installed; run: uv pip install gradio") from exc

    import json

    catalog_path = workspace / "assets" / "catalog.json"
    enemy_names: dict[int, str] = {}
    catalog_rows: list[list[object]] = []
    if catalog_path.exists():
        try:
            with open(catalog_path, encoding="utf-8") as f:
                catalog_data = json.load(f)
            for enemy in catalog_data.get("enemies", []):
                eid = int(enemy.get("enemy_id", 0))
                name = enemy.get("name", "")
                orig = enemy.get("original_name", "")
                enemy_names[eid] = name
                catalog_rows.append([eid, name, orig])
        except Exception:
            pass

    automatic = read_jsonl(workspace / "manifests" / "rounds.auto.jsonl", RoundSample)
    if not automatic:
        raise ValueError("automatic round manifest is empty; run extract first")
    store = ReviewStore(workspace / "review" / "corrections.jsonl")
    samples = store.overlay(automatic)

    def load(index: int):
        index = max(0, min(len(samples) - 1, int(index)))
        sample = samples[index]
        prep = _load_image(workspace / sample.evidence.prep) if sample.evidence.prep else None
        layout = _annotated_layout(workspace, sample)
        end = _load_image(workspace / sample.evidence.end) if sample.evidence.end else None

        left_desc = (
            ", ".join(f"{enemy_names.get(e.enemy_id, '未知')}(ID {e.enemy_id}) x{e.count}" for e in sample.left.roster)
            or "无"
        )
        right_desc = (
            ", ".join(f"{enemy_names.get(e.enemy_id, '未知')}(ID {e.enemy_id}) x{e.count}" for e in sample.right.roster)
            or "无"
        )
        reasons_text = (
            f"\n> [提示] **异常/置信度提示**: `{', '.join(sample.failure_reasons)}`" if sample.failure_reasons else ""
        )

        winner_text = sample.winner.value if sample.winner else "未知"
        title_md = (
            f"### 对局 {index + 1} / {len(samples)} (样本: `{sample.sample_id}`)\n"
            f"- **回合数**: 第 {sample.round_index} 回合 | **来源视频**: `{sample.source.video_relpath}`\n"
            f"- **审核状态**: `{sample.review_status.value}` | **预测胜方**: `{winner_text}`\n"
            f"- **左方识别阵容**: {left_desc}\n"
            f"- **右方识别阵容**: {right_desc}"
            f"{reasons_text}"
        )

        return (
            index,
            index + 1,
            title_md,
            prep,
            layout,
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

    with gr.Blocks(title="MAA 对决频道本地人工审核平台") as demo:
        index = gr.State(0)
        title = gr.Markdown()
        with gr.Row():
            prep_image = gr.Image(label="1. 准备阶段头像 (Prep)", interactive=False)
            layout_image = gr.Image(label="2. 倒计时归零站位与目标检测框 (Layout)", interactive=False)
            end_image = gr.Image(label="3. 胜负结算存活状态 (Winner Evidence)", interactive=False)
        with gr.Row():
            roster = gr.Dataframe(
                headers=["side", "enemy_id", "count", "confidence"],
                datatype=["str", "number", "number", "number"],
                type="array",
                label="阵容识别 (Roster: side, enemy_id, count, conf)",
            )
            units = gr.Dataframe(
                headers=["side", "enemy_id", "x", "y", "x1", "y1", "x2", "y2", "confidence"],
                datatype=["str", "number", "number", "number", "number", "number", "number", "number", "number"],
                type="array",
                label="战场站位与边界框 (Units)",
            )
        with gr.Row():
            winner = gr.Dropdown(["left", "right"], label="获胜方 (Winner)")
            note = gr.Textbox(label="审核备注 (Review note)")
        with gr.Row():
            previous = gr.Button("<- 上一局 (Previous)")
            accept = gr.Button("通过 (Accept)", variant="primary")
            reject = gr.Button("驳回 (Reject)", variant="stop")
            following = gr.Button("-> 下一局 (Next)")
        with gr.Row():
            jump_number = gr.Number(
                value=1, minimum=1, maximum=len(samples), step=1, label=f"跳转至指定局数 (1 - {len(samples)})"
            )
            jump_btn = gr.Button("跳转", scale=0)

        with gr.Accordion("绿藤城敌人图鉴与 ID 对照表 (Enemy Catalog Reference)", open=False):
            gr.Dataframe(
                value=catalog_rows,
                headers=["Enemy ID", "常用名", "原始代码名"],
                datatype=["number", "str", "str"],
                interactive=False,
            )

        outputs = [index, jump_number, title, prep_image, layout_image, end_image, roster, units, winner, note]
        demo.load(load, inputs=[index], outputs=outputs)
        previous.click(lambda value: load(int(value) - 1), inputs=[index], outputs=outputs)
        following.click(lambda value: load(int(value) + 1), inputs=[index], outputs=outputs)
        jump_btn.click(lambda value: load(int(value) - 1), inputs=[jump_number], outputs=outputs)
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

    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=True)


def launch_review(workspace: Path):
    """Prepare Platform datasets and open the hosted visual annotation editor."""

    import webbrowser

    from maa_duel.annotations import PLATFORM_URL, export_platform_annotations

    exported = export_platform_annotations(workspace)
    webbrowser.open(PLATFORM_URL)
    return exported


def launch_local_review_qt(workspace: Path, video_dir: Path | None = None) -> None:
    """Launch desktop PyQt6 review application."""
    from maa_duel.gui import launch_local_review_qt as _launch_qt

    _launch_qt(workspace, video_dir=video_dir)

