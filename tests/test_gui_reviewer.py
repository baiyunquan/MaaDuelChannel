from __future__ import annotations

import os
from pathlib import Path

import pytest

# Ensure offscreen Qt platform for testing in headless environment
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from maa_duel.gui.canvas import AnnotationBoxItem, AnnotationCanvas
from maa_duel.gui.enemy_palette import EnemyPalette
from maa_duel.gui.reviewer_window import ReviewerMainWindow
from maa_duel.gui.roster_panel import RosterPanel
from maa_duel.schema import (
    AnnotationSource,
    BoundingBox,
    EvidenceFrames,
    ReviewStatus,
    RosterEntry,
    RoundSample,
    SideData,
    SourceRef,
    Timestamps,
    UnitDetection,
    Winner,
)
from maa_duel.store import write_jsonl


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mock_workspace(tmp_path: Path) -> Path:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    catalog = {
        "enemies": [
            {"enemy_id": 1, "name": "源石虫", "original_name": "slug"},
            {"enemy_id": 25, "name": "狂暴宿主", "original_name": "host"},
            {"enemy_id": 34, "name": "盾卫", "original_name": "shield"},
        ]
    }
    import json

    with open(assets_dir / "catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f)

    frames_dir = tmp_path / "frames" / "test12345678"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # create dummy layout image
    import cv2
    import numpy as np

    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_img_path = frames_dir / "dummy-layout.jpg"
    cv2.imwrite(str(dummy_img_path), dummy_img)

    # create sample
    sample = RoundSample(
        sample_id="a" * 32,
        source=SourceRef(
            video_relpath="test.mp4",
            video_sha256="1" * 64,
        ),
        round_index=1,
        timestamps=Timestamps(prep=1.0, layout=2.0, battle_start=3.0, battle_end=10.0),
        evidence=EvidenceFrames(
            prep=None,
            layout="frames/test12345678/dummy-layout.jpg",
            end=None,
        ),
        left=SideData(
            roster=[RosterEntry(enemy_id=25, count=2, confidence=0.9)],
            units=[
                UnitDetection(
                    enemy_id=25,
                    x=0.25,
                    y=0.5,
                    bbox=BoundingBox(x1=0.2, y1=0.4, x2=0.3, y2=0.6),
                    confidence=0.9,
                    source=AnnotationSource.AUTO,
                )
            ],
        ),
        right=SideData(
            roster=[RosterEntry(enemy_id=34, count=1, confidence=0.95)],
            units=[],
        ),
        winner=Winner.LEFT,
        review_status=ReviewStatus.PENDING,
        pipeline_version="0.1.0",
        failure_reasons=["left:count_mismatch:25:expected=2:actual=1"],
    )

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifest_dir / "rounds.auto.jsonl", [sample])

    return tmp_path


def test_enemy_palette(qapp, mock_workspace):
    palette = EnemyPalette(mock_workspace, icon_size=32)
    assert 0 in palette.enemies  # Empty slot
    assert 1 in palette.enemies
    assert 25 in palette.enemies
    assert palette.get_enemy_name(25) == "狂暴宿主"

    # Test filter
    palette._filter_enemies("狂暴")
    visible = [
        palette.list_widget.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(palette.list_widget.count())
        if not palette.list_widget.item(i).isHidden()
    ]
    assert len(visible) >= 1


def test_roster_panel(qapp, mock_workspace):
    palette = EnemyPalette(mock_workspace, icon_size=32)
    panel = RosterPanel()
    assert len(panel.slots) == 6

    # Select slot 0 and assign enemy 25
    panel.set_active_slot(0)
    panel.assign_enemy_to_active_slot(25, palette)
    assert panel.slots[0].enemy_id == 25
    assert panel.slots[0].count == 1

    # Change count
    panel.slots[0].count_spin.setValue(2)
    assert panel.slots[0].count == 2

    # Update quotas
    panel.update_quotas(left_counts={25: 2}, right_counts={})
    assert panel.slots[0].quota_label.text() == "[OK] 2/2"

    panel.update_quotas(left_counts={25: 1}, right_counts={})
    assert panel.slots[0].quota_label.text() == "[待补] 1/2"

    panel.update_quotas(left_counts={25: 3}, right_counts={})
    assert panel.slots[0].quota_label.text() == "! 3/2"

    left_roster, right_roster = panel.get_rosters()
    assert len(left_roster) == 1
    assert left_roster[0].enemy_id == 25
    assert left_roster[0].count == 2


def test_canvas_box_operations(qapp):
    canvas = AnnotationCanvas()
    canvas.img_size = canvas.img_size.__class__(1000, 1000)

    box = AnnotationBoxItem(
        rect=QRectF(100, 100, 100, 200),
        side="left",
        enemy_id=25,
        enemy_name="狂暴宿主",
        confidence=0.9,
    )
    canvas.scene.addItem(box)

    # Check unit detection conversion
    unit = box.to_unit_detection(1000, 1000)
    assert unit.enemy_id == 25
    assert unit.bbox.x1 == 0.1
    assert unit.bbox.y1 == 0.1
    assert unit.bbox.x2 == 0.2
    assert unit.bbox.y2 == 0.3
    assert unit.x == pytest.approx(0.15)
    assert unit.y == pytest.approx(0.3)

    # Check counting
    left_counts, right_counts = canvas.count_units()
    assert left_counts == {25: 1}
    assert right_counts == {}


def test_reviewer_main_window_workflow(qapp, mock_workspace, monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)
    window = ReviewerMainWindow(workspace=mock_workspace)
    assert len(window.samples) == 1

    sample = window.get_current_sample()
    assert sample.sample_id == "a" * 32
    assert sample.review_status == ReviewStatus.PENDING

    # Attempting to accept with count mismatch should fail validation
    # (Left has 2 expected, but only 1 drawn; Right has 1 expected, 0 drawn)
    window.save_verdict(ReviewStatus.ACCEPTED)
    # Should still be pending because validation failed
    assert window.samples[0].review_status == ReviewStatus.PENDING

    # Now fix the roster and boxes to match:
    # Set slot 0 to enemy 25, count 1
    window.roster_panel.slots[0].count_spin.setValue(1)
    # Set slot 3 (right) to enemy 34, count 1
    window.roster_panel.slots[3].count_spin.setValue(1)
    # Add box for right side
    right_box = AnnotationBoxItem(
        rect=QRectF(600, 100, 50, 50),
        side="right",
        enemy_id=34,
        enemy_name="盾卫",
    )
    window.canvas.scene.addItem(right_box)
    window._sync_quotas()

    # Now accept
    window.save_verdict(ReviewStatus.ACCEPTED)
    assert window.samples[0].review_status == ReviewStatus.ACCEPTED

    # Verify saved to corrections.jsonl
    corrections = window.store.load()
    assert sample.sample_id in corrections
    saved = corrections[sample.sample_id].sample
    assert saved.review_status == ReviewStatus.ACCEPTED
    assert saved.left.roster_counts() == saved.left.unit_counts()
    assert saved.right.roster_counts() == saved.right.unit_counts()


def test_instruction_guide_dialog(qapp):
    from maa_duel.gui.guide_dialog import InstructionGuideDialog

    dialog = InstructionGuideDialog()
    assert "操作指引" in dialog.windowTitle()
    assert dialog.width() > 0
    assert dialog.height() > 0

