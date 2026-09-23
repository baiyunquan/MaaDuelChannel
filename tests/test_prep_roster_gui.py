from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

from maa_duel.gui.prep_roster_window import PrepRosterReviewWindow, SlotPairCardWidget
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
from maa_duel.store import write_jsonl


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mock_prep_workspace(tmp_path: Path) -> Path:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    catalog = {
        "enemies": [
            {"enemy_id": 1, "name": "源石虫", "original_name": "slug"},
            {"enemy_id": 25, "name": "狂暴宿主", "original_name": "host"},
            {"enemy_id": 34, "name": "盾卫", "original_name": "shield"},
        ]
    }
    with open(assets_dir / "catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f)

    frames_dir = tmp_path / "frames" / "test12345678"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # create dummy prep frame image (720x1280)
    dummy_prep = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_prep_path = frames_dir / "test-prep.jpg"
    cv2.imwrite(str(dummy_prep_path), dummy_prep)

    # create sample
    sample = RoundSample(
        sample_id="b" * 32,
        source=SourceRef(
            video_relpath="test_video.mp4",
            video_sha256="2" * 64,
        ),
        round_index=3,
        timestamps=Timestamps(prep=1.0, layout=2.0, battle_start=3.0, battle_end=10.0),
        evidence=EvidenceFrames(
            prep="frames/test12345678/test-prep.jpg",
            layout=None,
            end=None,
        ),
        left=SideData(
            roster=[RosterEntry(enemy_id=25, count=2, confidence=0.9)],
            units=[],
        ),
        right=SideData(
            roster=[RosterEntry(enemy_id=34, count=1, confidence=0.95)],
            units=[],
        ),
        winner=Winner.LEFT,
        review_status=ReviewStatus.PENDING,
        pipeline_version="0.1.0",
        failure_reasons=[],
    )

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifest_dir / "rounds.auto.jsonl", [sample])

    return tmp_path


def test_slot_pair_card_widget(qapp):
    slot_data = {
        "sample_idx": 0,
        "sample_id": "b" * 32,
        "round_index": 3,
        "side": "left",
        "slot_idx": 0,
        "enemy_id": 25,
        "count": 2,
        "orig_enemy_id": 25,
        "orig_count": 2,
    }
    crop_pix = QPixmap(54, 54)
    crop_pix.fill(Qt.GlobalColor.gray)
    cat_pix = QPixmap(54, 54)
    cat_pix.fill(Qt.GlobalColor.darkBlue)

    card = SlotPairCardWidget(
        slot_data=slot_data,
        crop_pixmap=crop_pix,
        catalog_pixmap=cat_pix,
        enemy_name="狂暴宿主",
    )

    assert "R3-左1" in card.badge_label.text()
    assert card.count_spin.value() == 2
    assert "狂暴宿主" in card.name_label.text()

    # Test count change
    card.count_spin.setValue(4)
    assert card.slot_data["count"] == 4

    # Test enemy update
    card.update_enemy(34, "盾卫", None)
    assert card.slot_data["enemy_id"] == 34
    assert "盾卫" in card.name_label.text()

    # Test selection
    card.set_selected(True)
    assert card.is_selected is True


def test_prep_roster_review_window(qapp, mock_prep_workspace):
    proceeded = False

    def on_proceed():
        nonlocal proceeded
        proceeded = True

    window = PrepRosterReviewWindow(
        workspace=mock_prep_workspace,
        on_proceed_to_workbench=on_proceed,
    )

    assert len(window.samples) == 1
    # 6 slots total (3 left, 3 right)
    assert len(window.all_slots) == 6

    # By default, only non-empty slots are shown (1 left + 1 right = 2)
    assert len(window.active_slots_filtered) == 2
    assert window.total_pages() == 1

    # Check toggle empty slots
    window.cb_include_empty.setChecked(True)
    assert len(window.active_slots_filtered) == 6

    window.cb_include_empty.setChecked(False)
    assert len(window.active_slots_filtered) == 2

    # Test card selection and assignment
    first_item = window.grid_layout.itemAt(0)
    assert first_item is not None
    first_card: SlotPairCardWidget = first_item.widget()
    first_card.card_clicked.emit(first_card)
    assert window.selected_card is first_card

    # Change enemy using palette
    window._on_palette_enemy_selected(1)
    assert first_card.slot_data["enemy_id"] == 1
    assert "源石虫" in first_card.name_label.text()

    # Test save
    window.save_current_page_edits()
    corrections = window.store.load()
    assert ("b" * 32) in corrections
    saved_sample = corrections["b" * 32].sample
    # Left roster should now have enemy 1
    assert saved_sample.left.roster[0].enemy_id == 1
    assert saved_sample.left.roster[0].slot == 0

    # Test proceed
    window.save_and_proceed()
    assert proceeded is True
