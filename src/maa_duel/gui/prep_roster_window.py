from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from maa_duel.gui.enemy_palette import EnemyPalette
from maa_duel.review import ReviewCorrection, ReviewStore
from maa_duel.schema import RosterEntry, RoundSample
from maa_duel.store import read_jsonl
from maa_duel.vision.roster import crop_normalized, default_slot_specs

if TYPE_CHECKING:
    pass


class SlotPairCardWidget(QFrame):
    """Card widget displaying a pair: [Real Prep Crop Image] + [Recognized Catalog Portrait]."""

    card_clicked = pyqtSignal(object)  # emits self

    def __init__(
        self,
        slot_data: dict,
        crop_pixmap: QPixmap | None,
        catalog_pixmap: QPixmap | None,
        enemy_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.slot_data = slot_data
        self.crop_pixmap = crop_pixmap
        self.catalog_pixmap = catalog_pixmap
        self.enemy_name = enemy_name
        self.is_selected = False

        self.setMinimumHeight(112)
        self.setMinimumWidth(130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._init_ui()
        self._update_style()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Header: Round badge + Count spinbox
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        side_str = "左" if self.slot_data["side"] == "left" else "右"
        slot_num = self.slot_data["slot_idx"] + 1
        round_idx = self.slot_data["round_index"]
        badge_text = f"R{round_idx}-{side_str}{slot_num}"

        self.badge_label = QLabel(badge_text)
        self.badge_label.setStyleSheet(
            "font-weight: bold; font-size: 11px; "
            + ("color: #ffaa33;" if self.slot_data["side"] == "left" else "color: #33bbff;")
        )
        header_layout.addWidget(self.badge_label)
        header_layout.addStretch()

        count_lbl = QLabel("x")
        count_lbl.setStyleSheet("font-size: 10px; color: #888888;")
        header_layout.addWidget(count_lbl)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(0, 99)
        self.count_spin.setValue(self.slot_data["count"])
        self.count_spin.setFixedSize(48, 20)
        self.count_spin.setStyleSheet(
            "QSpinBox { background-color: #2b2b2b; color: #ffffff; "
            "border: 1px solid #444; border-radius: 3px; font-size: 10px; }"
        )
        self.count_spin.valueChanged.connect(self._on_count_changed)
        header_layout.addWidget(self.count_spin)

        layout.addLayout(header_layout)

        # Middle: Two images side-by-side (Real Crop vs Catalog Portrait)
        images_layout = QHBoxLayout()
        images_layout.setContentsMargins(0, 0, 0, 0)
        images_layout.setSpacing(6)
        images_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Left: Real in-game prep crop
        self.crop_label = QLabel()
        self.crop_label.setFixedSize(50, 50)
        self.crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.crop_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; border-radius: 4px;")
        if self.crop_pixmap and not self.crop_pixmap.isNull():
            self.crop_label.setPixmap(
                self.crop_pixmap.scaled(
                    50, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )
        else:
            self.crop_label.setText("切图")
            self.crop_label.setStyleSheet("color: #666; font-size: 10px; border: 1px solid #333;")
        images_layout.addWidget(self.crop_label)

        arrow_label = QLabel("->")
        arrow_label.setStyleSheet("color: #666666; font-size: 10px; font-weight: bold;")
        images_layout.addWidget(arrow_label)

        # Right: Catalog thumbnail
        self.catalog_label = QLabel()
        self.catalog_label.setFixedSize(50, 50)
        self.catalog_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.catalog_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444; border-radius: 4px;")
        self._update_catalog_display()
        images_layout.addWidget(self.catalog_label)

        layout.addLayout(images_layout)

        # Bottom: Enemy name and ID
        self.name_label = QLabel()
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet("font-size: 10px; color: #ffffff;")
        self._update_name_display()
        layout.addWidget(self.name_label)

    def _update_catalog_display(self) -> None:
        if self.slot_data["enemy_id"] == 0:
            self.catalog_label.clear()
            self.catalog_label.setText("空")
            self.catalog_label.setStyleSheet(
                "background-color: #222; color: #777; font-size: 11px; border: 1px dashed #444; border-radius: 4px;"
            )
        elif self.catalog_pixmap and not self.catalog_pixmap.isNull():
            self.catalog_label.setPixmap(
                self.catalog_pixmap.scaled(
                    50, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )
            self.catalog_label.setStyleSheet(
                "background-color: #1a1a1a; border: 1px solid #00aaff; border-radius: 4px;"
            )
        else:
            self.catalog_label.clear()
            self.catalog_label.setText(str(self.slot_data["enemy_id"]))
            self.catalog_label.setStyleSheet(
                "color: #ddd; font-size: 10px; border: 1px solid #444; border-radius: 4px;"
            )

    def _update_name_display(self) -> None:
        eid = self.slot_data["enemy_id"]
        if eid == 0:
            self.name_label.setText("空槽位")
            self.name_label.setStyleSheet("font-size: 10px; color: #777777;")
        else:
            txt = f"{self.enemy_name} ({eid})"
            self.name_label.setText(txt)
            self.name_label.setStyleSheet("font-size: 10px; color: #dddddd; font-weight: bold;")
            side_str = self.slot_data["side"]
            r_idx = self.slot_data["round_index"]
            self.setToolTip(f"对局: R{r_idx} {side_str} | ID {eid}: {self.enemy_name}")

    def _on_count_changed(self, val: int) -> None:
        self.slot_data["count"] = val

    def set_selected(self, selected: bool) -> None:
        self.is_selected = selected
        self._update_style()

    def update_enemy(self, enemy_id: int, name: str, pixmap: QPixmap | None) -> None:
        self.slot_data["enemy_id"] = enemy_id
        self.enemy_name = name
        self.catalog_pixmap = pixmap
        if enemy_id > 0 and self.slot_data["count"] == 0:
            self.slot_data["count"] = 1
            self.count_spin.blockSignals(True)
            self.count_spin.setValue(1)
            self.count_spin.blockSignals(False)
        self._update_catalog_display()
        self._update_name_display()

    def _update_style(self) -> None:
        if self.is_selected:
            self.setStyleSheet(
                "QFrame { border: 2px solid #00d4ff; background-color: #242c38; border-radius: 6px; }"
            )
        else:
            self.setStyleSheet(
                "QFrame { border: 1px solid #383838; background-color: #202020; border-radius: 6px; }\n"
                "QFrame:hover { border: 1px solid #00aaff; background-color: #252525; }"
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.card_clicked.emit(self)
        super().mousePressEvent(event)


class PrepRosterReviewWindow(QMainWindow):
    """2K resolution batch review window for preparation stage round card slots."""

    def __init__(
        self,
        workspace: Path,
        video_dir: Path | None = None,
        on_proceed_to_workbench: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.video_dir = video_dir
        self.on_proceed_to_workbench = on_proceed_to_workbench

        self.setWindowTitle("MaaDuelChannel 准备区圆框批量审核工作台 (2K 密集校对)")
        self.resize(2400, 1350)

        self.store = ReviewStore(self.workspace / "review" / "corrections.jsonl")
        self.samples: list[RoundSample] = []
        self.all_slots: list[dict] = []
        self.active_slots_filtered: list[dict] = []
        self.current_page: int = 0
        self.page_size: int = 50  # 50 pairs per page
        self.selected_card: SlotPairCardWidget | None = None

        # Prep frame crop cache: key = (sample_id, side, slot_idx) -> QPixmap
        self.crop_cache: dict[tuple[str, str, int], QPixmap] = {}
        self.slot_specs = default_slot_specs()

        self._init_ui()
        self._init_shortcuts()
        self.load_dataset()

    def _init_ui(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #161616; color: #dddddd; }
            QLabel { color: #cccccc; }
            QLineEdit, QSpinBox {
                background-color: #262626; border: 1px solid #444; border-radius: 4px; color: #fff; padding: 2px 5px;
            }
            """
        )

        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # ----------------- Left Area: Dense Grid + Navigation -----------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Top Control Bar
        top_bar = QFrame()
        top_bar.setStyleSheet("background-color: #222222; border-radius: 6px; padding: 4px;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(6, 4, 6, 4)
        top_layout.setSpacing(10)

        title_lbl = QLabel("准备区圆框批量校对 (2K)")
        title_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff;")
        top_layout.addWidget(title_lbl)

        # Filter Checkbox
        self.cb_include_empty = QCheckBox("包含空卡槽")
        self.cb_include_empty.setChecked(False)
        self.cb_include_empty.toggled.connect(self._on_filter_toggled)
        top_layout.addWidget(self.cb_include_empty)

        top_layout.addStretch()

        # Pagination controls
        self.btn_prev_page = QPushButton("<- 上一页 (A)")
        self.btn_prev_page.clicked.connect(self.prev_page)
        top_layout.addWidget(self.btn_prev_page)

        self.page_info_label = QLabel("第 1 / 1 页 (共 0 槽位)")
        self.page_info_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #ffffff;")
        top_layout.addWidget(self.page_info_label)

        self.btn_next_page = QPushButton("下一页 -> (D)")
        self.btn_next_page.clicked.connect(self.next_page)
        top_layout.addWidget(self.btn_next_page)

        top_layout.addSpacing(16)

        # Action Buttons
        self.btn_save_page = QPushButton("保存本页修改")
        self.btn_save_page.setStyleSheet(
            "QPushButton { background-color: #2a6f3b; color: white; font-weight: bold; "
            "padding: 6px 12px; border-radius: 4px; border: 1px solid #3b8a4f; }\n"
            "QPushButton:hover { background-color: #358547; }"
        )
        self.btn_save_page.clicked.connect(self.save_current_page_edits)
        top_layout.addWidget(self.btn_save_page)

        self.btn_proceed = QPushButton("保存并进入单局拉框 (Enter)")
        self.btn_proceed.setStyleSheet(
            "QPushButton { background-color: #007acc; color: white; font-weight: bold; "
            "padding: 6px 14px; border-radius: 4px; border: 1px solid #0099ff; }\n"
            "QPushButton:hover { background-color: #008be6; }"
        )
        self.btn_proceed.clicked.connect(self.save_and_proceed)
        top_layout.addWidget(self.btn_proceed)

        left_layout.addWidget(top_bar)

        # Center Scrollable Dense Grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("QScrollArea { background-color: #141414; border: 1px solid #2d2d2d; }")

        self.grid_container = QWidget()
        self.grid_container.setStyleSheet("background-color: #141414;")
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(6, 6, 6, 6)
        self.grid_layout.setSpacing(6)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        for col in range(10):
            self.grid_layout.setColumnStretch(col, 1)
        for row in range(5):
            self.grid_layout.setRowStretch(row, 1)

        self.scroll_area.setWidget(self.grid_container)
        left_layout.addWidget(self.scroll_area)

        main_layout.addWidget(left_widget, stretch=1)

        # ----------------- Right Area: Fixed Width 360px -----------------
        right_widget = QWidget()
        right_widget.setFixedWidth(360)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        right_title = QLabel("目标敌人选择栏 (先选左侧卡片，再点此处)")
        right_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #ffaa33;")
        right_layout.addWidget(right_title)

        self.palette = EnemyPalette(self.workspace, icon_size=32, parent=self)
        self.palette.enemy_selected.connect(self._on_palette_enemy_selected)
        right_layout.addWidget(self.palette)

        main_layout.addWidget(right_widget, stretch=0)
        self.setCentralWidget(main_widget)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 | 点击左侧任意一组卡片选中，再点击右侧图谱头像即可修改")

    def _init_shortcuts(self) -> None:
        QShortcut(QKeySequence("A"), self, self.prev_page)
        QShortcut(QKeySequence("D"), self, self.next_page)
        QShortcut(QKeySequence("Return"), self, self.save_and_proceed)

    def load_dataset(self) -> None:
        manifest_path = self.workspace / "manifests" / "rounds.auto.jsonl"
        if not manifest_path.exists():
            QMessageBox.critical(self, "错误", f"未找到 manifests/rounds.auto.jsonl: {manifest_path}")
            return

        automatic = read_jsonl(manifest_path, RoundSample)
        if not automatic:
            QMessageBox.warning(self, "警告", "样本列表为空！")
            return

        self.samples = self.store.overlay(automatic)
        self._build_slots_index()
        self._filter_slots()
        self.load_page(0)

    def _build_slots_index(self) -> None:
        """Flatten samples into 6 slots per round."""
        self.all_slots = []
        for sample_idx, sample in enumerate(self.samples):
            # Left 3 slots (0, 1, 2)
            for slot_idx in range(3):
                if slot_idx < len(sample.left.roster):
                    entry = sample.left.roster[slot_idx]
                    eid, cnt = entry.enemy_id, entry.count
                else:
                    eid, cnt = 0, 0
                self.all_slots.append(
                    {
                        "sample_idx": sample_idx,
                        "sample_id": sample.sample_id,
                        "round_index": sample.round_index,
                        "side": "left",
                        "slot_idx": slot_idx,
                        "enemy_id": eid,
                        "count": cnt,
                        "orig_enemy_id": eid,
                        "orig_count": cnt,
                    }
                )

            # Right 3 slots (0, 1, 2)
            for slot_idx in range(3):
                if slot_idx < len(sample.right.roster):
                    entry = sample.right.roster[slot_idx]
                    eid, cnt = entry.enemy_id, entry.count
                else:
                    eid, cnt = 0, 0
                self.all_slots.append(
                    {
                        "sample_idx": sample_idx,
                        "sample_id": sample.sample_id,
                        "round_index": sample.round_index,
                        "side": "right",
                        "slot_idx": slot_idx,
                        "enemy_id": eid,
                        "count": cnt,
                        "orig_enemy_id": eid,
                        "orig_count": cnt,
                    }
                )

    def _filter_slots(self) -> None:
        include_empty = self.cb_include_empty.isChecked()
        if include_empty:
            self.active_slots_filtered = self.all_slots
        else:
            self.active_slots_filtered = [s for s in self.all_slots if s["enemy_id"] > 0]

    def _on_filter_toggled(self) -> None:
        self._filter_slots()
        self.load_page(0)

    def total_pages(self) -> int:
        total = len(self.active_slots_filtered)
        if total == 0:
            return 1
        return (total + self.page_size - 1) // self.page_size

    def prev_page(self) -> None:
        if self.current_page > 0:
            self.save_current_page_edits(silent=True)
            self.load_page(self.current_page - 1)

    def next_page(self) -> None:
        if self.current_page < self.total_pages() - 1:
            self.save_current_page_edits(silent=True)
            self.load_page(self.current_page + 1)

    def load_page(self, page_num: int) -> None:
        tot_pages = self.total_pages()
        self.current_page = max(0, min(tot_pages - 1, page_num))
        total_items = len(self.active_slots_filtered)

        self.page_info_label.setText(
            f"第 {self.current_page + 1} / {tot_pages} 页 (共 {total_items} 槽位)"
        )
        self.btn_prev_page.setEnabled(self.current_page > 0)
        self.btn_next_page.setEnabled(self.current_page < tot_pages - 1)

        # Clear existing cards in grid
        self.selected_card = None
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Calculate range
        start_idx = self.current_page * self.page_size
        end_idx = min(total_items, start_idx + self.page_size)
        page_items = self.active_slots_filtered[start_idx:end_idx]

        # Populate grid: 10 columns
        cols = 10
        for i, slot_data in enumerate(page_items):
            row = i // cols
            col = i % cols

            crop_pixmap = self._get_slot_crop(slot_data)
            eid = slot_data["enemy_id"]
            cat_pixmap = self.palette.get_enemy_pixmap(eid)
            name = self.palette.get_enemy_name(eid)

            card = SlotPairCardWidget(
                slot_data=slot_data,
                crop_pixmap=crop_pixmap,
                catalog_pixmap=cat_pixmap,
                enemy_name=name,
                parent=self.grid_container,
            )
            card.card_clicked.connect(self._on_card_clicked)
            self.grid_layout.addWidget(card, row, col)

        self.status_bar.showMessage(
            f"已加载第 {self.current_page + 1} 页 ({len(page_items)} 组卡片)", 2000
        )

    def _get_slot_crop(self, slot_data: dict) -> QPixmap | None:
        key = (slot_data["sample_id"], slot_data["side"], slot_data["slot_idx"])
        if key in self.crop_cache:
            return self.crop_cache[key]

        # Load prep frame
        sample = self.samples[slot_data["sample_idx"]]
        if not sample.evidence.prep:
            return None

        prep_path = self.workspace / sample.evidence.prep
        if not prep_path.exists():
            return None

        frame = cv2.imread(str(prep_path))
        if frame is None:
            return None

        # Crop all 6 slots for this sample and cache
        for spec in self.slot_specs:
            crop_bgr = crop_normalized(frame, spec.icon_rect)
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.crop_cache[(sample.sample_id, spec.side, spec.index)] = pix

        return self.crop_cache.get(key)

    def _on_card_clicked(self, card: SlotPairCardWidget) -> None:
        if self.selected_card:
            self.selected_card.set_selected(False)
        self.selected_card = card
        card.set_selected(True)
        eid = card.slot_data["enemy_id"]
        name = card.enemy_name
        self.status_bar.showMessage(
            f"选中卡片: [R{card.slot_data['round_index']} {card.slot_data['side']} 槽{card.slot_data['slot_idx']+1}] "
            f"当前敌人: {name} (ID {eid}) | 请在右侧图谱中点击要修改成的目标敌人",
            4000,
        )

    def _on_palette_enemy_selected(self, enemy_id: int) -> None:
        if not self.selected_card:
            self.status_bar.showMessage("请先在左侧网格中点击选中一个卡片，再从右侧选择敌人！", 3000)
            return

        name = self.palette.get_enemy_name(enemy_id)
        pix = self.palette.get_enemy_pixmap(enemy_id)
        self.selected_card.update_enemy(enemy_id, name, pix)
        self.status_bar.showMessage(f"已修改选中卡槽为: {name} (ID {enemy_id})", 3000)

    def save_current_page_edits(self, silent: bool = False) -> None:
        """Persist current modified slot values back into RoundSamples and save to corrections.jsonl."""
        modified_sample_indices = set()
        for slot in self.all_slots:
            if slot["enemy_id"] != slot["orig_enemy_id"] or slot["count"] != slot["orig_count"]:
                modified_sample_indices.add(slot["sample_idx"])
                slot["orig_enemy_id"] = slot["enemy_id"]
                slot["orig_count"] = slot["count"]

        if not modified_sample_indices:
            if not silent:
                self.status_bar.showMessage("当前无新修改", 2000)
            return

        # Reconstruct sample rosters
        corrections_to_save: list[ReviewCorrection] = []
        for s_idx in modified_sample_indices:
            sample = self.samples[s_idx]
            # Left slots
            left_entries = []
            for slot_data in [s for s in self.all_slots if s["sample_idx"] == s_idx and s["side"] == "left"]:
                if slot_data["enemy_id"] > 0 and slot_data["count"] > 0:
                    left_entries.append(
                        RosterEntry(enemy_id=slot_data["enemy_id"], count=slot_data["count"], confidence=1.0)
                    )

            # Right slots
            right_entries = []
            for slot_data in [s for s in self.all_slots if s["sample_idx"] == s_idx and s["side"] == "right"]:
                if slot_data["enemy_id"] > 0 and slot_data["count"] > 0:
                    right_entries.append(
                        RosterEntry(enemy_id=slot_data["enemy_id"], count=slot_data["count"], confidence=1.0)
                    )

            sample.left.roster = left_entries
            sample.right.roster = right_entries
            corrections_to_save.append(ReviewCorrection(sample=sample, note="准备区圆框批量审核校准"))

        self.store.save_many(corrections_to_save)
        if not silent:
            self.status_bar.showMessage(f"已保存 {len(corrections_to_save)} 局样本的卡槽修改", 3000)

    def save_and_proceed(self) -> None:
        """Save all edits and launch single-round workbench."""
        self.save_current_page_edits(silent=True)
        self.close()
        if self.on_proceed_to_workbench:
            self.on_proceed_to_workbench()
