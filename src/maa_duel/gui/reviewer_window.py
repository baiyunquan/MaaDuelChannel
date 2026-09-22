from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from maa_duel.gui.canvas import AnnotationBoxItem, AnnotationCanvas, CanvasMode
from maa_duel.gui.enemy_palette import EnemyPalette
from maa_duel.gui.guide_dialog import InstructionGuideDialog
from maa_duel.gui.roster_panel import RosterPanel
from maa_duel.gui.video_timeline import VideoTimelineFineTuner
from maa_duel.review import ReviewCorrection, ReviewStore
from maa_duel.schema import ReviewStatus, RoundSample, SideData, Winner
from maa_duel.store import read_jsonl


class ReviewerMainWindow(QMainWindow):
    """Integrated 3-column desktop review workbench for MaaDuelChannel."""

    def __init__(
        self,
        workspace: Path,
        video_dir: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.video_dir = video_dir

        self.setWindowTitle("MaaDuelChannel 本地审核与标注工作台")
        self.resize(1600, 950)

        self.store = ReviewStore(self.workspace / "review" / "corrections.jsonl")
        self.samples: list[RoundSample] = []
        self.filtered_indices: list[int] = []
        self.current_idx_in_filtered: int = 0
        self.guide_dialog: InstructionGuideDialog | None = None

        self._init_ui()
        self._init_shortcuts()
        self.load_dataset()

    def _init_ui(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #1e1e1e; color: #dddddd; }
            QGroupBox {
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                margin-top: 8px;
                font-weight: bold;
                color: #e0e0e0;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
            }
            QLabel { color: #cccccc; }
            QLineEdit, QSpinBox, QComboBox {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 4px;
                color: #ffffff;
                padding: 3px 6px;
            }
            QRadioButton { color: #dddddd; }
            QRadioButton::indicator:checked { background-color: #0099ff; border: 2px solid #ffffff; }
            """
        )

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ----------------- Left Column: Navigator & Timeline & Verdict -----------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)

        # 1. Sample Navigator
        nav_box = QFrame()
        nav_box.setStyleSheet("background-color: #252525; border-radius: 6px; padding: 4px;")
        nav_layout = QVBoxLayout(nav_box)
        nav_layout.setContentsMargins(6, 6, 6, 6)
        nav_layout.setSpacing(4)

        nav_title = QLabel("样本导航")
        nav_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #ffffff;")
        nav_layout.addWidget(nav_title)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("过滤:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["全部样本", "仅待审核 (Pending)", "仅已通过 (Accepted)", "仅已驳回 (Rejected)"])
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self.filter_combo)
        nav_layout.addLayout(filter_row)

        jump_row = QHBoxLayout()
        self.sample_idx_label = QLabel("样本 0/0")
        self.sample_idx_label.setStyleSheet("font-weight: bold; color: #00d4ff;")
        jump_row.addWidget(self.sample_idx_label)

        self.jump_spin = QSpinBox()
        self.jump_spin.setRange(1, 1)
        self.jump_spin.valueChanged.connect(self._on_jump_changed)
        jump_row.addWidget(self.jump_spin)
        nav_layout.addLayout(jump_row)

        # Sample summary card
        self.sample_info_label = QLabel("未加载样本")
        self.sample_info_label.setWordWrap(True)
        self.sample_info_label.setStyleSheet(
            "font-size: 11px; color: #bbb; background: #1c1c1c; padding: 6px; border-radius: 4px;"
        )
        nav_layout.addWidget(self.sample_info_label)

        # Failure reasons banner
        self.reasons_label = QLabel("")
        self.reasons_label.setWordWrap(True)
        self.reasons_label.setStyleSheet(
            "font-size: 10px; color: #ffaa55; background: #3d2b15; padding: 4px; border-radius: 4px;"
        )
        self.reasons_label.hide()
        nav_layout.addWidget(self.reasons_label)

        nav_btns = QHBoxLayout()
        self.prev_btn = QPushButton("<- 上一局 (A)")
        self.prev_btn.clicked.connect(self.prev_sample)
        self.next_btn = QPushButton("-> 下一局 (D)")
        self.next_btn.clicked.connect(self.next_sample)
        nav_btns.addWidget(self.prev_btn)
        nav_btns.addWidget(self.next_btn)
        nav_layout.addLayout(nav_btns)

        left_layout.addWidget(nav_box)

        # 2. Timeline Fine-Tuner (Step 1)
        self.timeline_tuner = VideoTimelineFineTuner(self.workspace, self.video_dir, self)
        self.timeline_tuner.evidence_updated.connect(self._on_evidence_updated)
        left_layout.addWidget(self.timeline_tuner)

        # 3. Verdict & Actions (Step 3)
        verdict_box = QFrame()
        verdict_box.setStyleSheet("background-color: #252525; border-radius: 6px; padding: 6px;")
        verdict_layout = QVBoxLayout(verdict_box)
        verdict_layout.setContentsMargins(6, 6, 6, 6)
        verdict_layout.setSpacing(6)

        verdict_title = QLabel("胜负与审核确认")
        verdict_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #ffffff;")
        verdict_layout.addWidget(verdict_title)

        # Winner Radios
        winner_row = QHBoxLayout()
        winner_row.addWidget(QLabel("胜方:"))
        self.winner_group = QButtonGroup(self)
        self.radio_left_win = QRadioButton("左方胜 (Left)")
        self.radio_right_win = QRadioButton("右方胜 (Right)")
        self.winner_group.addButton(self.radio_left_win)
        self.winner_group.addButton(self.radio_right_win)
        winner_row.addWidget(self.radio_left_win)
        winner_row.addWidget(self.radio_right_win)
        verdict_layout.addLayout(winner_row)

        # Note
        note_row = QHBoxLayout()
        note_row.addWidget(QLabel("备注:"))
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("选填审核备注...")
        note_row.addWidget(self.note_edit)
        verdict_layout.addLayout(note_row)

        # Action buttons
        actions_layout = QHBoxLayout()
        self.accept_btn = QPushButton("通过 (Accept)")
        self.accept_btn.setStyleSheet(
            "QPushButton { background-color: #1e7038; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 4px; border: 1px solid #2e9048; }\n"
            "QPushButton:hover { background-color: #288c47; }"
        )
        self.accept_btn.clicked.connect(lambda: self.save_verdict(ReviewStatus.ACCEPTED))

        self.reject_btn = QPushButton("驳回 (Reject)")
        self.reject_btn.setStyleSheet(
            "QPushButton { background-color: #7a2222; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 4px; border: 1px solid #993333; }\n"
            "QPushButton:hover { background-color: #962c2c; }"
        )
        self.reject_btn.clicked.connect(lambda: self.save_verdict(ReviewStatus.REJECTED))

        actions_layout.addWidget(self.accept_btn)
        actions_layout.addWidget(self.reject_btn)
        verdict_layout.addLayout(actions_layout)

        left_layout.addWidget(verdict_box)
        left_layout.addStretch()

        main_splitter.addWidget(left_widget)

        # ----------------- Center Column: Interactive Canvas -----------------
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(2, 2, 2, 2)
        center_layout.setSpacing(4)

        # Toolbar
        self.toolbar = QToolBar("标注工具栏")
        self.toolbar.setStyleSheet("QToolBar { background-color: #262626; border: 1px solid #3c3c3c; }")

        self.act_select = QAction("选择/移动", self)
        self.act_select.setCheckable(True)
        self.act_select.setChecked(True)
        self.act_select.triggered.connect(lambda: self._set_canvas_mode(CanvasMode.SELECT))
        self.toolbar.addAction(self.act_select)

        self.act_create = QAction("+ 绘制新框", self)
        self.act_create.setCheckable(True)
        self.act_create.triggered.connect(lambda: self._set_canvas_mode(CanvasMode.CREATE))
        self.toolbar.addAction(self.act_create)

        self.toolbar.addSeparator()

        self.act_delete = QAction("删除选中框 (Del)", self)
        self.act_delete.triggered.connect(self._delete_selected_box)
        self.toolbar.addAction(self.act_delete)

        self.act_reload_boxes = QAction("还原自动检测框", self)
        self.act_reload_boxes.triggered.connect(self._reload_boxes)
        self.toolbar.addAction(self.act_reload_boxes)

        self.toolbar.addSeparator()

        self.act_zoom_in = QAction("放大 (+)", self)
        self.act_zoom_in.triggered.connect(lambda: self.canvas.zoom(1.2))
        self.toolbar.addAction(self.act_zoom_in)

        self.act_zoom_out = QAction("缩小 (-)", self)
        self.act_zoom_out.triggered.connect(lambda: self.canvas.zoom(1.0 / 1.2))
        self.toolbar.addAction(self.act_zoom_out)

        self.act_fit = QAction("自适应", self)
        self.act_fit.triggered.connect(self._fit_canvas)
        self.toolbar.addAction(self.act_fit)

        self.act_reset_zoom = QAction("1:1", self)
        self.act_reset_zoom.triggered.connect(self._reset_canvas_zoom)
        self.toolbar.addAction(self.act_reset_zoom)

        self.toolbar.addSeparator()

        self.act_guide = QAction("操作指引 (F1)", self)
        self.act_guide.setToolTip("打开独立操作指引与快捷键速查窗口")
        self.act_guide.triggered.connect(self.show_guide_dialog)
        self.toolbar.addAction(self.act_guide)

        center_layout.addWidget(self.toolbar)

        # Canvas
        self.canvas = AnnotationCanvas(self)
        self.canvas.box_selected.connect(self._on_box_selected)
        self.canvas.boxes_changed.connect(self._on_boxes_changed)
        self.canvas.shortcut_slot_requested.connect(self._on_shortcut_slot)
        center_layout.addWidget(self.canvas)

        # Bottom info bar on canvas
        self.box_detail_label = QLabel("未选中任何战场单位框 (提示: 可按数字键 1~6 快捷切换卡槽或标注选框)")
        self.box_detail_label.setStyleSheet("font-size: 11px; color: #88bbff; padding: 2px 6px;")
        center_layout.addWidget(self.box_detail_label)

        main_splitter.addWidget(center_widget)

        # ----------------- Right Column: Roster & Enemy Palette -----------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        # Upper: Roster Panel (Step 2)
        self.roster_panel = RosterPanel(self)
        self.roster_panel.roster_changed.connect(self._on_roster_changed)
        right_layout.addWidget(self.roster_panel)

        # Lower: Enemy Palette (Compact thumbnails)
        self.enemy_palette = EnemyPalette(self.workspace, icon_size=32, parent=self)
        self.enemy_palette.enemy_selected.connect(self._on_enemy_palette_selected)
        right_layout.addWidget(self.enemy_palette)

        main_splitter.addWidget(right_widget)

        # Adjust column stretch
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 6)
        main_splitter.setStretchFactor(2, 3)

        self.setCentralWidget(main_splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 | 提示: 按 F1 查看独立操作指引与快捷键说明")

    def _init_shortcuts(self) -> None:
        # A: prev, D: next
        QShortcut(QKeySequence("A"), self, self.prev_sample)
        QShortcut(QKeySequence("D"), self, self.next_sample)
        # V: Select mode, R: Create mode
        QShortcut(QKeySequence("V"), self, lambda: self._set_canvas_mode(CanvasMode.SELECT))
        QShortcut(QKeySequence("R"), self, lambda: self._set_canvas_mode(CanvasMode.CREATE))
        # F1: Operation guide
        QShortcut(QKeySequence("F1"), self, self.show_guide_dialog)
        # Slots 1..6 shortcuts
        for i in range(6):
            key = str(i + 1)
            QShortcut(QKeySequence(key), self, lambda idx=i: self._on_shortcut_slot(idx))

    def show_guide_dialog(self) -> None:
        """Open or focus the independent operation guide window."""
        if self.guide_dialog is None:
            self.guide_dialog = InstructionGuideDialog(self)
        self.guide_dialog.show()
        self.guide_dialog.raise_()
        self.guide_dialog.activateWindow()

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
        self._apply_filter()
        self.jump_to_filtered_index(0)

    def _apply_filter(self) -> None:
        filter_idx = self.filter_combo.currentIndex()
        status_map = {
            1: ReviewStatus.PENDING,
            2: ReviewStatus.ACCEPTED,
            3: ReviewStatus.REJECTED,
        }
        target_status = status_map.get(filter_idx)
        self.filtered_indices = [
            i for i, s in enumerate(self.samples) if target_status is None or s.review_status == target_status
        ]

        count = len(self.filtered_indices)
        self.jump_spin.blockSignals(True)
        self.jump_spin.setRange(1, max(1, count))
        self.jump_spin.setValue(1)
        self.jump_spin.blockSignals(False)

    def _on_filter_changed(self) -> None:
        self._apply_filter()
        self.jump_to_filtered_index(0)

    def _on_jump_changed(self, val: int) -> None:
        self.jump_to_filtered_index(val - 1)

    def get_current_sample(self) -> RoundSample | None:
        if not self.filtered_indices or self.current_idx_in_filtered >= len(self.filtered_indices):
            return None
        return self.samples[self.filtered_indices[self.current_idx_in_filtered]]

    def jump_to_filtered_index(self, filtered_idx: int) -> None:
        if not self.filtered_indices:
            self.sample_idx_label.setText("样本 0/0")
            self.sample_info_label.setText("没有符合过滤条件的样本")
            return

        self.current_idx_in_filtered = max(0, min(len(self.filtered_indices) - 1, filtered_idx))
        sample = self.get_current_sample()
        if sample is None:
            return

        total_filtered = len(self.filtered_indices)
        current_num = self.current_idx_in_filtered + 1
        self.sample_idx_label.setText(f"样本 {current_num} / {total_filtered}")

        self.jump_spin.blockSignals(True)
        self.jump_spin.setValue(current_num)
        self.jump_spin.blockSignals(False)

        # Update sample info
        status_color = (
            "#55ff77"
            if sample.review_status == ReviewStatus.ACCEPTED
            else ("#ff5555" if sample.review_status == ReviewStatus.REJECTED else "#ffbb33")
        )
        self.sample_info_label.setText(
            f"<b>样本ID:</b> <code>{sample.sample_id[:12]}...</code><br>"
            f"<b>回合:</b> 第 {sample.round_index} 回合<br>"
            f"<b>状态:</b> <span style='color: {status_color}; font-weight: bold;'>"
            f"{sample.review_status.value}</span><br>"
            f"<b>视频:</b> {sample.source.video_relpath.split('/')[-1]}"
        )

        if sample.failure_reasons:
            self.reasons_label.show()
            self.reasons_label.setText("[提示] 异常: " + ", ".join(sample.failure_reasons))
        else:
            self.reasons_label.hide()

        # Update winner radio
        self.winner_group.blockSignals(True)
        if sample.winner == Winner.LEFT:
            self.radio_left_win.setChecked(True)
        elif sample.winner == Winner.RIGHT:
            self.radio_right_win.setChecked(True)
        else:
            self.winner_group.setExclusive(False)
            self.radio_left_win.setChecked(False)
            self.radio_right_win.setChecked(False)
            self.winner_group.setExclusive(True)
        self.winner_group.blockSignals(False)

        # Load video timeline (Step 1)
        self.timeline_tuner.load_sample(sample)

        # Load layout image into canvas (Step 3)
        layout_rel = sample.evidence.layout
        if layout_rel:
            layout_path = self.workspace / layout_rel
            if layout_path.exists():
                self.canvas.load_image(str(layout_path))

        # Load roster panel (Step 2)
        self.roster_panel.load_from_sample(sample, self.enemy_palette)

        # Load boxes on canvas (Step 3)
        self.canvas.load_boxes_from_sample(sample, self.enemy_palette)

        # Sync quotas
        self._sync_quotas()

        self.status_bar.showMessage(f"已加载样本: {sample.sample_id}", 3000)

    def prev_sample(self) -> None:
        if self.current_idx_in_filtered > 0:
            self.jump_to_filtered_index(self.current_idx_in_filtered - 1)

    def next_sample(self) -> None:
        if self.current_idx_in_filtered < len(self.filtered_indices) - 1:
            self.jump_to_filtered_index(self.current_idx_in_filtered + 1)

    def _set_canvas_mode(self, mode: CanvasMode) -> None:
        self.canvas.set_mode(mode)
        self.act_select.setChecked(mode == CanvasMode.SELECT)
        self.act_create.setChecked(mode == CanvasMode.CREATE)

    def _delete_selected_box(self) -> None:
        self.canvas.delete_selected_box()

    def _reload_boxes(self) -> None:
        sample = self.get_current_sample()
        if sample:
            self.canvas.load_boxes_from_sample(sample, self.enemy_palette)
            self._sync_quotas()

    def _fit_canvas(self) -> None:
        self.canvas.fit_in_view()

    def _reset_canvas_zoom(self) -> None:
        self.canvas.reset_zoom()

    def _on_box_selected(self, box: AnnotationBoxItem | None) -> None:
        if box is None:
            self.box_detail_label.setText("未选中任何战场单位框 (提示: 可按数字键 1~6 快捷切换卡槽或标注选框)")
            return
        side_tag = "左方" if box.side == "left" else "右方"
        r = box.get_scene_rect()
        self.box_detail_label.setText(
            f"选中单位: [{side_tag}] ID:{box.enemy_id} {box.enemy_name} | "
            f"位置: ({int(r.left())}, {int(r.top())}, {int(r.width())}x{int(r.height())}) | "
            f"置信度: {box.confidence:.2f} | [按 1~6 赋给对应卡槽类别]"
        )

    def _on_boxes_changed(self) -> None:
        self._sync_quotas()

    def _on_roster_changed(self) -> None:
        self._sync_quotas()

    def _sync_quotas(self) -> None:
        left_counts, right_counts = self.canvas.count_units()
        self.roster_panel.update_quotas(left_counts, right_counts)

    def _on_enemy_palette_selected(self, enemy_id: int) -> None:
        """When an icon in palette is clicked, assign to currently active slot in roster panel."""
        self.roster_panel.assign_enemy_to_active_slot(enemy_id, self.enemy_palette)
        # If a box on canvas is also selected, assign to it as well!
        selected_box = self.canvas.get_selected_box()
        if selected_box and enemy_id > 0:
            name = self.enemy_palette.get_enemy_name(enemy_id)
            self.canvas.assign_to_selected_box(selected_box.side, enemy_id, name)

    def _on_shortcut_slot(self, slot_idx: int) -> None:
        """Called when user presses keys 1..6."""
        if not (0 <= slot_idx < len(self.roster_panel.slots)):
            return
        self.roster_panel.set_active_slot(slot_idx)
        slot = self.roster_panel.slots[slot_idx]

        # If a box is selected on canvas, bind that slot's enemy and side to it!
        selected_box = self.canvas.get_selected_box()
        if selected_box and slot.enemy_id > 0:
            self.canvas.assign_to_selected_box(slot.side, slot.enemy_id, slot.enemy_name)

    def _on_evidence_updated(self, frame_type: str, timestamp: float, frame_bgr: object) -> None:
        """Triggered when user clicks 'Save as Evidence Frame' in timeline tuner."""
        sample = self.get_current_sample()
        if sample is None:
            return

        self.status_bar.showMessage(f"已更新 {frame_type.upper()} 证据帧时间戳: {timestamp:.3f}s", 4000)

        # If layout frame was modified, reload canvas image
        if frame_type == "layout" and sample.evidence.layout:
            layout_path = self.workspace / sample.evidence.layout
            if layout_path.exists():
                self.canvas.load_image(str(layout_path))

    def save_verdict(self, status: ReviewStatus) -> None:
        sample = self.get_current_sample()
        if sample is None:
            return

        # Determine winner
        winner = Winner.LEFT if self.radio_left_win.isChecked() else (
            Winner.RIGHT if self.radio_right_win.isChecked() else None
        )

        left_roster, right_roster = self.roster_panel.get_rosters()
        left_units, right_units = self.canvas.get_units_by_side()

        # Build candidate sample
        candidate_dict = sample.model_dump(mode="json")
        candidate_dict["left"] = SideData(roster=left_roster, units=left_units).model_dump(mode="json")
        candidate_dict["right"] = SideData(roster=right_roster, units=right_units).model_dump(mode="json")
        candidate_dict["winner"] = winner.value if winner else None
        candidate_dict["review_status"] = status.value

        if status == ReviewStatus.ACCEPTED:
            candidate_dict["failure_reasons"] = []
            candidate_dict["winner_confidence"] = 1.0

            # Perform schema validation
            try:
                validated_sample = RoundSample.model_validate(candidate_dict)
            except Exception as exc:
                msg = (
                    f"无法通过审核：\n{exc}\n\n"
                    "请确保：\n1. 已选定获胜方\n2. 左右卡槽均已填入敌人\n"
                    "3. 卡槽要求总数与画布拉框数量完全一致"
                )
                QMessageBox.warning(self, "审核校验未通过", msg)
                return
        else:
            validated_sample = RoundSample.model_validate(candidate_dict)

        # Update in-memory sample
        sample_idx_in_all = self.filtered_indices[self.current_idx_in_filtered]
        self.samples[sample_idx_in_all] = validated_sample

        # Save to corrections.jsonl
        note = self.note_edit.text().strip()
        self.store.save(ReviewCorrection(sample=validated_sample, note=note))

        action_name = "通过" if status == ReviewStatus.ACCEPTED else "驳回"
        self.status_bar.showMessage(f"已保存样本 {validated_sample.sample_id[:8]} ({action_name})", 3000)

        # Advance to next sample automatically
        if self.current_idx_in_filtered < len(self.filtered_indices) - 1:
            self.next_sample()
        else:
            self.jump_to_filtered_index(self.current_idx_in_filtered)
