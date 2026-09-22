from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from maa_duel.schema import RosterEntry, RoundSample

if TYPE_CHECKING:
    from maa_duel.gui.enemy_palette import EnemyPalette


class SlotWidget(QFrame):
    """Widget representing a single round card slot (0 to 5)."""

    clicked = pyqtSignal(int)
    count_changed = pyqtSignal(int, int)  # (slot_index, new_count)
    cleared = pyqtSignal(int)

    def __init__(self, slot_index: int, side: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slot_index = slot_index
        self.side = side  # "left" or "right"
        self.enemy_id: int = 0
        self.enemy_name: str = "空"
        self.count: int = 0
        self.is_active: bool = False
        self.drawn_count: int = 0

        self.setObjectName("SlotWidget")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._init_ui()
        self._update_style()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Header: Slot label & Clear button
        header_layout = QHBoxLayout()
        side_tag = "左" if self.side == "left" else "右"
        num = (self.slot_index % 3) + 1
        self.header_label = QLabel(f"[{side_tag}{num}] 快捷键 {self.slot_index + 1}")
        self.header_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #dddddd;")
        header_layout.addWidget(self.header_label)

        header_layout.addStretch()

        self.clear_btn = QPushButton("X")
        self.clear_btn.setFixedSize(18, 18)
        self.clear_btn.setToolTip("清空卡槽")
        self.clear_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #888888; border: none; font-size: 12px; }"
            "QPushButton:hover { color: #ff5555; }"
        )
        self.clear_btn.clicked.connect(self._on_clear)
        header_layout.addWidget(self.clear_btn)
        layout.addLayout(header_layout)

        # Center: Avatar & Info
        body_layout = QHBoxLayout()
        body_layout.setSpacing(8)

        self.avatar_label = QLabel()
        self.avatar_label.setFixedSize(36, 36)
        self.avatar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.avatar_label.setStyleSheet("background-color: #2a2a2a; border-radius: 4px; border: 1px solid #444;")
        body_layout.addWidget(self.avatar_label)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        self.name_label = QLabel("空槽位")
        self.name_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #ffffff;")
        info_layout.addWidget(self.name_label)

        self.id_label = QLabel("ID: --")
        self.id_label.setStyleSheet("font-size: 10px; color: #888888;")
        info_layout.addWidget(self.id_label)

        body_layout.addLayout(info_layout)
        body_layout.addStretch()
        layout.addLayout(body_layout)

        # Bottom: Count SpinBox & Quota badge
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(6)

        count_label = QLabel("数量:")
        count_label.setStyleSheet("font-size: 11px; color: #aaaaaa;")
        bottom_layout.addWidget(count_label)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(0, 99)
        self.count_spin.setValue(0)
        self.count_spin.setFixedWidth(50)
        self.count_spin.setStyleSheet(
            "QSpinBox { background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; border-radius: 3px; }"
        )
        self.count_spin.valueChanged.connect(self._on_spin_changed)
        bottom_layout.addWidget(self.count_spin)

        bottom_layout.addStretch()

        self.quota_label = QLabel("0/0")
        self.quota_label.setStyleSheet(
            "QLabel { background-color: #333333; color: #888888; "
            "padding: 2px 5px; border-radius: 3px; font-size: 11px; }"
        )
        bottom_layout.addWidget(self.quota_label)

        layout.addLayout(bottom_layout)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.slot_index)
        super().mousePressEvent(event)

    def _on_clear(self) -> None:
        self.cleared.emit(self.slot_index)

    def _on_spin_changed(self, val: int) -> None:
        self.count = val
        self.count_changed.emit(self.slot_index, val)
        self.update_quota(self.drawn_count)

    def set_active(self, active: bool) -> None:
        self.is_active = active
        self._update_style()

    def set_enemy(self, enemy_id: int, name: str, pixmap: QPixmap | None, count: int = 1) -> None:
        self.enemy_id = enemy_id
        self.enemy_name = name
        self.count = count
        self.count_spin.blockSignals(True)
        self.count_spin.setValue(count)
        self.count_spin.blockSignals(False)

        if enemy_id > 0:
            self.name_label.setText(name)
            self.id_label.setText(f"ID: {enemy_id}")
            if pixmap:
                scaled = pixmap.scaled(
                    36, 36, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                self.avatar_label.setPixmap(scaled)
            else:
                self.avatar_label.setText(str(enemy_id))
        else:
            self.name_label.setText("空槽位")
            self.id_label.setText("ID: --")
            self.avatar_label.clear()
            self.avatar_label.setText("空")

        self.update_quota(self.drawn_count)
        self._update_style()

    def update_quota(self, drawn_count: int) -> None:
        self.drawn_count = drawn_count
        if self.enemy_id == 0 or self.count == 0:
            self.quota_label.setText("--")
            self.quota_label.setStyleSheet(
                "background-color: #2d2d2d; color: #777777; border-radius: 3px; font-size: 11px;"
            )
            return

        text = f"{self.drawn_count}/{self.count}"
        if self.drawn_count == self.count:
            self.quota_label.setText(f"[OK] {text}")
            self.quota_label.setStyleSheet(
                "background-color: #1b4d24; color: #55ff77; font-weight: bold; "
                "border-radius: 3px; font-size: 11px; padding: 2px 4px;"
            )
        elif self.drawn_count < self.count:
            self.quota_label.setText(f"[待补] {text}")
            self.quota_label.setStyleSheet(
                "background-color: #553e10; color: #ffbb33; "
                "border-radius: 3px; font-size: 11px; padding: 2px 4px;"
            )
        else:
            self.quota_label.setText(f"! {text}")
            self.quota_label.setStyleSheet(
                "background-color: #5a1e1e; color: #ff6666; font-weight: bold; "
                "border-radius: 3px; font-size: 11px; padding: 2px 4px;"
            )

    def _update_style(self) -> None:
        side_color = "#ff8800" if self.side == "left" else "#00aaff"
        side_bg = "#25201a" if self.side == "left" else "#1a2028"

        if self.is_active:
            border_css = f"border: 2px solid {side_color}; background-color: {side_bg};"
        else:
            border_css = "border: 1px solid #3c3c3c; background-color: #222222;"

        self.setStyleSheet(
            f"QFrame#SlotWidget {{ {border_css} border-radius: 6px; }}"
            f"QFrame#SlotWidget:hover {{ border: 1px solid {side_color}; }}"
        )


class RosterPanel(QWidget):
    """Panel managing 6 bottom card slots (Left 0..2, Right 3..5) and their quotas."""

    active_slot_changed = pyqtSignal(int)
    roster_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.active_slot_index: int = 0
        self.slots: list[SlotWidget] = []

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        group = QGroupBox("本局阵容卡槽 (共 6 槽，按数字键 1~6 切换)")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(6, 6, 6, 6)
        group_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        slots_layout = QVBoxLayout(container)
        slots_layout.setContentsMargins(2, 2, 2, 2)
        slots_layout.setSpacing(6)

        # Left Side Header
        left_label = QLabel("<- 左方阵容 (Slot 1..3)")
        left_label.setStyleSheet("color: #ffaa33; font-weight: bold; font-size: 12px; margin-top: 2px;")
        slots_layout.addWidget(left_label)

        # Slots 0, 1, 2
        for i in range(3):
            slot = SlotWidget(i, "left", self)
            slot.clicked.connect(self._on_slot_clicked)
            slot.count_changed.connect(self._on_slot_count_changed)
            slot.cleared.connect(self._on_slot_cleared)
            self.slots.append(slot)
            slots_layout.addWidget(slot)

        # Right Side Header
        right_label = QLabel("-> 右方阵容 (Slot 4..6)")
        right_label.setStyleSheet("color: #33bbff; font-weight: bold; font-size: 12px; margin-top: 8px;")
        slots_layout.addWidget(right_label)

        # Slots 3, 4, 5
        for i in range(3, 6):
            slot = SlotWidget(i, "right", self)
            slot.clicked.connect(self._on_slot_clicked)
            slot.count_changed.connect(self._on_slot_count_changed)
            slot.cleared.connect(self._on_slot_cleared)
            self.slots.append(slot)
            slots_layout.addWidget(slot)

        slots_layout.addStretch()
        scroll.setWidget(container)
        group_layout.addWidget(scroll)

        main_layout.addWidget(group)

        # Default active slot is 0
        self.set_active_slot(0)

    def _on_slot_clicked(self, slot_index: int) -> None:
        self.set_active_slot(slot_index)

    def _on_slot_count_changed(self, slot_index: int, count: int) -> None:
        self.roster_changed.emit()

    def _on_slot_cleared(self, slot_index: int) -> None:
        slot = self.slots[slot_index]
        slot.set_enemy(0, "空", None, count=0)
        self.roster_changed.emit()

    def set_active_slot(self, slot_index: int) -> None:
        if not (0 <= slot_index < len(self.slots)):
            return
        self.active_slot_index = slot_index
        for i, s in enumerate(self.slots):
            s.set_active(i == slot_index)
        self.active_slot_changed.emit(slot_index)

    def get_active_slot(self) -> SlotWidget:
        return self.slots[self.active_slot_index]

    def assign_enemy_to_active_slot(self, enemy_id: int, palette: EnemyPalette) -> None:
        slot = self.get_active_slot()
        if enemy_id == 0:
            slot.set_enemy(0, "空", None, count=0)
        else:
            name = palette.get_enemy_name(enemy_id)
            pixmap = palette.get_enemy_pixmap(enemy_id)
            current_count = slot.count if slot.count > 0 else 1
            slot.set_enemy(enemy_id, name, pixmap, count=current_count)
        self.roster_changed.emit()

    def load_from_sample(self, sample: RoundSample, palette: EnemyPalette) -> None:
        """Populate the 6 slots from the sample's left and right rosters."""
        # Reset all slots
        for s in self.slots:
            s.set_enemy(0, "空", None, count=0)

        # Left side: up to 3 entries
        for i, entry in enumerate(sample.left.roster[:3]):
            name = palette.get_enemy_name(entry.enemy_id)
            pix = palette.get_enemy_pixmap(entry.enemy_id)
            self.slots[i].set_enemy(entry.enemy_id, name, pix, count=entry.count)

        # Right side: up to 3 entries
        for i, entry in enumerate(sample.right.roster[:3]):
            name = palette.get_enemy_name(entry.enemy_id)
            pix = palette.get_enemy_pixmap(entry.enemy_id)
            self.slots[3 + i].set_enemy(entry.enemy_id, name, pix, count=entry.count)

        self.roster_changed.emit()

    def update_quotas(self, left_counts: dict[int, int], right_counts: dict[int, int]) -> None:
        """Update quota display for all slots based on actual counts drawn on canvas."""
        for i in range(3):
            slot = self.slots[i]
            slot.update_quota(left_counts.get(slot.enemy_id, 0))
        for i in range(3, 6):
            slot = self.slots[i]
            slot.update_quota(right_counts.get(slot.enemy_id, 0))

    def get_rosters(self) -> tuple[list[RosterEntry], list[RosterEntry]]:
        """Return (left_roster, right_roster) from the current slot state."""
        left: list[RosterEntry] = []
        for s in self.slots[:3]:
            if s.enemy_id > 0 and s.count > 0:
                left.append(RosterEntry(enemy_id=s.enemy_id, count=s.count, confidence=1.0))

        right: list[RosterEntry] = []
        for s in self.slots[3:]:
            if s.enemy_id > 0 and s.count > 0:
                right.append(RosterEntry(enemy_id=s.enemy_id, count=s.count, confidence=1.0))

        return left, right
