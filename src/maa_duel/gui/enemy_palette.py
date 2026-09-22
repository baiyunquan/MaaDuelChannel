from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class EnemyPalette(QWidget):
    """Compact thumbnail palette for all enemies in the catalog."""

    enemy_selected = pyqtSignal(int)  # Emits enemy_id

    def __init__(self, workspace: Path, icon_size: int = 32, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.icon_size = icon_size
        self.enemies: dict[int, dict[str, str]] = {}
        self.pixmaps: dict[int, QPixmap] = {}

        self._init_ui()
        self.load_catalog()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        box = QGroupBox("敌人图谱 (点击赋予选中卡槽)")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(6, 6, 6, 6)
        box_layout.setSpacing(4)

        # Search Bar
        search_layout = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(" 搜索名称 / 原名 / ID...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._filter_enemies)
        search_layout.addWidget(self.search_edit)
        box_layout.addLayout(search_layout)

        # List Widget for Icons
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(self.icon_size, self.icon_size))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setGridSize(QSize(self.icon_size + 8, self.icon_size + 8))
        self.list_widget.setSpacing(2)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setStyleSheet(
            """
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
            QListWidget::item {
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 1px;
            }
            QListWidget::item:hover {
                background-color: #333333;
                border: 1px solid #00aaff;
            }
            QListWidget::item:selected {
                background-color: #005588;
                border: 1px solid #00d4ff;
            }
            """
        )
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemEntered.connect(self._on_item_entered)
        self.list_widget.setMouseTracking(True)
        box_layout.addWidget(self.list_widget)

        # Hover Info Label
        self.info_label = QLabel("鼠标悬停查看详情，点击选择")
        self.info_label.setStyleSheet("color: #aaaaaa; font-size: 11px; padding: 2px;")
        box_layout.addWidget(self.info_label)

        layout.addWidget(box)

    def load_catalog(self) -> None:
        self.list_widget.clear()
        self.enemies.clear()
        self.pixmaps.clear()

        # Add Empty Slot option first
        self._add_empty_slot_item()

        catalog_path = self.workspace / "assets" / "catalog.json"
        if not catalog_path.exists():
            return

        try:
            with open(catalog_path, encoding="utf-8") as f:
                data = json.load(f)
            raw_enemies = data.get("enemies", [])
        except Exception:
            return

        for e in sorted(raw_enemies, key=lambda x: int(x.get("enemy_id", 0))):
            enemy_id = int(e.get("enemy_id", 0))
            name = str(e.get("name", f"敌人 {enemy_id}"))
            orig = str(e.get("original_name", ""))
            self.enemies[enemy_id] = {"name": name, "original_name": orig}

            # Load thumbnail
            thumb_path = self.workspace / "assets" / "portraits" / f"{enemy_id:04d}" / "thumbnail.png"
            if thumb_path.exists():
                pix = QPixmap(str(thumb_path)).scaled(
                    self.icon_size,
                    self.icon_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            else:
                pix = self._create_fallback_icon(enemy_id, name)

            self.pixmaps[enemy_id] = pix

            item = QListWidgetItem()
            item.setIcon(QIcon(pix))
            item.setToolTip(f"ID {enemy_id}: {name}\n原名: {orig}")
            item.setData(Qt.ItemDataRole.UserRole, enemy_id)
            self.list_widget.addItem(item)

    def _add_empty_slot_item(self) -> None:
        """Add empty slot (ID 0) option."""
        self.enemies[0] = {"name": "空卡槽", "original_name": "empty"}
        pix = QPixmap(self.icon_size, self.icon_size)
        pix.fill(QColor(40, 40, 40))
        painter = QPainter(pix)
        painter.setPen(QColor(160, 160, 160))
        font = QFont("Sans", 8)
        painter.setFont(font)
        painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "空")
        painter.end()

        self.pixmaps[0] = pix
        item = QListWidgetItem()
        item.setIcon(QIcon(pix))
        item.setToolTip("ID 0: 空卡槽 (清空当前槽位)")
        item.setData(Qt.ItemDataRole.UserRole, 0)
        self.list_widget.addItem(item)

    def _create_fallback_icon(self, enemy_id: int, name: str) -> QPixmap:
        pix = QPixmap(self.icon_size, self.icon_size)
        pix.fill(QColor(50, 50, 60))
        painter = QPainter(pix)
        painter.setPen(QColor(220, 220, 220))
        font = QFont("Sans", 7)
        painter.setFont(font)
        painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, str(enemy_id))
        painter.end()
        return pix

    def _filter_enemies(self, text: str) -> None:
        query = text.strip().lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            enemy_id = item.data(Qt.ItemDataRole.UserRole)
            info = self.enemies.get(enemy_id, {})
            name = info.get("name", "").lower()
            orig = info.get("original_name", "").lower()

            match = (not query) or (str(enemy_id) == query) or (query in name) or (query in orig)
            item.setHidden(not match)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        enemy_id = item.data(Qt.ItemDataRole.UserRole)
        self.enemy_selected.emit(enemy_id)

    def _on_item_entered(self, item: QListWidgetItem) -> None:
        enemy_id = item.data(Qt.ItemDataRole.UserRole)
        info = self.enemies.get(enemy_id, {})
        name = info.get("name", "未知")
        orig = info.get("original_name", "")
        self.info_label.setText(f"选中目标: [ID {enemy_id}] {name} ({orig})")

    def get_enemy_name(self, enemy_id: int) -> str:
        info = self.enemies.get(enemy_id)
        if info:
            return info.get("name") or f"敌人 {enemy_id}"
        return f"ID {enemy_id}"

    def get_enemy_pixmap(self, enemy_id: int) -> QPixmap | None:
        return self.pixmaps.get(enemy_id)
