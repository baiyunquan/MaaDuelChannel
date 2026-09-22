from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from maa_duel.schema import AnnotationSource, BoundingBox, RoundSample, UnitDetection

if TYPE_CHECKING:
    from maa_duel.gui.enemy_palette import EnemyPalette


class HandlePosition(Enum):
    NONE = 0
    TOP_LEFT = 1
    TOP = 2
    TOP_RIGHT = 3
    RIGHT = 4
    BOTTOM_RIGHT = 5
    BOTTOM = 6
    BOTTOM_LEFT = 7
    LEFT = 8


HANDLE_SIZE = 7.0


class AnnotationBoxItem(QGraphicsRectItem):
    """Interactive bounding box item on the canvas with 8 resize handles."""

    def __init__(
        self,
        rect: QRectF,
        side: str,
        enemy_id: int,
        enemy_name: str = "",
        confidence: float = 1.0,
        img_bounds: QRectF | None = None,
    ) -> None:
        super().__init__(rect)
        self.side = side  # "left" or "right"
        self.enemy_id = enemy_id
        self.enemy_name = enemy_name or f"敌人 {enemy_id}"
        self.confidence = confidence
        self.img_bounds = img_bounds or QRectF(0, 0, 1920, 1080)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._active_handle = HandlePosition.NONE
        self._drag_start_pos: QPointF | None = None
        self._drag_start_rect: QRectF | None = None

    def get_handle_rect(self, handle: HandlePosition) -> QRectF:
        r = self.rect()
        hs = HANDLE_SIZE
        if handle == HandlePosition.TOP_LEFT:
            return QRectF(r.left() - hs / 2, r.top() - hs / 2, hs, hs)
        if handle == HandlePosition.TOP:
            return QRectF(r.center().x() - hs / 2, r.top() - hs / 2, hs, hs)
        if handle == HandlePosition.TOP_RIGHT:
            return QRectF(r.right() - hs / 2, r.top() - hs / 2, hs, hs)
        if handle == HandlePosition.RIGHT:
            return QRectF(r.right() - hs / 2, r.center().y() - hs / 2, hs, hs)
        if handle == HandlePosition.BOTTOM_RIGHT:
            return QRectF(r.right() - hs / 2, r.bottom() - hs / 2, hs, hs)
        if handle == HandlePosition.BOTTOM:
            return QRectF(r.center().x() - hs / 2, r.bottom() - hs / 2, hs, hs)
        if handle == HandlePosition.BOTTOM_LEFT:
            return QRectF(r.left() - hs / 2, r.bottom() - hs / 2, hs, hs)
        if handle == HandlePosition.LEFT:
            return QRectF(r.left() - hs / 2, r.center().y() - hs / 2, hs, hs)
        return QRectF()

    def get_handle_at(self, pos: QPointF) -> HandlePosition:
        if not self.isSelected():
            return HandlePosition.NONE
        for h in (
            HandlePosition.TOP_LEFT,
            HandlePosition.TOP,
            HandlePosition.TOP_RIGHT,
            HandlePosition.RIGHT,
            HandlePosition.BOTTOM_RIGHT,
            HandlePosition.BOTTOM,
            HandlePosition.BOTTOM_LEFT,
            HandlePosition.LEFT,
        ):
            if self.get_handle_rect(h).contains(pos):
                return h
        return HandlePosition.NONE

    def hoverMoveEvent(self, event) -> None:  # noqa: N802
        if self.isSelected():
            h = self.get_handle_at(event.pos())
            if h in (HandlePosition.TOP_LEFT, HandlePosition.BOTTOM_RIGHT):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif h in (HandlePosition.TOP_RIGHT, HandlePosition.BOTTOM_LEFT):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif h in (HandlePosition.TOP, HandlePosition.BOTTOM):
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif h in (HandlePosition.LEFT, HandlePosition.RIGHT):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.isSelected():
            h = self.get_handle_at(event.pos())
            if h != HandlePosition.NONE:
                self._active_handle = h
                self._drag_start_pos = event.scenePos()
                self._drag_start_rect = self.rect()
                event.accept()
                return
        self._active_handle = HandlePosition.NONE
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._active_handle != HandlePosition.NONE and self._drag_start_pos and self._drag_start_rect:
            diff = event.scenePos() - self._drag_start_pos
            orig = self._drag_start_rect
            x1, y1, x2, y2 = orig.left(), orig.top(), orig.right(), orig.bottom()

            h = self._active_handle
            if h in (HandlePosition.TOP_LEFT, HandlePosition.LEFT, HandlePosition.BOTTOM_LEFT):
                x1 = min(orig.right() - 8, orig.left() + diff.x())
            if h in (HandlePosition.TOP_LEFT, HandlePosition.TOP, HandlePosition.TOP_RIGHT):
                y1 = min(orig.bottom() - 8, orig.top() + diff.y())
            if h in (HandlePosition.TOP_RIGHT, HandlePosition.RIGHT, HandlePosition.BOTTOM_RIGHT):
                x2 = max(orig.left() + 8, orig.right() + diff.x())
            if h in (HandlePosition.BOTTOM_LEFT, HandlePosition.BOTTOM, HandlePosition.BOTTOM_RIGHT):
                y2 = max(orig.top() + 8, orig.bottom() + diff.y())

            # Clamp to image bounds
            if self.img_bounds:
                pos = self.pos()
                x1 = max(-pos.x(), min(self.img_bounds.width() - pos.x(), x1))
                y1 = max(-pos.y(), min(self.img_bounds.height() - pos.y(), y1))
                x2 = max(-pos.x(), min(self.img_bounds.width() - pos.x(), x2))
                y2 = max(-pos.y(), min(self.img_bounds.height() - pos.y(), y2))

            self.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            self.scene().update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._active_handle != HandlePosition.NONE:
            self._active_handle = HandlePosition.NONE
            self._drag_start_pos = None
            self._drag_start_rect = None
            self.scene().update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802
        r = self.rect()
        is_left = self.side == "left"
        base_color = QColor(255, 140, 0) if is_left else QColor(0, 170, 255)

        # Semi-transparent fill
        fill_color = QColor(base_color.red(), base_color.green(), base_color.blue(), 40)
        painter.fillRect(r, QBrush(fill_color))

        # Border
        pen = QPen(base_color, 2 if not self.isSelected() else 3)
        if self.isSelected():
            pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawRect(r)

        # Label at top
        tag = f"[{'左' if is_left else '右'}] {self.enemy_id}:{self.enemy_name}"
        font = QFont("Sans", 9, QFont.Weight.Bold)
        painter.setFont(font)

        label_rect = QRectF(r.left(), r.top() - 18, max(100.0, len(tag) * 11.0), 18)
        # Keep label on screen
        if label_rect.top() < 0:
            label_rect.moveTop(r.top())

        painter.fillRect(label_rect, QBrush(QColor(10, 10, 10, 200)))
        painter.setPen(QPen(base_color, 1))
        painter.drawRect(label_rect)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, tag)

        # Ground anchor at bottom-center
        anchor_x = r.center().x()
        anchor_y = r.bottom()
        anchor_radius = 4.0
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setBrush(QBrush(QColor(255, 235, 0)))  # bright yellow anchor
        painter.drawEllipse(QPointF(anchor_x, anchor_y), anchor_radius, anchor_radius)

        # Draw handles if selected
        if self.isSelected():
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QBrush(base_color))
            for h in (
                HandlePosition.TOP_LEFT,
                HandlePosition.TOP,
                HandlePosition.TOP_RIGHT,
                HandlePosition.RIGHT,
                HandlePosition.BOTTOM_RIGHT,
                HandlePosition.BOTTOM,
                HandlePosition.BOTTOM_LEFT,
                HandlePosition.LEFT,
            ):
                painter.drawRect(self.get_handle_rect(h))

    def get_scene_rect(self) -> QRectF:
        """Get absolute bounding rect in scene coordinates."""
        return self.mapRectToScene(self.rect())

    def to_unit_detection(self, img_w: float, img_h: float) -> UnitDetection:
        sr = self.get_scene_rect()
        x1 = max(0.0, min(1.0, sr.left() / img_w))
        y1 = max(0.0, min(1.0, sr.top() / img_h))
        x2 = max(0.0, min(1.0, sr.right() / img_w))
        y2 = max(0.0, min(1.0, sr.bottom() / img_h))

        # Ensure valid coordinates
        if x1 >= x2:
            x2 = min(1.0, x1 + 0.01)
        if y1 >= y2:
            y2 = min(1.0, y1 + 0.01)

        return UnitDetection(
            enemy_id=self.enemy_id,
            x=(x1 + x2) / 2.0,
            y=y2,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            confidence=self.confidence,
            source=AnnotationSource.MANUAL,
        )


class CanvasMode(Enum):
    SELECT = 0
    CREATE = 1


class AnnotationCanvas(QGraphicsView):
    """High-resolution canvas with zoom/pan and bounding box editing."""

    box_selected = pyqtSignal(object)  # AnnotationBoxItem or None
    boxes_changed = pyqtSignal()
    shortcut_slot_requested = pyqtSignal(int)  # 0..5 slot index requested via key 1..6

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.mode = CanvasMode.SELECT
        self.bg_item: QGraphicsPixmapItem | None = None
        self.img_size = QSize(1920, 1080)

        # Drawing state
        self._is_drawing = False
        self._draw_start_pos: QPointF | None = None
        self._temp_rect_item: QGraphicsRectItem | None = None

        # Pan state
        self._is_panning = False
        self._pan_start_pos: QPointF | None = None

        self._setup_view()

    def _setup_view(self) -> None:
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setStyleSheet("QGraphicsView { background-color: #181818; border: 1px solid #333; }")

        self.scene.selectionChanged.connect(self._on_selection_changed)

    def set_mode(self, mode: CanvasMode) -> None:
        self.mode = mode
        if mode == CanvasMode.CREATE:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def load_image(self, img_path: str) -> bool:
        """Load background image from file path."""
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            return False

        self.img_size = pixmap.size()
        self.scene.clear()

        self.bg_item = self.scene.addPixmap(pixmap)
        self.bg_item.setZValue(-10)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.fit_in_view()
        return True

    def load_image_from_qimage(self, qimage: QImage) -> None:
        """Load background from existing QImage."""
        pixmap = QPixmap.fromImage(qimage)
        self.img_size = pixmap.size()
        self.scene.clear()

        self.bg_item = self.scene.addPixmap(pixmap)
        self.bg_item.setZValue(-10)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.fit_in_view()

    def fit_in_view(self) -> None:
        if self.bg_item:
            self.fitInView(self.bg_item, Qt.AspectRatioMode.KeepAspectRatio)

    def reset_zoom(self) -> None:
        self.resetTransform()

    def zoom(self, factor: float) -> None:
        self.scale(factor, factor)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
        self.zoom(factor)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # Middle button or Space+Left button for panning
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        ):
            self._is_panning = True
            self._pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if self.mode == CanvasMode.CREATE and event.button() == Qt.MouseButton.LeftButton:
            self._is_drawing = True
            self._draw_start_pos = self.mapToScene(event.pos())
            self._temp_rect_item = QGraphicsRectItem()
            self._temp_rect_item.setPen(QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine))
            self._temp_rect_item.setBrush(QBrush(QColor(255, 255, 0, 30)))
            self.scene.addItem(self._temp_rect_item)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._is_panning and self._pan_start_pos:
            delta = event.pos() - self._pan_start_pos
            self._pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if self._is_drawing and self._draw_start_pos and self._temp_rect_item:
            cur = self.mapToScene(event.pos())
            x1 = min(self._draw_start_pos.x(), cur.x())
            y1 = min(self._draw_start_pos.y(), cur.y())
            x2 = max(self._draw_start_pos.x(), cur.x())
            y2 = max(self._draw_start_pos.y(), cur.y())
            self._temp_rect_item.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        if self._is_drawing and self._draw_start_pos and self._temp_rect_item:
            self._is_drawing = False
            cur = self.mapToScene(event.pos())
            x1 = max(0.0, min(self._draw_start_pos.x(), cur.x()))
            y1 = max(0.0, min(self._draw_start_pos.y(), cur.y()))
            x2 = min(float(self.img_size.width()), max(self._draw_start_pos.x(), cur.x()))
            y2 = min(float(self.img_size.height()), max(self._draw_start_pos.y(), cur.y()))

            self.scene.removeItem(self._temp_rect_item)
            self._temp_rect_item = None

            # Minimum size check (at least 10x10 px)
            if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                side = "left" if (x1 + x2) / 2.0 < (self.img_size.width() / 2.0) else "right"
                new_box = AnnotationBoxItem(
                    rect=rect,
                    side=side,
                    enemy_id=1,
                    enemy_name="新单位",
                    confidence=1.0,
                    img_bounds=QRectF(0, 0, self.img_size.width(), self.img_size.height()),
                )
                self.scene.addItem(new_box)
                self.scene.clearSelection()
                new_box.setSelected(True)
                self.boxes_changed.emit()

            event.accept()
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_box()
            event.accept()
            return

        # Number keys 1..6: request assign to active slot or switch slot
        key = event.key()
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_6:
            slot_idx = key - Qt.Key.Key_1
            self.shortcut_slot_requested.emit(slot_idx)
            event.accept()
            return

        super().keyPressEvent(event)

    def _on_selection_changed(self) -> None:
        items = self.scene.selectedItems()
        box = next((it for it in items if isinstance(it, AnnotationBoxItem)), None)
        self.box_selected.emit(box)

    def get_selected_box(self) -> AnnotationBoxItem | None:
        items = self.scene.selectedItems()
        return next((it for it in items if isinstance(it, AnnotationBoxItem)), None)

    def get_all_boxes(self) -> list[AnnotationBoxItem]:
        return [it for it in self.scene.items() if isinstance(it, AnnotationBoxItem)]

    def delete_selected_box(self) -> None:
        selected = self.get_selected_box()
        if selected:
            self.scene.removeItem(selected)
            self.boxes_changed.emit()

    def assign_to_selected_box(self, side: str, enemy_id: int, enemy_name: str) -> None:
        selected = self.get_selected_box()
        if selected:
            selected.side = side
            selected.enemy_id = enemy_id
            selected.enemy_name = enemy_name
            selected.update()
            self.boxes_changed.emit()

    def load_boxes_from_sample(self, sample: RoundSample, palette: EnemyPalette) -> None:
        """Load boxes from sample's left and right unit detections."""
        # Remove existing boxes
        for box in self.get_all_boxes():
            self.scene.removeItem(box)

        w = float(self.img_size.width())
        h = float(self.img_size.height())
        bounds = QRectF(0, 0, w, h)

        for side_name, units in (("left", sample.left.units), ("right", sample.right.units)):
            for unit in units:
                rect = QRectF(
                    unit.bbox.x1 * w,
                    unit.bbox.y1 * h,
                    (unit.bbox.x2 - unit.bbox.x1) * w,
                    (unit.bbox.y2 - unit.bbox.y1) * h,
                )
                name = palette.get_enemy_name(unit.enemy_id)
                item = AnnotationBoxItem(
                    rect=rect,
                    side=side_name,
                    enemy_id=unit.enemy_id,
                    enemy_name=name,
                    confidence=unit.confidence,
                    img_bounds=bounds,
                )
                self.scene.addItem(item)

        self.boxes_changed.emit()

    def get_units_by_side(self) -> tuple[list[UnitDetection], list[UnitDetection]]:
        """Return (left_units, right_units) from current canvas boxes."""
        left: list[UnitDetection] = []
        right: list[UnitDetection] = []
        w = float(self.img_size.width())
        h = float(self.img_size.height())

        for box in self.get_all_boxes():
            unit = box.to_unit_detection(w, h)
            if box.side == "left":
                left.append(unit)
            else:
                right.append(unit)

        # Sort stably
        left.sort(key=lambda u: (u.enemy_id, u.x, u.y))
        right.sort(key=lambda u: (u.enemy_id, u.x, u.y))
        return left, right

    def count_units(self) -> tuple[dict[int, int], dict[int, int]]:
        """Count enemies by side: (left_counts, right_counts)."""
        left_counts: dict[int, int] = {}
        right_counts: dict[int, int] = {}
        for box in self.get_all_boxes():
            if box.side == "left":
                left_counts[box.enemy_id] = left_counts.get(box.enemy_id, 0) + 1
            else:
                right_counts[box.enemy_id] = right_counts.get(box.enemy_id, 0) + 1
        return left_counts, right_counts

