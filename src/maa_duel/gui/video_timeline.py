from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from maa_duel.schema import RoundSample


class VideoTimelineFineTuner(QWidget):
    """Component for reviewing and fine-tuning timestamps of the 3 evidence frames (Prep, Layout, End)."""

    # Emits (frame_type: "prep"|"layout"|"end", new_timestamp: float, image_bgr: np.ndarray)
    evidence_updated = pyqtSignal(str, float, object)

    def __init__(self, workspace: Path, video_dir: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.video_dir = video_dir

        self.current_sample: RoundSample | None = None
        self.cap: cv2.VideoCapture | None = None
        self.current_frame_type: str = "layout"  # "prep" | "layout" | "end"
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.duration_seconds: float = 0.0
        self.current_time_sec: float = 0.0
        self.current_frame_bgr: np.ndarray | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        box = QGroupBox("⏱️ 三证据帧时间定位与微调 (Step 1)")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(6, 6, 6, 6)
        box_layout.setSpacing(6)

        # 1. Frame Type Selector (Prep / Layout / End)
        tab_layout = QHBoxLayout()
        tab_layout.setSpacing(4)
        self.btn_group = QButtonGroup(self)

        self.btn_prep = QPushButton("1. 准备帧 (Prep)")
        self.btn_prep.setCheckable(True)
        self.btn_layout = QPushButton("2. 站位帧 (Layout)")
        self.btn_layout.setCheckable(True)
        self.btn_layout.setChecked(True)
        self.btn_end = QPushButton("3. 结算帧 (End)")
        self.btn_end.setCheckable(True)

        for btn, ftype in ((self.btn_prep, "prep"), (self.btn_layout, "layout"), (self.btn_end, "end")):
            btn.setStyleSheet(
                "QPushButton { padding: 4px 8px; font-size: 11px; background-color: #2a2a2a; "
                "color: #ccc; border: 1px solid #444; border-radius: 4px; }\n"
                "QPushButton:checked { background-color: #007acc; color: #fff; "
                "font-weight: bold; border: 1px solid #0099ff; }"
            )
            self.btn_group.addButton(btn)
            btn.clicked.connect(lambda _, ft=ftype: self._switch_frame_type(ft))
            tab_layout.addWidget(btn)

        box_layout.addLayout(tab_layout)

        # 2. Preview Thumbnail
        self.preview_label = QLabel("视频未加载")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFixedHeight(140)
        self.preview_label.setStyleSheet(
            "QLabel { background-color: #1a1a1a; border: 1px solid #333; "
            "border-radius: 4px; color: #666; font-size: 11px; }"
        )
        box_layout.addWidget(self.preview_label)

        # 3. Time status info
        self.time_info_label = QLabel("时间: 00:00.000 / 00:00.000 (帧 0)")
        self.time_info_label.setStyleSheet("color: #00d4ff; font-family: monospace; font-size: 11px;")
        box_layout.addWidget(self.time_info_label)

        # 4. Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(0)
        self.slider.setStyleSheet(
            """
            QSlider::groove:horizontal { height: 6px; background: #333; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #007acc; border-radius: 3px; }
            QSlider::handle:horizontal {
                background: #ffffff; width: 14px; margin-top: -4px; margin-bottom: -4px; border-radius: 7px;
            }
            """
        )
        self.slider.sliderMoved.connect(self._on_slider_moved)
        box_layout.addWidget(self.slider)

        # 5. Stepping Buttons
        step_layout = QHBoxLayout()
        step_layout.setSpacing(2)

        btn_specs = [
            ("-1.0s", -1.0),
            ("-0.1s", -0.1),
            ("-1帧", "frame_prev"),
            ("+1帧", "frame_next"),
            ("+0.1s", 0.1),
            ("+1.0s", 1.0),
        ]
        for text, delta in btn_specs:
            b = QPushButton(text)
            b.setStyleSheet(
                "QPushButton { padding: 3px 4px; font-size: 10px; background-color: #2e2e2e; "
                "color: #ddd; border: 1px solid #444; border-radius: 3px; }\n"
                "QPushButton:hover { background-color: #3e3e3e; color: #fff; }"
            )
            b.clicked.connect(lambda _, d=delta: self._step(d))
            step_layout.addWidget(b)

        box_layout.addLayout(step_layout)

        # 6. Save as Evidence Frame button
        self.save_btn = QPushButton("💾 设为当前证据帧并重新落盘")
        self.save_btn.setStyleSheet(
            "QPushButton { padding: 6px; font-size: 11px; font-weight: bold; "
            "background-color: #1e6b36; color: #ffffff; border: 1px solid #2e8b46; border-radius: 4px; }\n"
            "QPushButton:hover { background-color: #268544; }"
        )
        self.save_btn.clicked.connect(self._save_as_evidence)
        box_layout.addWidget(self.save_btn)

        main_layout.addWidget(box)

    def load_sample(self, sample: RoundSample) -> None:
        """Load a new round sample and open its source video."""
        self.current_sample = sample
        self._close_video()

        video_path = self._resolve_video_path(sample.source.video_relpath)
        if not video_path or not video_path.exists():
            self.preview_label.setText(f"未找到视频文件:\n{sample.source.video_relpath}")
            return

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            self.preview_label.setText(f"无法打开视频:\n{video_path.name}")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.total_frames / self.fps if self.fps > 0 else 0.0

        self.slider.setRange(0, max(1, self.total_frames - 1))

        # Default to layout frame timestamp
        self._switch_frame_type(self.current_frame_type)

    def _resolve_video_path(self, relpath: str) -> Path | None:
        """Resolve video path using video_dir or workspace."""
        if self.video_dir:
            cand = self.video_dir / relpath
            if cand.exists():
                return cand
        cand = self.workspace / relpath
        if cand.exists():
            return cand
        cand = Path(relpath)
        if cand.exists():
            return cand
        return None

    def _close_video(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _switch_frame_type(self, frame_type: str) -> None:
        self.current_frame_type = frame_type
        if self.current_sample is None or self.cap is None:
            return

        if frame_type == "prep":
            target_time = self.current_sample.timestamps.prep
            self.btn_prep.setChecked(True)
        elif frame_type == "layout":
            target_time = self.current_sample.timestamps.layout
            self.btn_layout.setChecked(True)
        else:
            target_time = self.current_sample.timestamps.battle_end
            self.btn_end.setChecked(True)

        self.seek_to_time(target_time)

    def seek_to_time(self, time_sec: float) -> None:
        if self.cap is None or self.total_frames <= 0:
            return
        frame_idx = max(0, min(self.total_frames - 1, int(round(time_sec * self.fps))))
        self.seek_to_frame(frame_idx)

    def seek_to_frame(self, frame_idx: int) -> None:
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        self.current_frame_bgr = frame
        self.current_time_sec = frame_idx / self.fps
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)

        # Update preview
        self._update_preview(frame)

        # Update label
        cur_min, cur_sec = divmod(self.current_time_sec, 60.0)
        dur_min, dur_sec = divmod(self.duration_seconds, 60.0)
        self.time_info_label.setText(
            f"[{self.current_frame_type.upper()}] {int(cur_min):02d}:{cur_sec:06.3f} / "
            f"{int(dur_min):02d}:{dur_sec:06.3f} (帧 {frame_idx}/{self.total_frames})"
        )

    def _on_slider_moved(self, frame_idx: int) -> None:
        self.seek_to_frame(frame_idx)

    def _step(self, delta: float | str) -> None:
        if self.cap is None:
            return
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if delta == "frame_prev":
            target_frame = max(0, current_frame - 1)
        elif delta == "frame_next":
            target_frame = min(self.total_frames - 1, current_frame + 1)
        elif isinstance(delta, (int, float)):
            target_frame = max(0, min(self.total_frames - 1, int(round((self.current_time_sec + delta) * self.fps))))
        else:
            return
        self.seek_to_frame(target_frame)

    def _update_preview(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def _save_as_evidence(self) -> None:
        """Overwrite the evidence frame image on disk and update sample timestamp."""
        if self.current_sample is None or self.current_frame_bgr is None:
            return

        frame_type = self.current_frame_type
        # Determine target file path
        relpath = getattr(self.current_sample.evidence, frame_type, None)
        if not relpath:
            sha_prefix = self.current_sample.source.video_sha256[:12]
            relpath = f"frames/{sha_prefix}/{self.current_sample.sample_id}-{frame_type}.jpg"
            setattr(self.current_sample.evidence, frame_type, relpath)

        target_path = self.workspace / relpath
        target_path.parent.mkdir(parents=True, exist_ok=True)

        ok, encoded = cv2.imencode(".jpg", self.current_frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            encoded.tofile(target_path)

        # Update timestamp in sample
        if frame_type == "prep":
            self.current_sample.timestamps.prep = self.current_time_sec
        elif frame_type == "layout":
            self.current_sample.timestamps.layout = self.current_time_sec
        elif frame_type == "end":
            self.current_sample.timestamps.battle_end = self.current_time_sec

        self.evidence_updated.emit(frame_type, self.current_time_sec, self.current_frame_bgr)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._close_video()
        super().closeEvent(event)
