"""PyQt6-based GUI for offline review and annotation in MaaDuelChannel."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from maa_duel.gui.canvas import AnnotationCanvas
from maa_duel.gui.enemy_palette import EnemyPalette
from maa_duel.gui.reviewer_window import ReviewerMainWindow
from maa_duel.gui.roster_panel import RosterPanel
from maa_duel.gui.video_timeline import VideoTimelineFineTuner

__all__ = [
    "AnnotationCanvas",
    "EnemyPalette",
    "ReviewerMainWindow",
    "RosterPanel",
    "VideoTimelineFineTuner",
    "launch_local_review_qt",
]


def launch_local_review_qt(workspace: Path, video_dir: Path | None = None) -> None:
    """Launch desktop PyQt6 review application."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = ReviewerMainWindow(workspace, video_dir=video_dir)
    window.show()
    app.exec()
