"""PyQt6-based GUI for offline review and annotation in MaaDuelChannel."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from maa_duel.gui.canvas import AnnotationCanvas
from maa_duel.gui.enemy_palette import EnemyPalette
from maa_duel.gui.guide_dialog import InstructionGuideDialog
from maa_duel.gui.prep_roster_window import PrepRosterReviewWindow, SlotPairCardWidget
from maa_duel.gui.reviewer_window import ReviewerMainWindow
from maa_duel.gui.roster_panel import RosterPanel
from maa_duel.gui.video_timeline import VideoTimelineFineTuner

__all__ = [
    "AnnotationCanvas",
    "EnemyPalette",
    "InstructionGuideDialog",
    "PrepRosterReviewWindow",
    "ReviewerMainWindow",
    "RosterPanel",
    "SlotPairCardWidget",
    "VideoTimelineFineTuner",
    "launch_local_review_qt",
]


def launch_local_review_qt(
    workspace: Path,
    video_dir: Path | None = None,
    skip_prep_roster: bool = False,
    only_prep_roster: bool = False,
) -> None:
    """Launch desktop PyQt6 review application with two-stage flow."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    if skip_prep_roster:
        workbench = ReviewerMainWindow(workspace, video_dir=video_dir)
        workbench.show()
        app._active_window = workbench  # type: ignore[attr-defined]
    else:

        def open_workbench() -> None:
            workbench = ReviewerMainWindow(workspace, video_dir=video_dir)
            workbench.show()
            app._active_window = workbench  # type: ignore[attr-defined]

        prep_window = PrepRosterReviewWindow(
            workspace,
            video_dir=video_dir,
            on_proceed_to_workbench=None if only_prep_roster else open_workbench,
        )
        prep_window.show()
        app._active_window = prep_window  # type: ignore[attr-defined]

    app.exec()
