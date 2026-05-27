#!/usr/bin/env python3
"""
capture_and_playback.py — Main launcher menu.

Shows a two-button window that opens either the guided data-collection
screen or the recording viewer. Each sub-window runs in the same Qt
application instance; the menu closes when a sub-window is opened.

Usage:
    python capture_and_playback.py
"""

import sys
from typing import Dict

from PySide6.QtWidgets import QApplication

from core.window.empty_window import EmptyWindow
import collect_webcam_guided
import view_recordings


def main() -> None:
    """Create the Qt application and show the main menu window."""
    app = QApplication(sys.argv)
    app.setApplicationName("SignLanguageCollector")

    # Holds open sub-windows so they are not garbage-collected
    window_holder: Dict[str, object] = {}

    menu = EmptyWindow(width=586, height=160, frame_processor=None, paint_fps=False)

    def open_viewer() -> None:
        """Close the menu and launch the recording viewer."""
        menu.close()
        window_holder["win_vr"] = view_recordings.run(app)

    menu.add_button(
        "viewer",
        text="🎬Ver grabaciones",
        action=open_viewer,
        color=EmptyWindow.COLORS["blue"],
        hover_color=EmptyWindow.COLORS["blue_hover"],
        width=260,
        height=120,
        txt_size=15,
        alignment="center"
    )

    def open_recorder() -> None:
        """Close the menu and launch the guided webcam recorder."""
        menu.close()
        window_holder["win_cwg"] = collect_webcam_guided.run(app)

    menu.add_button(
        "recorder",
        text="📷Grabar",
        action=open_recorder,
        color=EmptyWindow.COLORS["orange"],
        hover_color=EmptyWindow.COLORS["orange_hover"],
        width=260,
        height=120,
        txt_size=15,
        alignment="center"
    )

    menu.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
