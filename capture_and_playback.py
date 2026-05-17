#!/usr/bin/env python3
import sys
from PySide6.QtWidgets import QApplication
from core.window.empty_window import EmptyWindow

import collect_webcam_guided
import view_recordings


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SignLanguageCollector")
    window_holder = {}

    menu = EmptyWindow(
        width=586,
        height=160,
        frame_processor=None,
        paint_fps = False
    )

    window_holder = {}

    # -------------------------
    # VIEW RECORDINGS
    # -------------------------
    def open_viewer():
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

    # -------------------------
    # RECORD WEB CAM
    # -------------------------
    def open_recorder():
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