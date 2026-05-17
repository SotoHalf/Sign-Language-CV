#!/usr/bin/env python3
import sys
import os
import time
from PySide6.QtWidgets import QApplication
from core.window.empty_window import EmptyWindow
from core.processors.viewer_processor import ViewerProcessor
from PySide6.QtWidgets import QMessageBox
from core.utils import AppPaths


def run(app):
    DATA_PATH = AppPaths.path(os.getenv("DATA_PATH", "data/processed/"))

    try:
        processor = ViewerProcessor(DATA_PATH, fps=30)
    except RuntimeError as e:
        QMessageBox.critical(
            None,
            "Error loading data",
            str(e)
        )
        return

    window = EmptyWindow(
        width=800,
        height=600,
        frame_processor=processor,
    )

    # PLAY/PAUSE
    window.add_button(
        "play_pause",
        text="⏸️",
        action=processor.toggle_play,
        tooltip="Play / Pause",
        color=EmptyWindow.COLORS['orange'],
        hover_color=EmptyWindow.COLORS['orange_hover'],
        shortcut="Space",
        alignment="right",
        width=50
    )

    window.add_button(
        "reset",
        text="⟳",
        action=processor.reset,
        tooltip="Restart playback",
        alignment="right",
        width=50
    )

    # SIGN (label)
    window.add_button(
        "prev_sign",
        text="🡸 Sign",
        action=processor.prev_label,
        tooltip="Previous sign",
        alignment="right",
        width=70
    )

    window.add_button(
        "next_sign",
        text="🡺 Sign",
        action=processor.next_label,
        tooltip="Next sign",
        alignment="right",
        width=70
    )

    # RECORDING
    window.add_button(
        "prev_recording",
        text="🡸 Rec",
        action=processor.prev_record,
        tooltip="Previous recording",
        shortcut="Left",
        alignment="right",
        width=70
    )

    window.add_button(
        "next_recording",
        text="Rec 🡺",
        action=processor.next_record,
        tooltip="Next recording",
        shortcut="Right",
        alignment="right",
        width=70
    )

    # FRAME CONTROL (fine control)
    window.add_button(
        "prev_frame",
        text="🡸 Frame",
        action=processor.prev_frame,
        tooltip="Previous frame (pause mode)",
        alignment="right",
        width=80
    )

    window.add_button(
        "next_frame",
        text="Frame 🡺",
        action=processor.next_frame,
        tooltip="Next frame (pause mode)",
        alignment="right",
        width=80
    )

    window.add_button(
        "delete_record",
        text="Delete",
        action=processor.delete_current_record,
        tooltip="Delete current recording",
        color=EmptyWindow.COLORS['red'],
        hover_color=EmptyWindow.COLORS['red'],
        alignment="right",
        width=70
    )

    window.show()
    return window

    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = run(app)
    sys.exit(app.exec())