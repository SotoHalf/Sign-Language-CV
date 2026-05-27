#!/usr/bin/env python3
"""
collect_webcam.py — Free-form webcam data collection.

Manually-driven alternative to the guided script. The user sets a label,
starts/stops recording at will, and saves when done.

Keyboard shortcuts (also available as buttons):
    R — Toggle recording on/off (3-second countdown).

Usage:
    python collect_webcam.py
"""

import sys
from PySide6.QtWidgets import QApplication

from core.window.webcam_window import WebcamWindow
from core.processors.recording_processor import RecordingProcessor

PROCESSOR = RecordingProcessor()


def main() -> None:
    """Open the webcam window and wire up record/label/save buttons."""
    app = QApplication(sys.argv)
    window = WebcamWindow(width=1280, height=720, frame_processor=PROCESSOR)

    def toggle_record() -> None:
        """Start a new recording (with countdown) or abort the current one."""
        if not PROCESSOR.is_recording() and not PROCESSOR.is_countdown():
            # Ask for a label if none has been set yet
            if PROCESSOR.current_label == RecordingProcessor.DEFAULT_LABEL:
                change_label()
            PROCESSOR.start_record(PROCESSOR.current_label)
        else:
            PROCESSOR.stop_record()

    def change_label() -> None:
        """Prompt the user to enter a new label for the next recording."""
        new_label = window.show_input_dialog(
            "Change Label", "Enter new label:", PROCESSOR.current_label
        )
        if new_label:
            PROCESSOR.current_label = new_label
            print(f"Current label set to: {PROCESSOR.current_label}")

    def save_records() -> None:
        """Write all in-memory recordings to disk."""
        PROCESSOR.save_records()
        print("Records saved!")

    window.add_button(
        "record_button",
        text="Record",
        action=toggle_record,
        color=WebcamWindow.COLORS['red'],
        hover_color=WebcamWindow.COLORS['red_hover'],
        tooltip="Start / Stop recording (with 3s countdown)",
        shortcut="R",
        alignment="left",
        width=80
    )

    window.add_button(
        "change_label",
        text="Label",
        action=change_label,
        tooltip="Change current label for next recordings",
        alignment="right",
        width=80
    )

    window.add_button(
        "save_button",
        text="Save",
        action=save_records,
        tooltip="Save recorded sequences to disk",
        alignment="right",
        width=80
    )

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
