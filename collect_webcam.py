#!/usr/bin/env python3
"""
Collect hand landmark data from webcam with visual feedback.
Usage: python collect_webcam.py
"""

import sys
import threading
import time
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from core.window.webcam_window import WebcamWindow
from core.processors.recording_processor import RecordingProcessor

def main():
    app = QApplication(sys.argv)
    
    processor = RecordingProcessor()
    window = WebcamWindow(0, width=1280, height=720, frame_processor=processor)
    
    # Flag to prevent multiple countdowns
    countdown_active = False
    
    def toggle_record():
        nonlocal countdown_active
        # Start countdown only if not already recording or counting down
        if not processor.is_recording() and not countdown_active:
            #ask for label if none is given
            if processor.current_label == RecordingProcessor.DEFAULT_LABEL:
                change_label()

            countdown_active = True
            label = processor.current_label
            processor.start_record(label)
            # Start a timer to reset the flag after countdown finishes
            # The flag will be reset when recording actually starts or is cancelled
            def reset_countdown_flag():
                nonlocal countdown_active
                countdown_active = False
            # Check after 3.1 seconds if still active (if recording didn't start)
            QTimer.singleShot(3100, reset_countdown_flag)
        else:
            # Cancel any ongoing countdown or recording
            processor.stop_record()
            countdown_active = False
    
    def change_label():
        new_label = window.show_input_dialog("Change Label", "Enter new label:", processor.current_label)
        if new_label:
            processor.current_label = new_label
            print(f"Current label set to: {processor.current_label}")
    
    def save_records():
        processor.save_records()
        print("Records saved!")
    
    # Add buttons
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