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

PROCESSOR = RecordingProcessor()

def main():
    app = QApplication(sys.argv)
    
    window = WebcamWindow(0, width=1280, height=720, frame_processor=PROCESSOR)
        
    def toggle_record():
        # Start countdown only if not already recording or counting down
        if not PROCESSOR.is_recording() and not PROCESSOR.is_countdown():
            #ask for label if none is given
            if PROCESSOR.current_label == RecordingProcessor.DEFAULT_LABEL:
                change_label()

            label = PROCESSOR.current_label
            PROCESSOR.start_record(label)
        else:
            # Cancel any ongoing countdown or recording
            PROCESSOR.stop_record()
    
    def change_label():
        new_label = window.show_input_dialog("Change Label", "Enter new label:", PROCESSOR.current_label)
        if new_label:
            PROCESSOR.current_label = new_label
            print(f"Current label set to: {PROCESSOR.current_label}")
    
    def save_records():
        PROCESSOR.save_records()
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