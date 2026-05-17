#!/usr/bin/env python3

import os
import sys
import cv2
import time
import numpy as np
from collections import deque

from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QListWidget, QPushButton, QLabel
from PySide6.QtCore import QTimer

from core.window.webcam_window import WebcamWindow
from core.processors.recording_processor import RecordingProcessor
from core.utils import AppPaths

class StartSelectorDialog(QDialog):
    def __init__(self, labels, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Start From")
        self.setFixedSize(220, 300)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Selecciona letra inicial:"))

        self.list_widget = QListWidget()
        self.list_widget.addItems(labels)
        layout.addWidget(self.list_widget)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)

    def selected_letter(self):
        item = self.list_widget.currentItem()
        return item.text() if item else None


VIDEOS_PATH = AppPaths.path("data/resources/lse_videos")

STACK_VIDEOS = deque([
    (
        os.path.splitext(f)[0],
        os.path.join(VIDEOS_PATH, f)
    )
    for f in sorted(os.listdir(VIDEOS_PATH))
    if f.endswith(".mov")
    for _ in range(2)
])

MAP_VIDEO_TO_LETTER = {
    "n-fuerte": u"ñ"
}

MAP_LETTER_TO_VIDEO = {
    u"ñ": "n-fuerte"
}

if not STACK_VIDEOS:
    print("No videos found")
    sys.exit()

PROCESSOR = RecordingProcessor(max_sequences_per_record=5)

def run(app):
    window = WebcamWindow(
        width=1280,
        height=720,
        frame_processor=PROCESSOR
    )


    text_to_print = ""
    confirm_button = None
    video_cap = None
    label = ""
    recording_started = False

    right_left_tracker =  set()

    phase = "idle"   # idle, demo, recording

    # Manage Controls
    def position_confirm_button():
        nonlocal confirm_button

        if not confirm_button:
            return

        label = window.display_label

        w = label.width()
        h = label.height()

        video_w = int(w * 0.55)
        video_h = int(h * 0.55)

        x = (w - confirm_button.width()) // 2
        y = (h // 2 + video_h // 2) - 20

        confirm_button.move(int(x), int(y))
        confirm_button.raise_()

    def jump_stack_to_letter(target_letter):
        target_letter = target_letter.strip().lower()

        while STACK_VIDEOS:
            current_label, _ = STACK_VIDEOS[0]

            if current_label.lower() == target_letter:
                print(f"Jumping to letter: {current_label}")
                return True

            STACK_VIDEOS.popleft()

        return False

    def start():
        nonlocal recording_started

        if recording_started:
            return

        available_letters = []
        visited = set()

        for lbl, _ in STACK_VIDEOS:
            letter = MAP_VIDEO_TO_LETTER.get(lbl, lbl)
            if letter not in visited:
                available_letters.append(letter.upper())
                visited.add(letter)

        selector = StartSelectorDialog(available_letters, window)

        if selector.exec():
            selected = selector.selected_letter()
            if selected:
                selected = selected.lower()
                jump_stack_to_letter(MAP_LETTER_TO_VIDEO.get(selected, selected))

        recording_started = True
        load_next()

    
    def load_next():
        nonlocal video_cap, label, phase, text_to_print

        PROCESSOR.save_records()

        if not STACK_VIDEOS:
            print("Finished dataset")
            app.quit()
            return
        
        if confirm_button:
            confirm_button.hide()

        current_video = STACK_VIDEOS.popleft()
        label, video_path = current_video

        text_to_print = "Mano {}: {}"
        if label in right_left_tracker:
            text_to_print = text_to_print.format("derecha", label)
        else:
            right_left_tracker.add(label)
            text_to_print = text_to_print.format("izquierda", label)

        if video_cap:
            video_cap.release()

        video_cap = cv2.VideoCapture(video_path)
        PROCESSOR.current_label = label

        print(f"Label: {label}")

        QTimer.singleShot(500, start_demo)

    def start_demo():
        nonlocal phase, confirm_button
        phase = "demo"
        if confirm_button:
            confirm_button.show()
        #start_record()

    def start_record():
        nonlocal phase, confirm_button
        if confirm_button:
            confirm_button.hide()
        phase = "recording"

        PROCESSOR.start_record(label, countdown=True)
        check_recording()

    def check_recording():
        if PROCESSOR.is_recording() or PROCESSOR.is_countdown():
            QTimer.singleShot(100, check_recording)
        else:
            QTimer.singleShot(500, load_next)


    ############################################
    # Overdrive processor with overlay

    original_process = PROCESSOR.process

    def draw_label_bottom(frame, text):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thickness = 3

        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = h - 40

        cv2.putText(frame, text.upper(), (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
 
    def process_with_overlay(frame):
        nonlocal video_cap, phase, text_to_print

        frame = original_process(frame)

        h, w = frame.shape[:2]

        if phase == "demo":
            # soft blur background
            blurred = cv2.GaussianBlur(frame, (35, 35), 0)
            frame = cv2.addWeighted(frame, 0.35, blurred, 0.65, 0)

        # ---------------- VIDEO CENTER ----------------
        if video_cap and phase == "demo":
            ret, vid = video_cap.read()

            if not ret:
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, vid = video_cap.read()

            if ret:
                vid = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)

                vw = int(w * 0.55)
                vh = int(h * 0.55)

                vid = cv2.resize(vid, (vw, vh))

                x1 = (w - vw) // 2
                y1 = (h - vh) // 2 - 40

                frame[y1:y1+vh, x1:x1+vw] = vid

            if label:
                draw_label_bottom(frame, text_to_print)

        return frame

    PROCESSOR.process = process_with_overlay

    window.add_button(
        "start_button",
        text="Start",
        action=start,
        color=WebcamWindow.COLORS["blue"],
        hover_color=WebcamWindow.COLORS["blue_hover"],
        tooltip="Start guided recording",
        shortcut="S",
        alignment="left",
        width=80
    )

    confirm_button = window.add_button(
        "confirm_button",
        text="Grabar",
        action=start_record,
        color=WebcamWindow.COLORS["orange"],
        hover_color=WebcamWindow.COLORS["orange_hover"],
        tooltip="Start recording the current letter",
        shortcut="G",
        width=160,
        height=40
    )
    window.button_panel_layout.removeWidget(confirm_button)
    confirm_button.setParent(window)
    confirm_button.hide()
    window.external_resizes.append(position_confirm_button)

    window.show()
    return window


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = run(app)
    sys.exit(app.exec())