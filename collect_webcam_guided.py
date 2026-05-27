#!/usr/bin/env python3
"""
collect_webcam_guided.py — Guided data collection with reference video overlay.

Each sign label is associated with a reference ``.mov`` video that plays on
screen as a visual cue while the user records the gesture. The dataset is
consumed in order from a queue (``STACK_VIDEOS``), recording each label twice
(left hand then right hand).

Phase state machine per sign:
    idle  →  demo  →  recording  →  (next sign)

Usage:
    python collect_webcam_guided.py
    OR imported and called as collect_webcam_guided.run(app)
"""

import os
import sys
import cv2
from collections import deque
from typing import Deque, Dict, Optional, Tuple

from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QListWidget, QPushButton, QLabel
from PySide6.QtCore import QTimer

from core.window.webcam_window import WebcamWindow
from core.processors.recording_processor import RecordingProcessor
from core.utils import AppPaths

# Path to the folder containing the reference .mov files
VIDEOS_PATH: str = AppPaths.path("data/resources/lse_videos")

# Each video appears twice: first capture = left hand, second = right hand
STACK_VIDEOS: Deque[Tuple[str, str]] = deque([
    (os.path.splitext(f)[0], os.path.join(VIDEOS_PATH, f))
    for f in sorted(os.listdir(VIDEOS_PATH))
    if f.endswith(".mov")
    for _ in range(2)
])

# Maps video filename stem → display character (for non-ASCII labels like ñ)
MAP_VIDEO_TO_LETTER: Dict[str, str] = {"n-fuerte": "ñ"}
MAP_LETTER_TO_VIDEO: Dict[str, str] = {"ñ": "n-fuerte"}

if not STACK_VIDEOS:
    print("No videos found")
    sys.exit()

# Shared processor instance (one per run call)
PROCESSOR = RecordingProcessor(max_sequences_per_record=5)


class StartSelectorDialog(QDialog):
    """
    Modal dialog that lets the user pick the starting letter before recording begins.

    Shown only once; after confirmation the queue is fast-forwarded to that letter.
    """

    def __init__(self, labels: list, parent=None) -> None:
        """
        :param labels: Alphabetically sorted list of available letter strings to display.
        :type labels: list[str]
        :param parent: Optional parent widget.
        :type parent: QWidget, optional
        """
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

    def selected_letter(self) -> Optional[str]:
        """
        Return the currently selected letter.

        :return: Selected label string, or ``None`` if nothing is selected.
        :rtype: str, optional
        """
        item = self.list_widget.currentItem()
        return item.text() if item else None


def run(app: QApplication) -> WebcamWindow:
    """
    Build the guided recording window and wire up all controls.

    :param app: The running Qt application instance (shared with the caller).
    :type app: QApplication
    :return: The webcam window (caller should hold a reference to prevent GC).
    :rtype: WebcamWindow
    """
    window = WebcamWindow(width=1280, height=720, frame_processor=PROCESSOR)

    # --- Session state ---
    text_to_print: str = ""
    confirm_button: Optional[object] = None
    video_cap: Optional[cv2.VideoCapture] = None
    label: str = ""
    recording_started: bool = False
    right_left_tracker: set = set()   # labels seen once (left hand) vs twice (right hand)
    phase: str = "idle"               # "idle" | "demo" | "recording"

    # --------------------------------------------------
    # Layout helper
    # --------------------------------------------------

    def position_confirm_button() -> None:
        """
        Place the confirm button centered horizontally just below the video overlay.
        Called from ``external_resizes`` on every window resize.
        """
        nonlocal confirm_button
        if not confirm_button:
            return
        lbl = window.display_label
        w, h = lbl.width(), lbl.height()
        video_h = int(h * 0.55)
        x = (w - confirm_button.width()) // 2
        y = (h // 2 + video_h // 2) - 20
        confirm_button.move(int(x), int(y))
        confirm_button.raise_()

    # --------------------------------------------------
    # Queue navigation
    # --------------------------------------------------

    def jump_stack_to_letter(target_letter: str) -> bool:
        """
        Fast-forward ``STACK_VIDEOS`` until the queue front matches ``target_letter``.

        :param target_letter: Lowercase label to seek to.
        :type target_letter: str
        :return: ``True`` if the letter was found and the queue is positioned at it.
        :rtype: bool
        """
        target_letter = target_letter.strip().lower()
        while STACK_VIDEOS:
            current_label, _ = STACK_VIDEOS[0]
            if current_label.lower() == target_letter:
                print(f"Jumping to letter: {current_label}")
                return True
            STACK_VIDEOS.popleft()
        return False

    # --------------------------------------------------
    # Session flow
    # --------------------------------------------------

    def start() -> None:
        """
        Show the start-selector dialog (once), optionally seek to a letter,
        then begin the recording loop.
        """
        nonlocal recording_started
        if recording_started:
            return

        # Build unique letter list in queue order
        available_letters = []
        visited: set = set()
        for lbl, _ in STACK_VIDEOS:
            letter = MAP_VIDEO_TO_LETTER.get(lbl, lbl)
            if letter not in visited:
                available_letters.append(letter.upper())
                visited.add(letter)

        selector = StartSelectorDialog(available_letters, window)
        if selector.exec():
            selected = selector.selected_letter()
            if selected:
                jump_stack_to_letter(MAP_LETTER_TO_VIDEO.get(selected.lower(), selected.lower()))

        recording_started = True
        load_next()

    def load_next() -> None:
        """
        Save pending records, pop the next video from the queue and start the demo phase.
        Quits the app when the queue is exhausted.
        """
        nonlocal video_cap, label, phase, text_to_print

        PROCESSOR.save_records()

        if not STACK_VIDEOS:
            print("Finished dataset")
            app.quit()
            return

        if confirm_button:
            confirm_button.hide()

        label, video_path = STACK_VIDEOS.popleft()

        # Determine hand label: first occurrence = left, second = right
        if label in right_left_tracker:
            text_to_print = f"Mano derecha: {label}"
        else:
            right_left_tracker.add(label)
            text_to_print = f"Mano izquierda: {label}"

        if video_cap:
            video_cap.release()

        video_cap = cv2.VideoCapture(video_path)
        PROCESSOR.current_label = label
        print(f"Label: {label}")

        QTimer.singleShot(500, start_demo)

    def start_demo() -> None:
        """Enter the demo phase: play the reference video overlay and show the confirm button."""
        nonlocal phase, confirm_button
        phase = "demo"
        if confirm_button:
            confirm_button.show()

    def start_record() -> None:
        """
        Hide the confirm button, switch to recording phase and start the countdown.
        Polls until the recording session ends, then loads the next sign.
        """
        nonlocal phase, confirm_button
        if confirm_button:
            confirm_button.hide()
        phase = "recording"
        PROCESSOR.start_record(label, countdown=True)
        check_recording()

    def check_recording() -> None:
        """
        Poll every 100 ms until the recording session ends, then schedule the next sign.
        Uses QTimer to avoid blocking the event loop.
        """
        if PROCESSOR.is_recording() or PROCESSOR.is_countdown():
            QTimer.singleShot(100, check_recording)
        else:
            QTimer.singleShot(500, load_next)

    # --------------------------------------------------
    # Frame overlay (monkey-patch the processor's process method)
    # --------------------------------------------------

    original_process = PROCESSOR.process

    def draw_label_bottom(frame, text: str) -> None:
        """
        Draw a centred, uppercase label near the bottom of the frame.

        :param frame: RGB NumPy frame to annotate in-place.
        :param text: Text to render.
        :type text: str
        """
        h, w = frame.shape[:2]
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(frame, text.upper(),
                    ((w - tw) // 2, h - 40),
                    font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def process_with_overlay(frame) -> object:
        """
        Wrap the processor's ``process`` method to add the reference video overlay.

        During the demo phase the webcam feed is blurred to keep attention on the
        video inset, and the reference clip loops continuously.

        :param frame: Raw webcam RGB frame.
        :return: Annotated frame.
        """
        nonlocal video_cap, phase, text_to_print
        frame = original_process(frame)
        h, w = frame.shape[:2]

        if phase == "demo":
            # Blur the background so the video inset stands out
            blurred = cv2.GaussianBlur(frame, (35, 35), 0)
            frame = cv2.addWeighted(frame, 0.35, blurred, 0.65, 0)

            if video_cap:
                ret, vid = video_cap.read()
                if not ret:
                    # Loop the reference clip
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, vid = video_cap.read()

                if ret:
                    vw, vh = int(w * 0.55), int(h * 0.55)
                    vid = cv2.resize(cv2.cvtColor(vid, cv2.COLOR_BGR2RGB), (vw, vh))
                    x1 = (w - vw) // 2
                    y1 = (h - vh) // 2 - 40
                    frame[y1:y1 + vh, x1:x1 + vw] = vid

            if label:
                draw_label_bottom(frame, text_to_print)

        return frame

    PROCESSOR.process = process_with_overlay

    # --------------------------------------------------
    # Buttons
    # --------------------------------------------------

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

    # Detach from the panel layout so it can be positioned freely over the video
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
