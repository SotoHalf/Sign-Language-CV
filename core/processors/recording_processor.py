import os
import cv2
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from core.processors.frame_processor import FrameProcessor
from core.hand_tracker import HandTracker
from core.landmark_handler import LandmarkHandler
from core.utils import generate_unique_id, AppPaths

AppPaths.load_env()


class SequenceRecord:
    """
    Container for a single recorded landmark sequence.

    Stores both the raw (unnormalized) and the preprocessed DataFrames,
    along with a unique identifier used for file naming.
    """

    def __init__(self, raw: pd.DataFrame, processed: pd.DataFrame) -> None:
        """
        :param raw: DataFrame with the original (x, y, z) landmark values.
        :type raw: pd.DataFrame
        :param processed: DataFrame after position/scale normalization and delta appending.
        :type processed: pd.DataFrame
        """
        self.uid: str = generate_unique_id()
        self.raw: pd.DataFrame = raw
        self.processed: pd.DataFrame = processed


class RecordingProcessor(FrameProcessor):
    """
    Frame processor that records hand landmark sequences for training data collection.

    Lifecycle per recording session:
        1. ``start_record(label)`` → 3-second countdown.
        2. Countdown finishes → landmark capture begins.
        3. Buffer fills (``n_frames``) → sequence saved, cooldown starts.
        4. After ``max_sequences_per_record`` captures, session ends automatically.

    All recorded sequences are kept in memory until :meth:`save_records` is called.
    """

    DEFAULT_LABEL: str = 'Unknown'

    def __init__(
        self,
        n_frames: Optional[int] = None,
        max_sequences_per_record: int = 10
    ) -> None:
        """
        :param n_frames: Landmark frames per sequence. If ``None``, reads from
            env (``CAPTURE_FRAME_RATE_FPS × CAPTURE_DURATION_SECONDS``).
        :type n_frames: int, optional
        :param max_sequences_per_record: Number of sequences to capture before
            the session stops automatically.
        :type max_sequences_per_record: int
        """
        super().__init__()
        if n_frames is None:
            n_frames = self.get_default_n_frames()

        self.tracker: HandTracker = HandTracker()
        self.landmark_handler: LandmarkHandler = LandmarkHandler(n_frames)
        self.n_frames: int = n_frames

        # --- Recording state machine ---
        self._recording_active: bool = False
        self._is_countdown: bool = False
        self._countdown_start_time: float = 0.0
        self._countdown_duration: float = 3.0
        self._flash_until: float = 0.0       # timestamp until the post-capture flash ends
        self._cooldown_until: float = 0.0    # timestamp until the next capture is allowed
        self._cooldown_duration: float = 1.5

        # --- Data storage ---
        self.records: Dict[str, List[SequenceRecord]] = {}
        self.max_sequences_per_record: int = max_sequences_per_record
        self._remaining_sequences: Dict[str, int] = {}
        self.current_label: str = RecordingProcessor.DEFAULT_LABEL

        # Allows flushing a partial buffer early (used when processing short video files)
        self._force_landmark_handler_export: bool = False

    # --------------------------------------------------
    # Control API
    # --------------------------------------------------

    def force_landmark_handler_export(self) -> None:
        """
        Request that the current (possibly incomplete) buffer be flushed on the
        next frame. Used when the source video ends before the buffer is full.
        """
        self._force_landmark_handler_export = True

    def get_count_records(self) -> Dict[str, int]:
        """
        Return the total number of recorded sequences grouped by label.

        :return: Mapping of ``{label: count}``.
        :rtype: dict[str, int]
        """
        return {label: len(seqs) for label, seqs in self.records.items()}

    def get_count_records_by_label(self, label: str) -> int:
        """
        Return the number of recorded sequences for a specific label.

        :param label: The gesture label to query.
        :type label: str
        :return: Number of sequences recorded under that label.
        :rtype: int
        """
        return len(self.records.get(label, []))

    def get_current_label(self) -> str:
        """
        Return the label assigned to the active recording session.

        :return: Current label string.
        :rtype: str
        """
        return self.current_label

    def _start_countdown(self, label: str) -> None:
        """
        Enter the countdown phase for the given label.

        Clears the landmark buffer so stale data from previous captures
        does not bleed into the new recording.

        :param label: Gesture label to assign to sequences captured in this session.
        :type label: str
        """
        if self._recording_active or self._is_countdown:
            print("Already recording or counting down")
            return
        print("Countdown started")
        self._is_countdown = True
        self._countdown_start_time = time.time()
        self.current_label = label
        self.landmark_handler.clear()

    def _begin_recording(self) -> None:
        """
        Transition from countdown to active capture.
        Called automatically when the countdown duration elapses.
        """
        self._is_countdown = False
        self._recording_active = True
        self._remaining_sequences[self.current_label] = self.max_sequences_per_record
        print(f"Recording started for label: {self.current_label}")

    def start_record(self, label: Optional[str] = None, countdown: bool = True) -> None:
        """
        Begin a recording session for the given label.

        :param label: Gesture label. Falls back to ``DEFAULT_LABEL`` if not provided.
        :type label: str, optional
        :param countdown: If ``True``, show a visual countdown before capture begins.
        :type countdown: bool
        """
        if not label:
            label = RecordingProcessor.DEFAULT_LABEL
        if countdown:
            self._start_countdown(label)
        else:
            self._begin_recording()

    def stop_record(self) -> None:
        """
        Abort any active countdown or recording and reset all session state.
        Already-captured sequences remain in :attr:`records`.
        """
        print("Recording stopped")
        self._recording_active = False
        self._is_countdown = False
        self.current_label = RecordingProcessor.DEFAULT_LABEL
        self.landmark_handler.clear()
        self._remaining_sequences = {}
        self._force_landmark_handler_export = False

    def is_recording(self) -> bool:
        """
        Return whether landmark capture is currently active.

        :return: ``True`` if actively recording frames.
        :rtype: bool
        """
        return self._recording_active

    def is_countdown(self) -> bool:
        """
        Return whether the pre-recording countdown is in progress.

        :return: ``True`` during the countdown phase.
        :rtype: bool
        """
        return self._is_countdown

    def _is_cooldown(self) -> bool:
        """Return ``True`` if we are within the post-capture cooldown window."""
        return time.time() < self._cooldown_until

    # --------------------------------------------------
    # Main process (FrameProcessor override)
    # --------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Per-frame pipeline:

        1. Detect and draw hands.
        2. Overlay visual feedback (countdown, REC indicator, flash, cooldown).
        3. If recording and not in cooldown: extract landmarks, buffer them.
        4. When the buffer is full (or a flush was forced): save the sequence,
           trigger flash and cooldown, decrement remaining count.

        :param frame: RGB input frame of shape ``(H, W, 3)``.
        :type frame: np.ndarray
        :return: Annotated frame.
        :rtype: np.ndarray
        """
        frame = self.tracker.find_hands(frame, draw=True)
        frame = self._draw_visual_feedback(frame)

        if self._recording_active and not self._is_cooldown():
            landmarks_raw = self.tracker.export_landmarks(frame, hand_id=0, draw=True)
            if landmarks_raw is not None and len(landmarks_raw) > 0:
                self.landmark_handler.add_frame(np.array(landmarks_raw, dtype=np.float32))

            if self.landmark_handler.ready() or self._force_landmark_handler_export:
                raw = self.landmark_handler.export()
                df_raw = LandmarkHandler.to_dataframe(raw)
                processed = LandmarkHandler.preprocess_landmarks(raw, self.n_frames)
                df_processed = LandmarkHandler.to_dataframe(processed)

                self.records.setdefault(self.current_label, []).append(
                    SequenceRecord(raw=df_raw, processed=df_processed)
                )
                print(f"Sequence recorded. Total for '{self.current_label}': "
                      f"{len(self.records[self.current_label])}")

                self._flash_until = time.time() + 0.15
                self.landmark_handler.clear()
                self._cooldown_until = time.time() + self._cooldown_duration
                self._force_landmark_handler_export = False

                remaining = self._remaining_sequences.get(self.current_label, 0) - 1
                if remaining <= 0:
                    self.stop_record()
                else:
                    self._remaining_sequences[self.current_label] = remaining

        elif self._is_countdown:
            if time.time() - self._countdown_start_time >= self._countdown_duration:
                self._begin_recording()

        return frame

    def _draw_visual_feedback(self, frame: np.ndarray) -> np.ndarray:
        """
        Render all recording-state overlays onto the frame:

        - White flash immediately after a sequence is captured.
        - Countdown number centered on screen.
        - Blinking REC dot and sequence counter while recording.
        - Cooldown timer during the pause between captures.

        :param frame: RGB frame to annotate in-place.
        :type frame: np.ndarray
        :return: Annotated frame.
        :rtype: np.ndarray
        """
        h, w = frame.shape[:2]

        # Flash effect
        if time.time() < self._flash_until:
            frame = cv2.addWeighted(frame, 0.25, np.full(frame.shape, 255, dtype=np.uint8), 0.5, 0)

        if self._is_countdown:
            elapsed = time.time() - self._countdown_start_time
            remaining = max(0, int(np.ceil(self._countdown_duration - elapsed)))
            if remaining > 0:
                font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 4, 8
                (tw, th), _ = cv2.getTextSize(str(remaining), font, scale, thickness)
                cx, cy = w // 2, h // 2
                cv2.circle(frame, (cx, cy), max(tw, th) // 2 + 50, (0, 0, 0), -1)
                cv2.putText(frame, str(remaining),
                            (cx - tw // 2, cy + th // 2),
                            font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Recording indicator (blinking)
        if self._recording_active:
            if int(time.time() * 2) % 2 == 0:   # blink at 2 Hz
                cv2.circle(frame, (40, 70), 20, (255, 0, 0), -1)
                cv2.circle(frame, (40, 70), 20, (255, 255, 255), 2)
                cv2.putText(frame, "REC", (70, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"Records: {self.get_count_records_by_label(self.current_label)}",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        if self._is_cooldown():
            remaining_cd = self._cooldown_until - time.time()
            cv2.putText(frame, f"RESETTING... {remaining_cd:.1f}s",
                        (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        return frame

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    def save_records(
        self,
        output_path_raw: Optional[str] = None,
        output_path_processed: Optional[str] = None
    ) -> None:
        """
        Write all in-memory sequences to disk as CSV files.

        Files are written under ``<output_dir>/<label>/<uid>.csv``.
        Existing files are never overwritten (the UID collision probability
        is negligible given the generation strategy).

        :param output_path_raw: Root folder for raw CSVs. Defaults to the
            ``DATA_RAW_PATH`` environment variable.
        :type output_path_raw: str, optional
        :param output_path_processed: Root folder for processed CSVs. Defaults to
            the ``DATA_PATH`` environment variable.
        :type output_path_processed: str, optional
        """
        raw_dir = output_path_raw or AppPaths.path(os.getenv("DATA_RAW_PATH", "data/raw/"))
        processed_dir = output_path_processed or AppPaths.path(os.getenv("DATA_PATH", "data/processed/"))

        if not self.records:
            print("No records to save")
            return

        for label, sequences in self.records.items():
            raw_label_dir = os.path.join(raw_dir, label)
            processed_label_dir = os.path.join(processed_dir, label)
            os.makedirs(raw_label_dir, exist_ok=True)
            os.makedirs(processed_label_dir, exist_ok=True)

            for seq in sequences:
                raw_path = os.path.join(raw_label_dir, f"{seq.uid}.csv")
                if not os.path.exists(raw_path):
                    seq.raw.to_csv(raw_path, index=False)

                processed_path = os.path.join(processed_label_dir, f"{seq.uid}.csv")
                if not os.path.exists(processed_path):
                    seq.processed.to_csv(processed_path, index=False)

            print(f"Saved {len(sequences)} sequences for label '{label}'")


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    from core.window.webcam_window import WebcamWindow

    app = QApplication(sys.argv)
    processor = RecordingProcessor()
    window = WebcamWindow(width=1280, height=720, frame_processor=processor)

    def toggle_record():
        if not processor.is_recording() and not processor.is_countdown():
            if processor.current_label == RecordingProcessor.DEFAULT_LABEL:
                change_label()
            processor.start_record(processor.current_label)
        else:
            processor.stop_record()

    def change_label():
        new_label = window.show_input_dialog("Change Label", "Enter new label:", processor.current_label)
        if new_label:
            processor.current_label = new_label
            print(f"Current label set to: {processor.current_label}")

    def save_records():
        processor.save_records()
        print("Records saved!")

    window.add_button("record_button", text="Record", action=toggle_record,
                      checkable=True, color=WebcamWindow.COLORS['red'],
                      hover_color=WebcamWindow.COLORS['red_hover'],
                      tooltip="Start / Stop recording (with 3s countdown)",
                      shortcut="R", alignment="left", width=80)

    window.add_button("change_label", text="Label", action=change_label,
                      tooltip="Change current label for next recordings",
                      alignment="right", width=80)

    window.add_button("save_button", text="Save", action=save_records,
                      tooltip="Save recorded sequences to disk",
                      alignment="right", width=80)

    window.show()
    sys.exit(app.exec())
