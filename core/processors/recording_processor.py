import os
import cv2
import time
import numpy as np
import pandas as pd
from typing import Dict, List

from core.processors.frame_processor import FrameProcessor
from core.hand_tracker import HandTracker
from core.landmark_handler import LandmarkHandler
from core.utils import generate_unique_id, AppPaths

AppPaths.load_env()

class SequenceRecord:
    def __init__(self, raw: pd.DataFrame, processed: pd.DataFrame):
        self.uid =  generate_unique_id()
        self.raw = raw
        self.processed = processed

class RecordingProcessor(FrameProcessor):
    """
    Processor to record sequences of hand landmarks and store them as CSV-ready data.
    """

    DEFAULT_LABEL = 'Unknown'

    def __init__(self, n_frames: int = None, max_sequences_per_record: int = 10):
        if n_frames is None:
            n_frames = self.get_default_n_frames()

        self.tracker = HandTracker()
        self.landmark_handler = LandmarkHandler(n_frames)
        self.n_frames = n_frames

        # Recording state control
        self._recording_active = False      # Actually recording landmarks
        self._is_countdown = False          # Countdown in progress
        self._countdown_start_time = 0.0
        self._countdown_duration = 3.0      # seconds
        self._flash_until = 0.0             # timestamp until flash effect
        # time for the next record
        self._cooldown_until = 0.0
        self._cooldown_duration = 1.5  # seconds

        # Records manage
        self.records: Dict[str, List[SequenceRecord]] = {}
        """
        label: [
            {
                "raw": [],
                "processed": []
            }
        ]
        """
        self.max_sequences_per_record = max_sequences_per_record
        self.current_sequences_per_record: Dict[str, int] = {}
        self.current_label: str = RecordingProcessor.DEFAULT_LABEL

        # test - for less frames
        self._force_landmark_handler_export = False

    
    # CONTROL METHODS

    def force_landmark_handler_export(self):
        self._force_landmark_handler_export = True

    def get_count_records(self) -> Dict[str, int]:
        return {label: len(seqs) for label, seqs in self.records.items()}
    
    def get_count_records_by_label(self, label) -> Dict[str, int]:
        return len(self.records.get(label,[]))
    
    def get_current_label(self) -> str:
        return self.current_label
    
    def _start_countdown(self, label: str = None):
        if not label:
            label = RecordingProcessor.DEFAULT_LABEL
        """Begin countdown before actual recording starts."""
        if self._recording_active or self._is_countdown:
            print("Already recording or counting down")
            return
        print("Countdown started")
        self._is_countdown = True
        self._countdown_start_time = time.time()
        self.current_label = label
        # Clear buffer for old data
        self.landmark_handler.clear()

    def _begin_recording(self):
        """Called when countdown finishes. Starts actual landmark recording."""
        self._is_countdown = False
        self._recording_active = True
        # start count records
        self.current_sequences_per_record[self.current_label] = self.max_sequences_per_record
        print(f"Recording started for label: {self.current_label}")

    def start_record(self, label: str = None, countdown: bool = True):
        if not label:
            label = RecordingProcessor.DEFAULT_LABEL
        if countdown:
            self._start_countdown(label)
        else:
            self._begin_recording()


    def stop_record(self):
        """
        Stop recording.
        """
        print("Recording stopped")

        self._recording_active = False
        self._is_countdown = False

        # reset label
        self.current_label = RecordingProcessor.DEFAULT_LABEL

        # Clear buffer
        self.landmark_handler.clear()
        self.current_sequences_per_record = {}
        self._force_landmark_handler_export = False

    def is_recording(self) -> bool:
        return self._recording_active
    
    def is_countdown(self) -> bool:
        return self._is_countdown
    
    def _is_cooldown(self):
        return time.time() < self._cooldown_until
    
    # MAIN PROCESS (override)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame:
        - Draw visual feedback (countdown, REC indicator, flash)
        - Detect hands
        - Extract landmarks
        - If recording, accumulate frames
        - When buffer is full, store sequence
        """

        # Detect and draw hands
        frame = self.tracker.findHands(frame, draw=True)

        # Add visual feedback (countdown, REC, flash, label ...)
        frame = self._draw_visual_feedback(frame)

        # if the user is recording
        if self._recording_active and not self._is_cooldown():
            # Get landmarks
            landmarks_raw = self.tracker.exportLandmarks(frame, hand_id=0, draw=True)
            if landmarks_raw is not None and len(landmarks_raw) > 0:
                landmarks_np = np.array(landmarks_raw, dtype=np.float32)
                self.landmark_handler.add_frame(landmarks_np)


            if self.landmark_handler.ready() or self._force_landmark_handler_export:
                # Save sequence
                raw = self.landmark_handler.export()
                df_raw = LandmarkHandler.to_dataframe(raw)
                processed = LandmarkHandler.preprocess_landmarks(raw, self.n_frames)
                df_processed = LandmarkHandler.to_dataframe(processed)
                
                self.records.setdefault(self.current_label, []).append(
                    SequenceRecord(raw=df_raw, processed=df_processed)
                )
                print(f"Sequence recorded. Total for label {self.current_label}: {len(self.records[self.current_label])}")
                
                # Trigger flash effect
                self._flash_until = time.time() + 0.15  # 150ms flash
                
                self.landmark_handler.clear()

                self._cooldown_until = time.time() + self._cooldown_duration
                
                # Decrement remaining sequences for this label
                remaining = self.current_sequences_per_record.get(self.current_label, 0) - 1
                if remaining <= 0:
                    self.stop_record()
                else:
                    self.current_sequences_per_record[self.current_label] = remaining
        
        elif self._is_countdown:
            # Check if countdown finished
            elapsed = time.time() - self._countdown_start_time
            if elapsed >= self._countdown_duration:
                self._begin_recording()
        
        return frame
    

    def _draw_visual_feedback(self, frame: np.ndarray) -> np.ndarray:
        """Draw countdown number, REC indicator, and flash overlay."""
        h, w = frame.shape[:2]
        
        # Flash effect
        if time.time() < self._flash_until:
            white_overlay = np.full(frame.shape, 255, dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.25, white_overlay, 0.5, 0)
        
        # Countdown
        if self._is_countdown:
            elapsed = time.time() - self._countdown_start_time
            remaining = max(0, int(np.ceil(self._countdown_duration - elapsed)))
            if remaining > 0:
                text = str(remaining)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 4
                thickness = 8
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                center_x = w // 2
                center_y = h // 2

                # draw circle
                radius = max(text_w, text_h) // 2 + 50
                cv2.circle(frame, (center_x, center_y), radius, (0, 0, 0), -1)
                
                text_x = center_x - text_w // 2
                text_y = center_y + text_h // 2
            
                cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                            (255, 255, 255), thickness, cv2.LINE_AA)

        
        # Recording indicator (blinking)
        if self._recording_active:
            # blink every 0.5 seconds
            if int(time.time() * 2) % 2 == 0:
                # Red circle top-left
                cv2.circle(frame, (40, 70), 20, (255, 0, 0), -1)
                cv2.circle(frame, (40, 70), 20, (255, 255, 255), 2)
                # Text "REC"
                cv2.putText(frame, "REC", (70, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)
                
            cv2.putText(frame, 
                        f"Records: {self.get_count_records_by_label(self.get_current_label())}", 
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                        )

        # Cooldown        
        if time.time() < self._cooldown_until:
            remaining = self._cooldown_until - time.time()
            cv2.putText(frame, f"RESETTING... {remaining:.1f}s",
                        (w//2 - 120, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2)
        
        return frame

    # SAVE METHODS
    
    def save_records(self, output_path_raw: str = None, output_path_processed: str = None):
        """
        Save recorded sequences into raw and processed folders organized by label.
        Each sequence gets a unique ID used in both raw and processed CSVs.
        """
    
        data_raw_path = AppPaths.path(os.getenv("DATA_RAW_PATH", "data/raw/"))
        data_processed_path = AppPaths.path(os.getenv("DATA_PATH", "data/processed/"))

        raw_dir = output_path_raw or AppPaths.path(data_raw_path)
        processed_dir = output_path_processed or AppPaths.path(data_processed_path)

        if not self.records:
            print("No records to save")
            return

        # Go over labels and their sequences
        for label, sequences in self.records.items():
            raw_label_dir = os.path.join(raw_dir, label)
            processed_label_dir = os.path.join(processed_dir, label)

            os.makedirs(raw_label_dir, exist_ok=True)
            os.makedirs(processed_label_dir, exist_ok=True)

            for seq in sequences:
                uid = seq.uid # One UID per sequence

                # Save RAW
                raw_path = os.path.join(raw_label_dir, f"{uid}.csv")
                if not os.path.exists(raw_path):
                    seq.raw.to_csv(raw_path, index=False)

                # Save PROCESSED
                processed_path = os.path.join(processed_label_dir, f"{uid}.csv")
                if not os.path.exists(processed_path):
                    seq.processed.to_csv(processed_path, index=False)

            print(f"Saved {len(sequences)} sequences for label '{label}'")



if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from core.window.webcam_window import WebcamWindow
    app = QApplication(sys.argv)
    
    processor = RecordingProcessor()
    window = WebcamWindow(0, width=1280, height=720, frame_processor=processor)
        
    def toggle_record():
        # Start countdown only if not already recording or counting down
        if not processor.is_recording() and not processor.is_countdown():
            #ask for label if none is given
            if processor.current_label == RecordingProcessor.DEFAULT_LABEL:
                change_label()

            label = processor.current_label
            processor.start_record(label)
        else:
            # Cancel any ongoing countdown or recording
            processor.stop_record()
    
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
        checkable=True,
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