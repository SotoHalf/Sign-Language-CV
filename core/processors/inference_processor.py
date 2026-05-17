import numpy as np
import cv2
from typing import Optional, Tuple

from core.processors.frame_processor import FrameProcessor
from core.hand_tracker import HandTracker
from core.landmark_handler import LandmarkHandler
from core.model_handler import ModelHandler


class InferenceProcessor(FrameProcessor):
    """
    Processor that loads a trained model and performs real-time inference
    on hand gesture sequences.
    """

    def __init__(self, model_path: str, encoder_path: str, n_frames: int = None):
        """
        Initialize the inference processor.

        :param model_path: Path to the trained model file.
        :param encoder_path: Path to the label encoder classes file.
        :param n_frames: Number of frames to buffer for a sequence.
                         If None, uses the model's expected input shape.
        """
        super().__init__()
        self.model_handler = ModelHandler()
        self.model_handler.load(model_path, encoder_path)

        # Determine sequence length from model's input shape
        expected_seq_len = self.model_handler.input_shape[0]
        if n_frames is None:
            n_frames = expected_seq_len
        else:
            if n_frames != expected_seq_len:
                print(f"Warning: n_frames ({n_frames}) does not match model expected {expected_seq_len}. Using {expected_seq_len}")
                n_frames = expected_seq_len

        self.landmark_handler = LandmarkHandler(n_frames)
        self.tracker = HandTracker()

        self.inference_enabled: bool = True
        self.last_prediction: str = "None"
        self.last_confidence: float = 0.0

    def toggle_inference(self) -> None:
        """Toggle inference on/off."""
        self.inference_enabled = not self.inference_enabled
        print(f"Inference enabled: {self.inference_enabled}")
        if not self.inference_enabled:
            self.landmark_handler.clear()
            self.last_prediction = "None"
            self.last_confidence = 0.0

    def get_last_prediction(self) -> Tuple[str, float]:
        """Return the latest prediction and confidence."""
        return self.last_prediction, self.last_confidence

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame: detect hands, buffer landmarks, run inference when buffer is full,
        and overlay prediction text on the frame.

        :param frame: RGB image as numpy array.
        :return: Frame with optional overlay.
        """
        # Detect and draw hands
        frame = self.tracker.findHands(frame, draw=True)

        if not self.inference_enabled:
            # Draw current prediction status
            self._draw_prediction(frame)
            return frame

        # Extract landmarks from first detected hand
        landmarks_raw = self.tracker.exportLandmarks(frame, hand_id=0, draw=False)

        if landmarks_raw is not None and len(landmarks_raw) > 0:
            # Convert to numpy array (21, 3)
            landmarks_np = np.array(landmarks_raw, dtype=np.float32)
            self.landmark_handler.add_frame(landmarks_np)

            # If buffer is full, run inference
            if self.landmark_handler.ready():
                raw = self.landmark_handler.export()
                processed = LandmarkHandler.preprocess_landmarks(raw)

                # Check shape consistency
                expected_shape = (self.landmark_handler.buffer.maxlen, self.model_handler.input_shape[1])
                if processed.shape != expected_shape:
                    print(f"Warning: Processed shape {processed.shape} does not match model input {expected_shape}")
                else:
                    label, confidence = self.model_handler.predict(processed)
                    self.last_prediction = label
                    self.last_confidence = confidence
                    print(f"Prediction: {label} ({confidence:.2f})")

                # Clear buffer for next sequence
                self.landmark_handler.clear()
        else:
            # No hand detected: clear buffer and reset prediction
            self.landmark_handler.clear()
            if self.last_prediction != "No hand":
                self.last_prediction = "No hand"
                self.last_confidence = 0.0

        # Overlay prediction text on frame
        self._draw_prediction(frame)

        return frame

    def _draw_prediction(self, frame: np.ndarray) -> None:
        """Draw the current prediction on the frame."""
        if not self.inference_enabled:
            text = "Inference OFF"
            color = (0, 0, 255)  # Red
        else:
            text = f"{self.last_prediction} ({self.last_confidence:.2f})"
            color = (0, 255, 0)  # Green

        # Put text at top-left corner
        cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2, cv2.LINE_4)
        


if __name__ == "__main__":
    import sys
    import os
    from core.processors.inference_processor import InferenceProcessor
    from core.window.webcam_window import WebcamWindow
    from PySide6.QtWidgets import QApplication
    from core.utils import AppPaths

    AppPaths.load_env()

    # Configuration
    MODEL_PATH   = AppPaths.path(os.getenv("SIGN_TRANSLATE_MODEL", "models/sign_lstm.keras"))
    ENCODER_OUTPUT = AppPaths.path(os.getenv("LABEL_ENCODER_FILE", "models/sign_lstm.npy"))

    # Load data
    app = QApplication(sys.argv)

    # Create inference processor
    processor = InferenceProcessor(MODEL_PATH, ENCODER_OUTPUT)

    # Create window with processor
    window = WebcamWindow(
        width=1280,
        height=720,
        frame_processor=processor
    )

    # Add a button to toggle inference
    def toggle():
        processor.toggle_inference()
        print(f"Inference toggled: {processor.inference_enabled}")

    window.add_button(
        "toggle_inference", 
        text="I", 
        tooltip="Start / Stop inference prediction",
        action=toggle,
        color=WebcamWindow.COLORS["orange"],
        hover_color=WebcamWindow.COLORS["orange_hover"],
        shortcut="I",
        width=80
    )

    window.show()
    sys.exit(app.exec())