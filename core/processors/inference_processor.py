import numpy as np
import cv2
import time
from typing import Dict, List, Optional, Set, Tuple

from core.processors.frame_processor import FrameProcessor
from core.hand_tracker import HandTracker
from core.landmark_handler import LandmarkHandler
from core.model_handler import ModelHandler
from PIL import Image, ImageDraw, ImageFont


class InferenceProcessor(FrameProcessor):
    """
    Frame processor that performs real-time gesture recognition.

    Each frame is passed through hand detection and landmark extraction.
    Once the buffer holds a full sequence, the model predicts a gesture label.

    Predicted labels accumulate into a text buffer through a two-step
    confirmation mechanism:

    * A sign must be predicted consistently before it becomes *pending*.
    * The special ``"confirm"`` gesture commits the pending sign to the text buffer.
    * The ``"delete"`` gesture pops the last character (or clears all if repeated ≥ 3 times).
    """

    def __init__(
        self,
        model_path: str,
        encoder_path: str,
        n_frames: Optional[int] = None
    ) -> None:
        """
        Load the model and initialize the inference pipeline.

        :param model_path: Path to the trained Keras model file (``.keras`` or ``.h5``).
        :type model_path: str
        :param encoder_path: Path to the label encoder classes file (``.npy``).
        :type encoder_path: str
        :param n_frames: Buffer length (frames per sequence). If ``None``, the value
            is inferred from the model's input shape.
        :type n_frames: int, optional
        """
        super().__init__()
        self.model_handler: ModelHandler = ModelHandler()
        self.model_handler.load(model_path, encoder_path)

        # Use the model's own expected sequence length unless overridden
        expected_seq_len: int = self.model_handler.input_shape[0]
        if n_frames is None:
            n_frames = expected_seq_len
        elif n_frames != expected_seq_len:
            print(f"Warning: n_frames ({n_frames}) differs from model expected "
                  f"({expected_seq_len}). Using {expected_seq_len}.")
            n_frames = expected_seq_len

        self.landmark_handler: LandmarkHandler = LandmarkHandler(n_frames)
        self.tracker: HandTracker = HandTracker()

        self.inference_enabled: bool = True
        self.last_prediction: str = "None"
        self.last_confidence: float = 0.0

        # Accumulated word as a list of committed sign strings
        self.text_buffer: List[str] = []

        # Pending sign: must be confirmed explicitly before appending to text_buffer
        self.pending_sign: Optional[str] = None
        self.pending_count: int = 0

        # Special gesture labels with reserved behavior
        self.confirm_label: str = "confirm"
        self.delete_label: str = "delete"
        self.ignore_labels: Set[str] = {"resting", "no hand", "none"}

        # Normalizes display variants to canonical characters (e.g. "ñ")
        self.sign_mapping: Dict[str, str] = {"n-fuerte": "ñ"}

        # Reset the buffer when no hand has been visible for this many seconds
        self.no_hand_since: Optional[float] = None
        self.no_hand_reset_seconds: float = 0.5

    # --------------------------------------------------
    # Controls
    # --------------------------------------------------

    def toggle_inference(self) -> None:
        """
        Enable or disable gesture prediction.

        When disabled, the landmark buffer is cleared and the last prediction
        is reset so stale values do not persist on screen.
        """
        self.inference_enabled = not self.inference_enabled
        print(f"Inference enabled: {self.inference_enabled}")
        if not self.inference_enabled:
            self.landmark_handler.clear()
            self.last_prediction = "None"
            self.last_confidence = 0.0

    def clear_text_buffer(self) -> None:
        """Clear the accumulated text and reset any pending sign."""
        self.text_buffer.clear()
        self.pending_sign = None
        self.pending_count = 0

    def get_text(self) -> str:
        """
        Return the accumulated text built from confirmed signs.

        :return: Concatenated sign labels as a plain string.
        :rtype: str
        """
        return "".join(self.text_buffer)

    def get_last_prediction(self) -> Tuple[str, float]:
        """
        Return the most recent prediction result.

        :return: Tuple of ``(label, confidence)``.
        :rtype: tuple[str, float]
        """
        return self.last_prediction, self.last_confidence

    # --------------------------------------------------
    # Text buffer logic
    # --------------------------------------------------

    def _handle_text_prediction(self, label: str) -> None:
        """
        Update the pending sign tracker based on the latest prediction.

        Ignored labels (resting, no hand, etc.) are silently dropped.
        A ``"confirm"`` label commits the current pending sign immediately.
        Any other label sets or updates the pending sign.

        :param label: Predicted gesture label (already lowercased).
        :type label: str
        """
        label = label.lower()

        if label in self.ignore_labels:
            return

        if label == self.confirm_label:
            self._commit_pending_sign()
            return

        if self.pending_sign is None:
            self.pending_sign = label
            self.pending_count = 1
            return

        if label == self.pending_sign:
            self.pending_count += 1
        else:
            # Sign changed before confirmation — restart tracking with the new label
            self.pending_sign = label
            self.pending_count = 1

    def _commit_pending_sign(self) -> None:
        """
        Append the pending sign to the text buffer or handle deletion.

        ``"delete"`` with count ≥ 3 clears the entire buffer;
        a single ``"delete"`` pops the last character.
        """
        if self.pending_sign is None:
            return

        sign, count = self.pending_sign, self.pending_count

        if sign == self.delete_label:
            if count >= 3:
                self.text_buffer.clear()
            elif self.text_buffer:
                self.text_buffer.pop()
        else:
            self.text_buffer.append(sign)

        self.pending_sign = None
        self.pending_count = 0

    # --------------------------------------------------
    # Main process (FrameProcessor override)
    # --------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Full per-frame inference pipeline:

        1. Detect hands and draw skeleton.
        2. Extract landmarks from the first detected hand.
        3. Buffer landmarks; run prediction when the buffer is full.
        4. If no hand is detected for ``no_hand_reset_seconds``, reset the buffer.
        5. Draw prediction and text overlays.

        :param frame: RGB input frame of shape ``(H, W, 3)``.
        :type frame: np.ndarray
        :return: Annotated frame with prediction and text overlays.
        :rtype: np.ndarray
        """
        frame = self.tracker.find_hands(frame, draw=True)

        if not self.inference_enabled:
            self._draw_prediction(frame)
            return frame

        landmarks_raw = self.tracker.export_landmarks(frame, hand_id=0, draw=False)

        if landmarks_raw is not None and len(landmarks_raw) > 0:
            self.no_hand_since = None
            self.landmark_handler.add_frame(np.array(landmarks_raw, dtype=np.float32))

            if self.landmark_handler.ready():
                raw = self.landmark_handler.export()
                processed = LandmarkHandler.preprocess_landmarks(raw)

                expected_shape = (
                    self.landmark_handler.buffer.maxlen,
                    self.model_handler.input_shape[1]
                )
                if processed.shape != expected_shape:
                    print(f"Warning: shape {processed.shape} != expected {expected_shape}")
                else:
                    label, confidence = self.model_handler.predict(processed)
                    label = self.sign_mapping.get(label, label)
                    self.last_prediction = label
                    self.last_confidence = confidence
                    self._handle_text_prediction(label)
                    print(f"Prediction: {label} ({confidence:.2f})")
                    print(f"Text buffer: {self.get_text()}")
                
                self.landmark_handler.clear()
        else:
            if self.no_hand_since is None:
                self.no_hand_since = time.time()

            # Reset the buffer only after the hand has been absent long enough
            # to avoid dropping sequences mid-gesture during brief occlusion
            if time.time() - self.no_hand_since >= self.no_hand_reset_seconds:
                self.landmark_handler.clear()
                if self.last_prediction != "No hand":
                    self.last_prediction = "No hand"
                    self.last_confidence = 0.0

        self._draw_prediction(frame)
        return frame

    # --------------------------------------------------
    # Drawing helpers
    # --------------------------------------------------

    def draw_unicode_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_size: int = 32,
        color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Render Unicode text onto a frame using Pillow (supports characters like ñ).

        OpenCV's ``putText`` only supports ASCII; this method converts the frame
        to a Pillow image, draws the text, and converts back to NumPy.

        :param frame: RGB NumPy array to draw on.
        :type frame: np.ndarray
        :param text: Text string to render (may contain non-ASCII characters).
        :type text: str
        :param position: ``(x, y)`` top-left anchor for the text.
        :type position: tuple[int, int]
        :param font_size: Font size in points.
        :type font_size: int
        :param color: RGB color tuple.
        :type color: tuple[int, int, int]
        :return: Frame with text rendered.
        :rtype: np.ndarray
        """
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        draw.text(position, text, font=font, fill=color)
        return np.array(image)

    def _draw_prediction(self, frame: np.ndarray) -> None:
        """
        Overlay the current prediction, accumulated text, and pending sign on the frame.

        Writes directly into ``frame`` via slice assignment.

        :param frame: RGB frame to annotate in-place.
        :type frame: np.ndarray
        """
        if not self.inference_enabled:
            pred_text = "Inference OFF"
            pred_color = (255, 0, 0)
        else:
            pred_text = f"{self.last_prediction} ({self.last_confidence:.2f})"
            pred_color = (0, 255, 0)

        frame[:] = self.draw_unicode_text(frame, pred_text, (10, 40), font_size=34, color=pred_color)
        frame[:] = self.draw_unicode_text(frame, f"Text: {self.get_text()}", (10, 105),
                                          font_size=34, color=(255, 255, 255))

        if self.pending_sign is not None:
            frame[:] = self.draw_unicode_text(
                frame,
                f"Pending: {self.pending_sign} x{self.pending_count}",
                (10, 170), font_size=28, color=(255, 255, 0)
            )


if __name__ == "__main__":
    import sys
    import os
    from core.window.webcam_window import WebcamWindow
    from PySide6.QtWidgets import QApplication
    from core.utils import AppPaths

    AppPaths.load_env()

    MODEL_PATH     = AppPaths.path(os.getenv("SIGN_TRANSLATE_MODEL", "models/sign_lstm.keras"))
    ENCODER_OUTPUT = AppPaths.path(os.getenv("LABEL_ENCODER_FILE", "models/sign_lstm_encoder.npy"))

    app = QApplication(sys.argv)
    processor = InferenceProcessor(MODEL_PATH, ENCODER_OUTPUT)

    window = WebcamWindow(width=1280, height=720, frame_processor=processor)

    def toggle():
        processor.toggle_inference()

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
