import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from core.utils import AppPaths
import numpy as np
from typing import Any, List, Optional

AppPaths.load_env()

# 21-point hand skeleton defined as pairs of landmark indices.
# Order matches MediaPipe's landmark numbering (0 = wrist, 4 = thumb tip, etc.).
HAND_CONNECTIONS: List[tuple] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm cross-connections
]


class HandTracker:
    """
    Wraps the MediaPipe HandLandmarker to detect and track hands in video frames.
    Runs in VIDEO mode for continuous, timestamp-based tracking.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Load the MediaPipe ``.task`` model and configure the hand landmarker.

        :param model_path: Path to the ``.task`` model file. If ``None``,
            reads ``HAND_DETECTION_MODEL`` from the environment.
        :type model_path: str, optional
        :param max_hands: Maximum number of hands to detect simultaneously.
        :type max_hands: int
        :param min_detection_confidence: Minimum confidence score for initial detection.
        :type min_detection_confidence: float
        :param min_tracking_confidence: Minimum confidence score to continue tracking.
        :type min_tracking_confidence: float
        :raises ValueError: If ``model_path`` is None and the env variable is not set.
        """
        if model_path is None:
            model_relative = os.getenv("HAND_DETECTION_MODEL")
            if model_relative is None:
                raise ValueError("HAND_DETECTION_MODEL not defined in .env")
            model_path = AppPaths.path(model_relative)

        self.max_hands: int = max_hands
        self.min_detection_confidence: float = min_detection_confidence
        self.min_tracking_confidence: float = min_tracking_confidence

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Filled after each call to find_hands(); consumed by find_position / export_landmarks
        self.results: Any = None

    @staticmethod
    def _normalized_to_pixel_coords(frame: np.ndarray, landmark: Any) -> tuple:
        """
        Convert a landmark's normalized [0, 1] coordinates to pixel coordinates,
        clamped to the frame boundaries.

        :param frame: The image used to determine pixel dimensions.
        :type frame: np.ndarray
        :param landmark: MediaPipe landmark object with ``.x`` and ``.y`` attributes.
        :type landmark: Any
        :return: Tuple ``(pixel_x, pixel_y)``.
        :rtype: tuple[int, int]
        """
        h, w, _ = frame.shape
        px = min(max(int(landmark.x * w), 0), w - 1)
        py = min(max(int(landmark.y * h), 0), h - 1)
        return px, py

    def find_hands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Detect hands in the frame and store the results for subsequent calls.

        Internally downscales frames wider than 640 px for faster inference,
        then maps landmarks back to the original resolution for drawing.

        :param frame: RGB image as a NumPy array.
        :type frame: np.ndarray
        :param draw: Whether to draw skeleton connections and landmark dots on the frame.
        :type draw: bool
        :return: The (optionally annotated) frame.
        :rtype: np.ndarray
        """
        h, w = frame.shape[:2]

        # Downscale for faster inference; landmarks are still mapped to original size
        if w > 640:
            scale = 640 / w
            frame_small = cv2.resize(frame, (640, int(h * scale)))
        else:
            frame_small = frame

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_small)
        timestamp_ms = int(time.time() * 1000)
        self.results = self.detector.detect_for_video(mp_image, timestamp_ms)

        if draw and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:

                # Draw Hand connections between points
                for start_idx, end_idx in HAND_CONNECTIONS:
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]

                    # Scale coordinates to the original frame size
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                 # Draw landmark points
                for lm in hand_landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (255, 0, 255), cv2.FILLED)

        return frame

    def find_position(
        self, frame: np.ndarray, hand_id: int = 0, draw: bool = True
    ) -> List[List[int]]:
        """
        Return pixel positions for each landmark of the specified hand.

        Must be called after :meth:`find_hands`.

        :param frame: RGB image used for coordinate scaling and optional drawing.
        :type frame: np.ndarray
        :param hand_id: Index of the hand to query (0 = first detected).
        :type hand_id: int
        :param draw: Whether to draw landmark circles on the frame.
        :type draw: bool
        :return: List of ``[landmark_id, pixel_x, pixel_y]`` for each of the 21 landmarks,
            or an empty list if the requested hand was not detected.
        :rtype: list[list[int]]
        """
        lm_list: List[List[int]] = []
        if self.results and self.results.hand_landmarks:
            if hand_id >= len(self.results.hand_landmarks):
                return lm_list

            for idx, lm in enumerate(self.results.hand_landmarks[hand_id]):
                px, py = HandTracker._normalized_to_pixel_coords(frame, lm)
                lm_list.append([idx, px, py])
                if draw:
                    cv2.circle(frame, (px, py), 6, (255, 255, 0), cv2.FILLED)
        return lm_list

    def export_landmarks(
        self, frame: np.ndarray, hand_id: int = 0, draw: bool = True
    ) -> List[List[float]]:
        """
        Return the normalized (x, y, z) coordinates for each landmark of the specified hand.

        Must be called after :meth:`find_hands`. Values are in the [0, 1] range as
        provided by MediaPipe (z is relative depth, not absolute).

        :param frame: RGB image used for coordinate scaling and optional drawing.
        :type frame: np.ndarray
        :param hand_id: Index of the hand to query (0 = first detected).
        :type hand_id: int
        :param draw: Whether to draw landmark circles on the frame.
        :type draw: bool
        :return: List of ``[x, y, z]`` per landmark (21 entries), or ``[]`` if not detected.
        :rtype: list[list[float]]
        """
        landmarks_position: List[List[float]] = []
        if self.results and self.results.hand_landmarks:
            if hand_id >= len(self.results.hand_landmarks):
                return landmarks_position

            for lm in self.results.hand_landmarks[hand_id]:
                landmarks_position.append([lm.x, lm.y, lm.z])
                if draw:
                    px, py = HandTracker._normalized_to_pixel_coords(frame, lm)
                    cv2.circle(frame, (px, py), 6, (255, 255, 0), cv2.FILLED)
        return landmarks_position

    def get_handedness(self) -> List[str]:
        """
        Return the handedness label for each detected hand.

        Must be called after :meth:`find_hands`.

        :return: List of ``'Left'`` / ``'Right'`` strings, one per detected hand.
        :rtype: list[str]
        """
        if self.results and self.results.handedness:
            return [hand[0].category_name for hand in self.results.handedness]
        return []


if __name__ == "__main__":
    from window.webcam_window import WebcamWindow
    from PySide6.QtWidgets import QApplication
    from core.processors.hand_tracking_processor import HandTrackingProcessor
    import sys

    app = QApplication(sys.argv)
    window = WebcamWindow(
        width=480,
        height=320,
        frame_processor=HandTrackingProcessor()
    )
    window.show()
    sys.exit(app.exec())
