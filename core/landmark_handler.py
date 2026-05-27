import os
from core.utils import AppPaths
from collections import deque
from typing import Optional
import numpy as np
import pandas as pd

AppPaths.load_env()


class LandmarkHandler:
    """
    Manages a temporal ring-buffer of hand landmark frames, normalizes them,
    and prepares the resulting sequence as input for the LSTM model.

    Each frame is a flattened array of 21 landmarks × 3 coordinates = 63 values.
    After preprocessing, delta features are appended, producing 126 values per frame.
    """

    # Landmark indices as defined by MediaPipe, configurable via .env
    LANDMARK_WRIST: int        = int(os.getenv("HAND_LANDMARK_WRIST", 0))
    LANDMARK_MIDDLE_FINGER: int = int(os.getenv("HAND_LANDMARK_MIDDLE_FINGER", 12))
    LANDMARK_THUMBCMC: int     = int(os.getenv("HAND_LANDMARK_THUMBCMC", 1))
    LANDMARK_PINCKYMCP: int    = int(os.getenv("HAND_LANDMARK_PINCKYMCP", 17))
    TOTAL_LANDMARKS: int       = int(os.getenv("HAND_TOTAL_LANDMARKS", 21))

    def __init__(self, n_frames: int) -> None:
        """
        Create an internal ring-buffer of size ``n_frames``.

        When the buffer is full, the handler is ready to produce model input.
        For static gestures use ``n_frames=1``; for dynamic sequences use
        ``FPS × duration_seconds``.

        :param n_frames: Number of frames the buffer can hold.
        :type n_frames: int
        """
        self.buffer: deque = deque(maxlen=n_frames)

    def add_frame(self, landmarks: np.ndarray) -> None:
        """
        Flatten and append a single frame of landmarks to the buffer.

        :param landmarks: Array of shape ``(21, 3)`` with (x, y, z) per landmark.
        :type landmarks: np.ndarray
        """
        self.buffer.append(landmarks.flatten())

    def ready(self) -> bool:
        """
        Return ``True`` when the buffer has reached its configured capacity.

        :return: Whether the buffer is full.
        :rtype: bool
        """
        return len(self.buffer) == self.buffer.maxlen

    def clear(self) -> None:
        """Remove all frames from the buffer."""
        self.buffer.clear()

    def export(self) -> np.ndarray:
        """
        Export the raw buffer contents as a NumPy array.

        :return: Array of shape ``(n_frames, 63)``.
        :rtype: np.ndarray
        """
        return np.array(self.buffer)

    # --------------------------------------------------
    # Normalization / Transformation / Preprocessing
    # --------------------------------------------------

    @staticmethod
    def get_landmark_cols(ndarray: np.ndarray, index: int, size: int = 3) -> np.ndarray:
        """
        Extract the (x, y, z) columns for the landmark at the given index.

        :param ndarray: Flattened landmark array of shape ``(n_frames, 63)``.
        :type ndarray: np.ndarray
        :param index: MediaPipe landmark index (0–20).
        :type index: int
        :param size: Number of coordinate components per landmark (default 3 for x, y, z).
        :type size: int
        :return: Array of shape ``(n_frames, size)``.
        :rtype: np.ndarray
        """
        cols = index * size + np.arange(size)
        return ndarray[:, cols]

    @classmethod
    def _preprocess_scale(cls, landmarks_frame_data: np.ndarray) -> np.ndarray:
        """
        Scale normalization: divide all coordinates by the mean Euclidean distance
        between thumb-CMC and pinky-MCP across all frames.

        This makes the hand size invariant to camera distance.

        :param landmarks_frame_data: Array of shape ``(n_frames, 63)``.
        :type landmarks_frame_data: np.ndarray
        :return: Scaled array of the same shape.
        :rtype: np.ndarray
        """
        landmarks_frame_data = np.array(landmarks_frame_data, copy=True)

        thumb_cmc = cls.get_landmark_cols(landmarks_frame_data, cls.LANDMARK_THUMBCMC)
        pinky_mcp = cls.get_landmark_cols(landmarks_frame_data, cls.LANDMARK_PINCKYMCP)

        # Mean distance across frames gives a stable scale reference
        scale: float = np.mean(np.linalg.norm(pinky_mcp - thumb_cmc, axis=1))
        if scale == 0:
            scale = 1e-6  # avoid division by zero on empty/degenerate hands

        return landmarks_frame_data / scale

    @classmethod
    def _preprocess_position(cls, landmarks_frame_data: np.ndarray) -> np.ndarray:
        """
        Position normalization: translate all landmarks so that the wrist at
        frame 0 becomes the origin (0, 0, 0).

        Anchoring to frame 0 instead of each individual frame preserves
        relative hand movement across the sequence.

        :param landmarks_frame_data: Array of shape ``(n_frames, 63)``.
        :type landmarks_frame_data: np.ndarray
        :return: Translated array of the same shape.
        :rtype: np.ndarray
        """
        landmarks_frame_data = np.array(landmarks_frame_data, copy=True)

        wrist_positions = cls.get_landmark_cols(landmarks_frame_data, cls.LANDMARK_WRIST)

        reshaped = landmarks_frame_data.reshape(
            landmarks_frame_data.shape[0], cls.TOTAL_LANDMARKS, 3
        )

        # Subtract only the first frame's wrist so motion across frames is retained
        reshaped -= wrist_positions[0][None, None, :]

        return reshaped.reshape(landmarks_frame_data.shape)

    @classmethod
    def _preprocess_delta(
        cls,
        landmarks_frame_data: np.ndarray,
        cols: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute frame-to-frame velocity (delta) for all or selected columns.

        The first row is always zero (no previous frame to subtract from).

        :param landmarks_frame_data: Array of shape ``(n_frames, n_cols)``.
        :type landmarks_frame_data: np.ndarray
        :param cols: Column indices to compute delta for. If ``None``, all columns are used.
        :type cols: np.ndarray, optional
        :return: Delta array with the same shape as the input (or selected subset).
        :rtype: np.ndarray
        """
        landmarks_frame_data = np.array(landmarks_frame_data, copy=True)
        data_to_delta = landmarks_frame_data if cols is None else landmarks_frame_data[:, cols]

        delta = np.zeros_like(data_to_delta)
        delta[1:] = data_to_delta[1:] - data_to_delta[:-1]

        return delta

    @classmethod
    def preprocess_landmarks(
        cls,
        landmarks_frame_data: np.ndarray,
        n_frames: Optional[int] = None
    ) -> np.ndarray:
        """
        Full preprocessing pipeline applied before feeding data to the model:

        1. Position normalization (wrist at frame 0 as origin).
        2. Scale normalization (thumb-CMC ↔ pinky-MCP distance).
        3. Append per-frame velocity (delta) as extra features.

        Output shape is ``(n_frames, 126)``:
        63 landmark coordinates + 63 delta coordinates.

        :param landmarks_frame_data: Raw buffer export of shape ``(n_frames, 63)``.
        :type landmarks_frame_data: np.ndarray
        :param n_frames: If the sequence is shorter than this value, the last frame
            is repeated to pad it to the required length.
        :type n_frames: int, optional
        :return: Processed array of shape ``(n_frames, 126)``.
        :rtype: np.ndarray
        """
        result = np.array(landmarks_frame_data, copy=True)
        result = cls._preprocess_position(result)
        result = cls._preprocess_scale(result)

        delta = cls._preprocess_delta(result)
        result = np.hstack([result, delta])

        # Pad with the last frame if the sequence is shorter than expected
        if n_frames and result.shape[0] < n_frames:
            last_frame = result[-1] if result.shape[0] > 0 else np.zeros(result.shape[1], dtype=np.float32)
            padding = np.tile(last_frame, (n_frames - result.shape[0], 1))
            result = np.vstack([result, padding])

        return result

    @classmethod
    def to_dataframe(cls, landmarks_frame_data: np.ndarray) -> pd.DataFrame:
        """
        Convert a landmark array to a DataFrame with descriptive column names.

        - 63-column arrays  → ``lmX_x``, ``lmX_y``, ``lmX_z`` (raw landmarks).
        - 126-column arrays → above + ``dX_x``, ``dX_y``, ``dX_z`` (delta values).

        :param landmarks_frame_data: Array of shape ``(n_frames, 63)`` or ``(n_frames, 126)``.
        :type landmarks_frame_data: np.ndarray
        :return: DataFrame with named columns.
        :rtype: pd.DataFrame
        :raises ValueError: If the number of columns does not match 63 or 126.
        """
        _, n_cols = landmarks_frame_data.shape

        lm_cols = [f"lm{i}_{ax}" for i in range(cls.TOTAL_LANDMARKS) for ax in ('x', 'y', 'z')]

        if n_cols == cls.TOTAL_LANDMARKS * 3:
            column_names = lm_cols
        else:
            delta_cols = [f"d{i}_{ax}" for i in range(cls.TOTAL_LANDMARKS) for ax in ('x', 'y', 'z')]
            column_names = lm_cols + delta_cols

        if n_cols != len(column_names):
            raise ValueError(f"Shape mismatch: array has {n_cols} columns, expected {len(column_names)}")

        return pd.DataFrame(landmarks_frame_data, columns=column_names)


if __name__ == "__main__":
    handler = LandmarkHandler(n_frames=2)

    frame_0 = np.array([
        [0.5795815, 0.73815686, -1.8038166e-07],
        [0.55215597, 0.6747151, -0.01099427],
        [0.5049775, 0.6430089, -0.02111784],
        [0.4683286, 0.6703038, -0.0297561],
        [0.45375395, 0.7237727, -0.03962269],
        [0.49235883, 0.62782544, -0.02785791],
        [0.45791447, 0.6919856, -0.04511432],
        [0.4612876, 0.7189338, -0.05663097],
        [0.4696094, 0.7235899, -0.06393994],
        [0.5109241, 0.66942954, -0.03136811],
        [0.47782707, 0.745063, -0.04599185],
        [0.48567954, 0.7604859, -0.05130501],
        [0.4981397, 0.7540408, -0.05566616],
        [0.53220457, 0.71796024, -0.03511057],
        [0.5018302, 0.78522193, -0.0497081],
        [0.50925815, 0.79420924, -0.04805692],
        [0.520422, 0.7833466, -0.04640864],
        [0.5514658, 0.7645711, -0.03927271],
        [0.5284891, 0.8183487, -0.05078094],
        [0.5353743, 0.8214972, -0.04739401],
        [0.5443849, 0.80915415, -0.0436805],
    ], dtype=np.float32)

    frame_1 = np.array([
        [0.56771237, 0.67958665, -3.7033789e-07],
        [0.54511535, 0.66276723, -0.01225006],
        [0.52102906, 0.62057233, -0.02380931],
        [0.5017718, 0.58466864, -0.03451538],
        [0.48849005, 0.56022024, -0.04429118],
        [0.56286085, 0.49751887, -0.01910602],
        [0.52913374, 0.49198934, -0.04274818],
        [0.5115025, 0.5324064, -0.06085577],
        [0.50408137, 0.5721977, -0.06990312],
        [0.56831604, 0.49835512, -0.02099565],
        [0.5321308, 0.5006625, -0.04019247],
        [0.5101404, 0.5454227, -0.04964491],
        [0.5025562, 0.5887555, -0.05415682],
        [0.56973356, 0.5147706, -0.02491779],
        [0.53236204, 0.5216817, -0.04318246],
        [0.5104225, 0.565379, -0.04597792],
        [0.5034066, 0.60528, -0.04480152],
        [0.5663274, 0.5443039, -0.02941359],
        [0.5284542, 0.5526613, -0.0433124],
        [0.51208436, 0.5831215, -0.04355562],
        [0.5073924, 0.6091327, -0.04049787],
    ], dtype=np.float32)

    handler.add_frame(frame_0)
    handler.add_frame(frame_1)

    if handler.ready():
        raw_landmarks = handler.export()
        print("Landmarks raw shape:", raw_landmarks.shape)

        processed_landmarks = handler.preprocess_landmarks(raw_landmarks)
        print("Landmarks processed shape:", processed_landmarks.shape)
        print(processed_landmarks)

        handler.clear()
