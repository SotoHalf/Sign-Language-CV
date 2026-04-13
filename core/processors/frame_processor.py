import os
from abc import ABC, abstractmethod
import numpy as np
from core.utils import load_env

load_env()

class FrameProcessor(ABC):
    """
    Interface for processing frames.
    """

    @classmethod
    def get_default_n_frames(cls) -> int:
        fps = int(os.getenv("CAPTURE_FRAME_RATE_FPS", 30))
        duration = int(os.getenv("CAPTURE_DURATION_SECONDS", 2))
        return fps * duration

    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a frame and returns the processed frame.

        :param frame: Input frame as a NumPy array
        :return: Processed frame as a NumPy array
        """
        pass