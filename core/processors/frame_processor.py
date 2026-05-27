import os
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from core.utils import AppPaths

AppPaths.load_env()


class FrameProcessor(ABC):
    """
    Abstract base class for all frame processors.

    Subclasses implement :meth:`process` to transform frames. 
    
    The optional ``finished`` flag lets a processor signal to the
    host window that it has completed its work (e.g. dataset fully recorded).
    """

    def __init__(self) -> None:
        """Initialize the processor with ``finished = False``."""
        self.finished: bool = False
        self.finished_reason: str = ""

    @classmethod
    def get_default_n_frames(cls) -> int:
        """
        Calculate the default sequence length from environment variables.

        Reads ``CAPTURE_FRAME_RATE_FPS`` and ``CAPTURE_DURATION_SECONDS``
        and returns their product.

        :return: Number of frames per sequence (FPS × duration).
        :rtype: int
        """
        fps: int = int(os.getenv("CAPTURE_FRAME_RATE_FPS", 30))
        duration: int = int(os.getenv("CAPTURE_DURATION_SECONDS", 2))
        return fps * duration

    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame and return the result.

        :param frame: Input RGB frame as a NumPy array of shape ``(H, W, 3)``.
        :type frame: np.ndarray
        :return: Processed frame, same shape and dtype.
        :rtype: np.ndarray
        """
        pass
