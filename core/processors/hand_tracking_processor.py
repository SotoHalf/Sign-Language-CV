import numpy as np
from core.hand_tracker import HandTracker
from core.processors.frame_processor import FrameProcessor


class HandTrackingProcessor(FrameProcessor):
    """
    Detects hands in every frame and draws the 21-point skeleton on it.
    """

    def __init__(self) -> None:
        """Initialize the processor and load the MediaPipe hand tracker."""
        super().__init__()
        self.tracker: HandTracker = HandTracker()

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect hands and draw landmarks on the frame.

        :param frame: RGB input frame of shape ``(H, W, 3)``.
        :type frame: np.ndarray
        :return: Frame annotated with hand skeleton and landmark dots.
        :rtype: np.ndarray
        """
        return self.tracker.find_hands(frame, draw=True)
