import numpy as np
from core.hand_tracker import HandTracker
from core.processors.frame_processor import FrameProcessor


class HandTrackingProcessor(FrameProcessor):

    def __init__(self):
        super().__init__()
        self.tracker = HandTracker()

    def process(self, frame: np.ndarray) -> np.ndarray:
        #Detect and draw hand
        frame = self.tracker.findHands(frame, draw=True)

        return frame