import os
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Any, List
from landmark_handler import LandmarkHandler
from dotenv import load_dotenv

load_dotenv()

# Hand Conections for MediaPipe Hands
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Connections between fingers
]

class HandTracker:
    def __init__(
        self,
        model_path: str = None,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        
        if model_path is None:
            root = os.getenv("PROJECT_ROOT")
            if root is None:
                raise ValueError("PROJECT_ROOT not defined in .env")
            
            model_relative = os.getenv("HAND_DETECTION_MODEL")
            if model_relative is None:
                raise ValueError("HAND_DETECTION_MODEL not defined in .env")
            
            model_path = os.path.join(root, model_relative)
        
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # load model .task
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

    @staticmethod
    def normalizedToPixelCoordenates(frame: np.ndarray, landmark: Any):
        h, w, _ = frame.shape
        px, py = int(landmark.x * w), int(landmark.y * h)
        return px, py

    def findHands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        #Hands only uses RGB images and cv by default use BGR
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect
        detection_result = self.detector.detect_for_video(mp_image, int(cv.getTickCount()))
        self.results = detection_result
        
        if draw and detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw connections
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    
                    start_point = HandTracker.normalizedToPixelCoordenates(frame, start)
                    end_point = HandTracker.normalizedToPixelCoordenates(frame, end)
                    cv.line(frame, start_point, end_point, (0, 255, 0), 2)
                
                # Draw dots
                for lm in hand_landmarks:
                    px, py = HandTracker.normalizedToPixelCoordenates(frame, lm)
                    cv.circle(frame, (px, py), 5, (255, 0, 255), cv.FILLED)
        
        return frame

    def findPosition(
        self, frame: np.ndarray, hand_id: int = 0, draw: bool = True
    ) -> List[List[int]]:
        lmList = []
        if hasattr(self, 'results') and self.results and self.results.hand_landmarks:
            if hand_id >= len(self.results.hand_landmarks):
                return lmList
            
            hand_landmarks = self.results.hand_landmarks[hand_id]
            for id, lm in enumerate(hand_landmarks):
                px, py = HandTracker.normalizedToPixelCoordenates(frame, lm)
                lmList.append([id, px, py])
                if draw:
                    cv.circle(frame, (px, py), 10, (255, 255, 0), cv.FILLED)
        return lmList

    def exportLandmarks(
        self, frame: np.ndarray, hand_id: int = 0, draw: bool = True
    ) -> List[List[float]]:
        landmarksPosition = []
        if hasattr(self, 'results') and self.results and self.results.hand_landmarks:
            if hand_id >= len(self.results.hand_landmarks):
                return landmarksPosition
            
            hand_landmarks = self.results.hand_landmarks[hand_id]
            for lm in hand_landmarks:
                print([lm.x, lm.y, lm.z])
                landmarksPosition.append([lm.x, lm.y, lm.z])
                if draw:
                    px, py = HandTracker.normalizedToPixelCoordenates(frame, lm)
                    cv.circle(frame, (px, py), 10, (255, 255, 0), cv.FILLED)
        return landmarksPosition

    def getHandedness(self) -> list:
        """Retrun which hand is left or right"""
        if hasattr(self, 'results') and self.results and self.results.handedness:
            return [hand[0].category_name for hand in self.results.handedness]
        return []


"""

class HandTracker:
    def __init__(
        self,
        max_hands: int = 2,
        model_complexity: float = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # MediaPipe Hands object
        #https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.mpDraw = mp.solutions.drawing_utils

    #convert normalized [0,1] to pixel coordenates
    @staticmethod
    def normalizedToPixelCoordenates(frame: np.ndarray, landmark: Any):
        h, w, _ = frame.shape
        px, py = int(landmark.x * w), int(landmark.y * h)
        return px, py

    def findHands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        #Hands only uses RGB images and cv by default use BGR
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return frame

    def findPosition(
        self, frame: np.ndarray, hand_id: int = 0, draw: bool = True
    ) -> List[List[int]]:
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_id >= len(self.results.multi_hand_landmarks):
                return lmList # hand_id out of range
            
            myHand = self.results.multi_hand_landmarks[hand_id]
            for id, lm in enumerate(myHand.landmark):
                px, py = HandTracker.normalizedToPixelCoordenates(frame, lm)
                lmList.append([id, px, py])
                if draw:
                    cv.circle(frame, (px, py), 10, (255, 0, 255), cv.FILLED)
        return lmList

    def exportLandmarks(
        self, frame: np.ndarray, hand_id: int = 0, draw: bool = True
    ) -> List[List[float]]:
        landmarksPosition = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_id >= len(self.results.multi_hand_landmarks):
                return landmarksPosition # hand_id out of range
            
            myHand = self.results.multi_hand_landmarks[hand_id]
            for lm in myHand.landmark:
                landmarksPosition.append([lm.x, lm.y, lm.z])
                if draw:
                    px, py = HandTracker.normalizedToPixelCoordenates(frame, lm)
                    cv.circle(frame, (px, py), 10, (255, 0, 255), cv.FILLED)
        return landmarksPosition
        
"""
   

class HandTrackingProcessor:
    def __init__(self, n_frames: int = 30):
        self.tracker = HandTracker()
        self.landmark_handler = LandmarkHandler(n_frames)

    def process(self, frame: np.ndarray) -> np.ndarray:
        #Detect and draw hand
        frame = self.tracker.findHands(frame, draw=True)

        #Extract landmarks from the first hand (if exist)
        landmarks_raw = self.tracker.exportLandmarks(frame, hand_id=0, draw=True)
        if landmarks_raw:
            # Convert into a numpy array (21,3) and add into the buffer
            landmarks_np = np.array(landmarks_raw, dtype=np.float32)
            self.landmark_handler.add_frame(landmarks_np)

            # if the buffer it's full preprocess and export data (now just print)
            if self.landmark_handler.ready():
                raw = self.landmark_handler.export()
                processed = self.landmark_handler.preprocess_landmarks(raw)
                print(f"Buffer is ready: original {raw.shape} → process {processed.shape}")

                self.landmark_handler.clear()  # restart buffer

        return frame
    

if __name__ == "__main__":

    from window.webcam_window import WebcamWindow
    from window.video_window import VideoWindow
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    fps = int(os.getenv("CAPTURE_FRAME_RATE_FPS", 0))
    duration = int(os.getenv("CAPTURE_DURATION_SECONDS", 0))
    n_frames = fps * duration
    window = WebcamWindow(
        0,
        width=1280, 
        height=720, 
        frame_processor=HandTrackingProcessor(
            n_frames=n_frames or 30
        )
    )

    window.show()
    sys.exit(app.exec())