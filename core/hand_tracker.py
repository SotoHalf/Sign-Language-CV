import os
import time
import cv2
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

        base_options = python.BaseOptions(model_asset_path=model_path)

        # load model .task
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
        #px, py = int(landmark.x * w), int(landmark.y * h)
        px = min(max(int(landmark.x * w), 0), w-1)
        py = min(max(int(landmark.y * h), 0), h-1)
        return px, py

    def findHands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
    
        # Resize to improve acceleration
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            new_w, new_h = 640, int(h * scale)
            frame_small = cv2.resize(frame, (new_w, new_h))
        else:
            frame_small = frame
            scale = 1.0

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_small)

        # Detect
        timestamp_ms = int(time.time() * 1000)
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        self.results = detection_result

        if draw and detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw Hand connections between points
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]

                    # Scale coordinates to the original frame size
                    x1 = int(start.x * w)
                    y1 = int(start.y * h)
                    x2 = int(end.x * w)
                    y2 = int(end.y * h)

                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw landmark points
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (255, 0, 255), cv2.FILLED)

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
                    cv2.circle(frame, (px, py), 10, (255, 255, 0), cv2.FILLED)
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
                landmarksPosition.append([lm.x, lm.y, lm.z])
                if draw:
                    px, py = HandTracker.normalizedToPixelCoordenates(frame, lm)
                    cv2.circle(frame, (px, py), 10, (255, 255, 0), cv.FILLED)
        return landmarksPosition

    def getHandedness(self) -> list:
        """Retrun which hand is left or right"""
        if hasattr(self, 'results') and self.results and self.results.handedness:
            return [hand[0].category_name for hand in self.results.handedness]
        return []

class HandTrackingProcessor:
    def __init__(self, n_frames: int = 30):
        self.tracker = HandTracker()
        self.landmark_handler = LandmarkHandler(n_frames)

    def process(self, frame: np.ndarray) -> np.ndarray:
        #Detect and draw hand
        frame = self.tracker.findHands(frame, draw=True)

        #Extract landmarks from the first hand (if exist)
        landmarks_raw = self.tracker.exportLandmarks(frame, hand_id=0, draw=False)

        """
        if landmarks_raw:
            # Convert into a numpy array (21,3) and add into the buffer
            landmarks_np = np.array(landmarks_raw, dtype=np.float32)
            self.landmark_handler.add_frame(landmarks_np)

            # if the buffer it's full preprocess and export data (now just print)
            if self.landmark_handler.ready():
                raw = self.landmark_handler.export()
                processed = self.landmark_handler.preprocess_landmarks(raw)
                #print(f"Buffer is ready: original {raw.shape} → process {processed.shape}")
                
                #df_landmarks = self.landmark_handler.to_dataframe(processed)
                #df_landmarks.to_csv(f"processed_landmarks_{self.test}.csv", index=False)

                self.landmark_handler.clear()  # restart buffer
        """

        return frame
    

if __name__ == "__main__":
    
    from window.webcam_window import WebcamWindow
    from window.video_window import VideoWindow
    from window.image_window import ImageWindow
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    fps = int(os.getenv("CAPTURE_FRAME_RATE_FPS", 0))
    duration = int(os.getenv("CAPTURE_DURATION_SECONDS", 0))
    n_frames = fps * duration

    window = WebcamWindow(
        0,
        width=480, 
        height=320, 
        #width=1280, 
        #height=720, 
        frame_processor=HandTrackingProcessor(
            n_frames=n_frames or 30
        )
    )

    #SET RECORD BUTTON
    def button_has_been_pressed():
        print("RECORD")
    window.add_button("record_button", "R", button_has_been_pressed)

    def button_has_been_pressed():
        print("OTHER")
    window.add_button("other_button", "O", button_has_been_pressed)

    """
    window = VideoWindow(
        "test_video.mp4",
        width=1280, 
        height=720, 
        frame_processor=HandTrackingProcessor(
            n_frames=n_frames or 30
        )
    )
    """

    window.show()
    sys.exit(app.exec())

    """
    cap = cv2.VideoCapture(0)
    hand_tracker = HandTracker()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_with_hands = hand_tracker.findHands(frame)
        cv2.imshow("Frame", frame_with_hands)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    """
    
