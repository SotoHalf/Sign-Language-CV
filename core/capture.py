import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import glob
import os

import pandas as pd
from typing import Any, List
from typing import Tuple
import core.config as config


# Esta classe esta bastante bien no la modificaria mucho
class HandTracker:
    def __init__(
        self,
        max_hands: int = 2,
        model_complexity: float = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        
        self.max_hands: int = max_hands
        self.model_complexity: float = model_complexity
        self.min_detection_confidence: float = min_detection_confidence
        self.min_tracking_confidence: float = min_tracking_confidence

        # MediaPipe Hands object
        #https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
        self.mpHands = mp.solutions.hands
        self.hands: Any = self.mpHands.Hands(
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.mpDraw = mp.solutions.drawing_utils

    #convert normalized [0,1] to pixel coordenates
    def normalizedToPixelCoordenates(frame: np.ndarray, landmark: Any):
        h: float
        w: float

        h, w, _ = frame.shape
        px, py = int(landmark.x * w), int(landmark.y * h)
        return px, py

    def findHands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:

        frameRGB: np.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #Hands only uses RGB images and cv by default use BGR
        self.results: np.ndarray = self.hands.process(frameRGB)

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
        
        lmList: List[List[int]] = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_id >= len(self.results.multi_hand_landmarks):
                return lmList  # hand_id out of range
            myHand: mp.framework.formats.landmark_pb2.NormalizedLandmarkList = self.results.multi_hand_landmarks[hand_id]
            
            for id, lm in enumerate(myHand.landmark):
                px, py = HandTracker.normalizedToPixelCoordenates(frame,lm)
                lmList.append([id, px, py])
                if draw:
                    cv.circle(frame, (px, py), 10, (255, 0, 255), cv.FILLED)

        return lmList
    
    def exportLandmarks(
        self,
        frame: np.ndarray,
        hand_id: int = 0,
        draw: bool = True
    ) -> List[List[float]]:
        
        # yo creo que eso se puede mejorar
        landmarksPosition: List[List[int]] = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_id >= len(self.results.multi_hand_landmarks):
                return landmarksPosition  # hand_id out of range
            myHand: mp.framework.formats.landmark_pb2.NormalizedLandmarkList = self.results.multi_hand_landmarks[hand_id]

            for lm in myHand.landmark:
                
                landmarksPosition.append([lm.x, lm.y, lm.z])
                if draw:
                    px, py = HandTracker.normalizedToPixelCoordenates(frame,lm)
                    cv.circle(frame, (px, py), 10, (255, 0, 255), cv.FILLED)

        return landmarksPosition

        
    
class VideoManager:
    def __init__(self, webcam_id: int = 0, imshow_name: str = "Image", show_fps: bool = True, stop_key: str = 'q') -> None:
        self.webcam_id: int = webcam_id
        self.imshow_name: str = imshow_name
        self.show_fps: bool = show_fps
        self._current_frame_rate: int = 0
        self.last_key: int = -1
        self.stop_key: str = stop_key[:1]
        self.last_time: float = 0
        self.current_time: float = 0

        self.capture: cv.VideoCapture = cv.VideoCapture(self.webcam_id)

        self.running: bool = True

    @property
    def current_frame_rate(self):
        return self._current_frame_rate
    
    def check_key_pressed(self, key):
        return self.last_key == ord(key)

    def read_frame(self) -> Tuple[bool, np.ndarray]:
        success: bool
        frame: np.ndarray

        if self.is_running():
            success, frame = self.capture.read()
            self.last_key = cv.waitKey(1) & 0xFF

            if not success:
                raise RuntimeError("Can't receive frame (stream end?). Exiting ...")

            self.current_time = time.time()
            fps = 1/(self.current_time-self.last_time)
            self.last_time = self.current_time
            self._current_frame_rate = int(fps)

            if self.show_fps:
                self.set_text(frame, str(self._current_frame_rate), (10,70), 3, (255,0,255), 3)
            
            return True, frame
        
        return False, np.array([])
    
    def show_frame(self, frame) -> None:
        cv.imshow(self.imshow_name, frame)

    def set_text(
            self, 
            frame: np.ndarray, 
            txt: str, 
            position: Tuple[float, float], 
            scale: float, 
            color: Tuple[float, float, float],
            thickness: int
        ) -> None:
        cv.putText(frame, txt, position, cv.FONT_HERSHEY_PLAIN, scale, color, thickness)

    def get_text_size(
        self,
        text: str,
        font_scale: float = 2,
        thickness: int = 2,
        font=cv.FONT_HERSHEY_PLAIN
    ) -> Tuple[int, int]:

        text_size, _ = cv.getTextSize(text, font, font_scale, thickness)
        width, height = text_size
        return width, height

    #check if the key to stop has been pressed
    def is_running(self) -> bool:
        if self.check_key_pressed(self.stop_key):
            self.terminate()
        
        return self.running

    def terminate(self) -> None:
        self.capture.release()
        cv.destroyAllWindows()
        self.running = False


class RecordingPreview:
    def __init__(
        self,
        dir: str,
        window_name: str = "Preview",
        frame_size: Tuple[int,int] = (640,480),
        delay_ms: int = 30
    ):
        self.dir = dir
        self.window_name = window_name
        self.frame_w, self.frame_h = frame_size
        self.delay_ms = delay_ms

        # Obtener todos los CSV
        self.csv_files = sorted(glob.glob(os.path.join(self.dir, "*.csv")))
        if not self.csv_files:
            raise RuntimeError(f"No CSV files found in {dir}")

        cv.namedWindow(self.window_name)

    def _landmarks_to_frame(self, row, file_name):
        """
        Convierte los landmarks del CSV a una imagen para mostrar,
        incluyendo el delta de la muñeca (dw).
        """
        frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        cx, cy = self.frame_w // 2, self.frame_h // 2
        scale = 100  # Ajusta el tamaño de la mano en pantalla

        # Extraer solo las columnas de landmarks (x,y,z)
        lm_cols = [f"lm{i}_{axis}" for i in range(config.HAND_LANDMARKS_LEN) for axis in ("x","y","z")]
        lms = row[lm_cols].values.reshape(-1, 3)

        # Extraer delta muñeca
        dw = np.array([row["dw_x"], row["dw_y"], row["dw_z"]])

        # Dibujar landmarks
        for lm in lms:
            px = int(lm[0] * scale + cx)
            py = int(lm[1] * scale + cy)
            cv.circle(frame, (px, py), 6, (255, 0, 255), -1)

        # Dibujar muñeca como círculo distinto
        wrist_px = int(dw[0] * scale + cx)
        wrist_py = int(dw[1] * scale + cy)
        cv.circle(frame, (wrist_px, wrist_py), 8, (0, 255, 0), -1)

        cv.putText(
            frame, 
            file_name, 
            (10, 30), 
            cv.FONT_HERSHEY_SIMPLEX, 
            1,
            (255, 255, 255),
            2,
            cv.LINE_AA
        )

        return frame

    def play_all(self):
        print(f"Playing all {len(self.csv_files)} recordings. Press 'q' to quit.")

        for idx, file in enumerate(self.csv_files, 1):
            print(f"Recording {idx}/{len(self.csv_files)}: {os.path.basename(file)}")
            df = pd.read_csv(file)

            for _, row in df.iterrows():
                frame = self._landmarks_to_frame(row, os.path.basename(file))
                cv.imshow(self.window_name, frame)
                key = cv.waitKey(self.delay_ms) & 0xFF
                if key == ord('q'):
                    cv.destroyAllWindows()
                    return

        cv.destroyAllWindows()

def main():
    vm = VideoManager(
        webcam_id = 0,
        stop_key = 'q'
    )
    
    hand_tracker = HandTracker()

    while True:
        success, frame = vm.read_frame()
        if not success:
            break
        
        frame_with_hands = hand_tracker.findHands(frame)
        lmList = hand_tracker.findPosition(frame)
        if lmList:
            print(lmList[4])
        
        vm.show_frame(frame_with_hands)

if __name__ == "__main__":
    main()