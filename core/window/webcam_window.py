import cv2
from core.window.base_window import BaseWindow


class WebcamWindow(BaseWindow):

    def __init__(self, webcam_id, **kwargs):
        self.webcam_id = webcam_id
        super().__init__(**kwargs)

    def setup(self):
        self.cap = cv2.VideoCapture(self.webcam_id)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.timer.setInterval(int(1000 / fps))

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            #cv by default use BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        return None
    
    def cleanup(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        self.timer.start(BaseWindow.DEFAULT_TIMER)