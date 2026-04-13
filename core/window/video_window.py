import cv2
from core.window.base_window import BaseWindow


class VideoWindow(BaseWindow):

    def __init__(self, video_path, **kwargs):
        self.video_path = video_path
        super().__init__(**kwargs)

    def setup(self):
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"he video could not be opened: {self.video_path}")

        # sync FPS
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            interval = int(1000 / fps)
            self.timer.setInterval(interval)

    def get_frame(self):
        ret, frame = self.cap.read()

        if ret:
            #cv by default use BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # return None
        return None

    def cleanup(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        self.timer.start(BaseWindow.DEFAULT_TIMER)
