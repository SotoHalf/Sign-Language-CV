import cv2
from core.window.base_window import BaseWindow


class VideoWindow(BaseWindow):

    def __init__(self, video_path, **kwargs):
        self.video_path = video_path
        super().__init__(**kwargs)

    def setup(self):
        self.open_video()
        
        # sync FPS
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            interval = int(1000 / fps)
            self.timer.setInterval(interval)

    def set_video_path(self, video_path):
        if not video_path:
            return 

        if video_path != self.video_path:
            self.video_path = video_path
            self.open_video()

    def open_video(self):
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"The video could not be opened: {self.video_path}")
    
    def get_frame(self):
        ret, frame = self.cap.read()

        if ret:
            #cv by default use BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        
        return None
    
    def is_finished(self):
        if not hasattr(self, "cap") or not self.cap.isOpened():
            return True

        current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        return current >= total

    def cleanup(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        self.timer.start(BaseWindow.DEFAULT_TIMER)

    def get_fps(self):
        if not hasattr(self, "cap") or not self.cap.isOpened():
            return 30
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30

    def seek_seconds(self, seconds: float):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)

    def seek_frame(self, frame_number: int):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_current_time(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return 0.0




