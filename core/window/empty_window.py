import cv2
from core.window.base_window import BaseWindow


class EmptyWindow(BaseWindow):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_frame(self):
        return None