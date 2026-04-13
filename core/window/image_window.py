import cv2
from core.window.base_window import BaseWindow


class ImageWindow(BaseWindow):

    def __init__(self, image_path, **kwargs):
        self.image_path = image_path
        super().__init__(**kwargs)

    def setup(self):
        self.frame = cv2.imread(self.image_path)
        if self.frame is None:
            raise FileNotFoundError(self.image_path)

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def get_frame(self):
        return self.frame