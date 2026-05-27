import cv2
import numpy as np
from typing import Optional

from core.window.base_window import BaseWindow


class ImageWindow(BaseWindow):
    """
    Window that displays a single static image.

    The image is loaded once in :meth:`setup` and served on every timer tick.
    """

    def __init__(self, image_path: str, **kwargs) -> None:
        """
        :param image_path: Path to the image file (any format supported by OpenCV).
        :type image_path: str
        :param kwargs: Forwarded to :class:`~core.window.base_window.BaseWindow`.
        :raises FileNotFoundError: If the image cannot be loaded during :meth:`setup`.
        """
        self.image_path: str = image_path
        super().__init__(**kwargs)

    def setup(self) -> None:
        """
        Load the image from disk and convert it to RGB.

        :raises FileNotFoundError: If the file does not exist or cannot be read.
        """
        self.frame: Optional[np.ndarray] = cv2.imread(self.image_path)
        if self.frame is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def get_frame(self) -> np.ndarray:
        """
        Return the loaded image on every timer tick.

        :return: RGB NumPy array of shape ``(H, W, 3)``.
        :rtype: np.ndarray
        """
        return self.frame
