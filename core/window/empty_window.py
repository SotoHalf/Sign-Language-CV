import cv2
from typing import Optional

from core.window.base_window import BaseWindow


class EmptyWindow(BaseWindow):
    """
    A window with no video source.

    Renders a blank (black) frame on every tick. Useful as a container for
    frame processors that generate their own visuals (e.g. :class:`ViewerProcessor`)
    or as a plain menu/dialog window when ``frame_processor`` is ``None``.
    """

    def __init__(self, **kwargs) -> None:
        """
        :param kwargs: Forwarded to :class:`~core.window.base_window.BaseWindow`.
        """
        super().__init__(**kwargs)

    def get_frame(self) -> None:
        """
        Return ``None`` so that the base class renders a black background.

        :return: Always ``None``.
        :rtype: None
        """
        return None
