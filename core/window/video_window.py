import cv2
from typing import Optional

from core.window.base_window import BaseWindow


class VideoWindow(BaseWindow):
    """
    Window that reads frames from a video file.

    On construction the video is opened and the frame timer is synced to the
    video's native FPS. The video path can be swapped at runtime via
    :meth:`set_video_path`.
    """

    def __init__(self, video_path: str, **kwargs) -> None:
        """
        :param video_path: Absolute or relative path to the video file.
        :type video_path: str
        :param kwargs: Forwarded to :class:`~core.window.base_window.BaseWindow`.
        :raises FileNotFoundError: If the file cannot be opened by OpenCV.
        """
        self.video_path: str = video_path
        super().__init__(**kwargs)

    def setup(self) -> None:
        """Open the video file and sync the timer to the video FPS."""
        self.open_video()
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.timer.setInterval(int(1000 / fps))

    def set_video_path(self, video_path: str) -> None:
        """
        Switch to a different video file.

        No-op if ``video_path`` is empty or identical to the current one.

        :param video_path: Path to the new video file.
        :type video_path: str
        """
        if not video_path or video_path == self.video_path:
            return
        self.video_path = video_path
        self.open_video()

    def open_video(self) -> None:
        """
        Open ``self.video_path`` with OpenCV.

        :raises FileNotFoundError: If the video cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"The video could not be opened: {self.video_path}")

    def get_frame(self) -> Optional[object]:
        """
        Read the next frame from the video and convert it to RGB.

        :return: RGB NumPy array of shape ``(H, W, 3)``, or ``None`` when the video ends.
        :rtype: np.ndarray or None
        """
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def is_finished(self) -> bool:
        """
        Check whether playback has reached the last frame.

        :return: ``True`` if the capture is closed or all frames have been read.
        :rtype: bool
        """
        if not hasattr(self, "cap") or not self.cap.isOpened():
            return True
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def cleanup(self) -> None:
        """Release the video capture when the window closes."""
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        self.timer.start(BaseWindow.DEFAULT_TIMER)

    def get_fps(self) -> float:
        """
        Return the video's native frame rate.

        :return: FPS reported by the capture, or ``30`` as a safe fallback.
        :rtype: float
        """
        if not hasattr(self, "cap") or not self.cap.isOpened():
            return 30.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    def seek_seconds(self, seconds: float) -> None:
        """
        Seek to the given position in seconds.

        :param seconds: Target position from the start of the video.
        :type seconds: float
        """
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)

    def seek_frame(self, frame_number: int) -> None:
        """
        Seek to a specific frame index.

        :param frame_number: Zero-based frame index to jump to.
        :type frame_number: int
        """
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_current_time(self) -> float:
        """
        Return the current playback position in seconds.

        :return: Position in seconds, or ``0.0`` if the capture is closed.
        :rtype: float
        """
        if hasattr(self, "cap") and self.cap.isOpened():
            return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return 0.0
