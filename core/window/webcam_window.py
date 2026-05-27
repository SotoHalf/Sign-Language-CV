import cv2
from typing import List, Optional, Tuple

from core.window.base_window import BaseWindow
from PySide6.QtWidgets import QInputDialog
from cv2_enumerate_cameras import enumerate_cameras


class WebcamWindow(BaseWindow):
    """
    Window that captures live frames from a webcam.

    On startup, the first available camera is opened automatically.
    A camera-selector button lets the user switch between devices at runtime.
    """

    def __init__(self, **kwargs) -> None:
        """
        Enumerate available cameras before calling the base constructor.

        Camera enumeration must happen here (before ``super().__init__``) because
        ``__init__`` calls ``setup()``, which needs the camera list ready.

        :param kwargs: Forwarded to :class:`~core.window.base_window.BaseWindow`.
        """
        self.cams: List[Tuple[int, str]] = WebcamWindow.get_available_cameras()
        super().__init__(**kwargs)

    def setup(self) -> None:
        """
        Open the first detected camera and add the camera-selector button.
        Shows an error dialog if no cameras are found.
        """
        if not self.cams:
            self.show_dialog("Error", "No webcams detected")
            return

        cam_id, _ = self.cams[0]
        self.init_camera(cam_id)

        self.add_button(
            "select_cam",
            text="CAM",
            color=BaseWindow.COLORS["green"],
            hover_color=BaseWindow.COLORS["green_hover"],
            action=self.select_camera,
            tooltip="Select webcam",
            alignment="left"
        )

    def select_camera(self) -> None:
        """
        Show a dialog listing all available cameras and open the selected one.
        Shows an error dialog if no cameras are detected.
        """
        if not self.cams:
            self.show_dialog("Error", "No webcams detected")
            return

        names = [name for _, name in self.cams]
        selected_name, ok = QInputDialog.getItem(
            self, "Seleccionar cámara", "Webcams disponibles:", names, 0, False
        )

        if ok:
            for cam_id, name in self.cams:
                if name == selected_name:
                    self.webcam_id = cam_id
                    self.init_camera(cam_id)
                    break

    def init_camera(self, cam_id: int) -> None:
        """
        Open the camera with the given device index.

        Releases any previously open capture first, then syncs the
        frame timer to the camera's native FPS.

        :param cam_id: OpenCV device index for the camera to open.
        :type cam_id: int
        """
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

        self.cap = cv2.VideoCapture(cam_id)

        if not self.cap.isOpened():
            self.cap = None
            self.show_dialog("Error", f"Could not open camera {cam_id}")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.timer.setInterval(int(1000 / fps))

    def get_frame(self) -> Optional[object]:
        """
        Read the next frame from the webcam and convert it to RGB.

        :return: RGB NumPy array of shape ``(H, W, 3)``, or ``None`` on read failure.
        :rtype: np.ndarray or None
        """
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def cleanup(self) -> None:
        """Release the camera capture when the window closes."""
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        self.timer.start(BaseWindow.DEFAULT_TIMER)

    @staticmethod
    def get_available_cameras() -> List[Tuple[int, str]]:
        """
        Enumerate all cameras that can be successfully opened by OpenCV.

        :return: List of ``(device_index, display_name)`` tuples.
        :rtype: list[tuple[int, str]]
        """
        cameras: List[Tuple[int, str]] = []
        for camera_info in enumerate_cameras():
            cap = cv2.VideoCapture(camera_info.index)
            if cap.isOpened():
                cameras.append((camera_info.index, camera_info.name))
        return cameras
