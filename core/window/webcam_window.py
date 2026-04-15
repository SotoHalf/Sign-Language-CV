import cv2
from core.window.base_window import BaseWindow
from PySide6.QtWidgets import QInputDialog


class WebcamWindow(BaseWindow):

    def __init__(self, webcam_id, **kwargs):
        self.webcam_id = webcam_id
        super().__init__(**kwargs)

    def setup(self):
        self.init_camera(self.webcam_id)

        self.add_button(
            "select_cam",
            text="CAM",
            color=BaseWindow.COLORS["green"],
            hover_color=BaseWindow.COLORS["green_hover"],
            action=self.select_camera,
            tooltip="Select webcam",
            alignment="left"
        )

    def select_camera(self):
        cams = WebcamWindow.get_available_cameras()

        if not cams:
            self.show_dialog("Error", "No webcams detected")
            return

        cam_str = [str(c) for c in cams]

        selected, ok = QInputDialog.getItem(
            self,
            "Seleccionar cámara",
            "Webcams disponibles:",
            cam_str,
            0,
            False
        )

        if ok:
            cam_id = int(selected)
            self.webcam_id = cam_id
            self.init_camera(cam_id)

    def init_camera(self, cam_id):
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

    def get_available_cameras(max_tested=5):
        available = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                available.append(i)
                cap.release()
        return available