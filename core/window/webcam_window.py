import cv2
from core.window.base_window import BaseWindow
from PySide6.QtWidgets import QInputDialog
from cv2_enumerate_cameras import enumerate_cameras


class WebcamWindow(BaseWindow):

    def __init__(self, **kwargs):
        self.cams = WebcamWindow.get_available_cameras()
        super().__init__(**kwargs)

    def setup(self):
       
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

    def select_camera(self):
        if not self.cams:
            self.show_dialog("Error", "No webcams detected")
            return

        names = [name for _, name in self.cams]

        selected_name, ok = QInputDialog.getItem(
            self,
            "Seleccionar cámara",
            "Webcams disponibles:",
            names,
            0,
            False
        )

        if ok:
            for cam_id, name in self.cams:
                if name == selected_name:
                    self.webcam_id = cam_id
                    self.init_camera(cam_id)
                    break

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

    @staticmethod
    def get_available_cameras():
        cameras = []
        for camera_info in enumerate_cameras():
            cameras.append((camera_info.index, camera_info.name))
        return cameras