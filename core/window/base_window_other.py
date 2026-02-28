from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt
import numpy as np

class BaseWindow(QWidget):

    DEFAULT_TIMER = 30

    def __init__(self, width=800, height=600):
        super().__init__()

        # -----------------------------
        # Window and Styles
        # -----------------------------

        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; }
            QLabel { border: 2px solid #444; border-radius: 8px; }
            QPushButton {
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 5px;
            }
            #actionButton {
                background-color: #0078ff;
            }
            #actionButton:hover {
                background-color: #0053ff;
            }
        """)

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        # Top Bar
        top_bar = QHBoxLayout()
        top_bar.addStretch()

        self.action_button = QPushButton("≡")
        self.action_button.setFixedSize(30, 30)
        self.action_button.setObjectName("actionButton")
        # Sin conectar a ninguna función (por ahora)
        top_bar.addWidget(self.action_button)

        main_layout.addLayout(top_bar)

        # Display Label
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        #self.display_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        main_layout.addWidget(self.display_label)

        self.setLayout(main_layout)

        self.frame_width = width
        self.frame_height = height

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(BaseWindow.DEFAULT_TIMER)

        # Hook opcional
        self.setup()

    # -----------------------------
    # Main label manage
    # -----------------------------

    def _update_frame(self):
        frame = self.get_frame()
        if frame is not None:
            self._show_frame(frame)
        else:
            self._show_frame(np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8))

    def _show_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.display_label.setPixmap(
            pixmap.scaled(
                self.frame_width,
                self.frame_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    # -----------------------------
    # Abstract methods
    # -----------------------------

    def get_frame(self):
        """
        It should return a frame RGB ndarray (h, w, ch)
        """
        raise NotImplementedError("Implement get_frame() in the child class")

    def setup(self):
        """
        Optional: initialize things like camera, image, etc ...
        """
        pass

    def cleanup(self):
        """
        Optional: used to release resources like camera, image, etc ...
        """
        pass

    # -----------------------------
    # Events
    # -----------------------------

    def resizeEvent(self, event):
        """Actualiza las dimensiones del label cuando la ventana se redimensiona."""
        super().resizeEvent(event)
        # El label ya ha sido redimensionado por el layout, obtenemos su tamaño actual
        self.frame_width = self.display_label.width()
        self.frame_height = self.display_label.height()

    def closeEvent(self, event):
        self.cleanup()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_C and event.modifiers() & Qt.ControlModifier:
            self.close()

if __name__ == "__main__":
    from video_window import VideoWindow
    from image_window import ImageWindow
    from webcam_window import WebcamWindow
    import sys

    app = QApplication(sys.argv)

    #window = WebcamWindow(0,width=1280, height=720)
    #window = ImageWindow("./imagen_prueba.png")
    window = VideoWindow("video_prueba.mp4", width=1280, height=720)

    #window.resize(800, 600)
    window.show()

    sys.exit(app.exec())