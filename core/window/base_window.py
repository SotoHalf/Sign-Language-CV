from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QSizePolicy
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
            /*QLabel { border: 2px solid #444; border-radius: 8px; }*/
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

        # test button
        self.action_button = QPushButton("≡")
        self.action_button.setFixedSize(30, 30)
        self.action_button.setObjectName("actionButton")
        top_bar.addWidget(self.action_button)

        main_layout.addLayout(top_bar)

        # Display Label
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setMinimumSize(1, 1)  # Permite que se encoja si es necesario
        # self.display_label.setAttribute(Qt.WA_TransparentForMouseEvents)  #avoid events
        main_layout.addWidget(self.display_label)

        self.setLayout(main_layout)

        self.frame_width = width
        self.frame_height = height
        self.resize(width, height)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(BaseWindow.DEFAULT_TIMER)

        # Hook for extended classes
        self.setup()

    # -----------------------------
    # Main label manage
    # -----------------------------

    def _update_frame(self):
        frame = self.get_frame()
        if frame is not None:
            self._show_frame(frame)
        else:
            # default frame
            default_frame = np.zeros((self.display_label.height(), self.display_label.width(), 3), dtype=np.uint8)
            self._show_frame(default_frame)

    def _show_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        #set the actual window size
        label_width = self.display_label.width()
        label_height = self.display_label.height()

        if label_width > 0 and label_height > 0:
            scaled_pixmap = pixmap.scaled(
                label_width,
                label_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.display_label.setPixmap(scaled_pixmap)

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