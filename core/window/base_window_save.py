from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt
import numpy as np

class BaseWindow(QWidget):

    DEFAULT_TIMER = 30

    CURSOR_MARGIN = 50  #Used to determine the selection area for size

    def __init__(self, width=800, height=600):
        super().__init__()

        self.resizing = False
        self.frame_width = width
        self.frame_height = height

        # -----------------------------
        # Window and Styles
        # -----------------------------
        #self.setWindowFlags(Qt.FramelessWindowHint)
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; }
            QLabel { border: 2px solid #444; border-radius: 8px; }
            QPushButton {
                background-color: #ff5555;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 5px;
            }
                          
            #resizeButton { background-color: #5555ff; }
            #resizeButton:hover { background-color: #2222ff; }
                           
            #closeButton { background-color: #ff5555; }
            #closeButton:hover { background-color: #ff2222; }
        """)

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        # Top Bar
        top_bar = QHBoxLayout()
        top_bar.addStretch()

        self.resize_button = QPushButton("⛶")
        self.resize_button.setFixedSize(30, 30)
        self.resize_button.clicked.connect(self._toggle_resize)
        self.resize_button.setObjectName("resizeButton") 
        top_bar.addWidget(self.resize_button)

        self.close_button = QPushButton("✕")
        self.close_button.setFixedSize(30, 30)
        self.close_button.clicked.connect(self.close)
        self.close_button.setObjectName("closeButton")
        top_bar.addWidget(self.close_button)

        main_layout.addLayout(top_bar)

        # Display Label
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setFixedSize(self.frame_width, self.frame_height)
        self.display_label.setAttribute(Qt.WA_TransparentForMouseEvents) # no interfer with mouse
        main_layout.addWidget(self.display_label)

        self.setLayout(main_layout)

        self.adjustSize()

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(BaseWindow.DEFAULT_TIMER)

        self.drag_pos = None

        # Hook optional
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
        It should return an frame RBG ndarray (h, w, ch)
        """
        raise NotImplementedError("ImplementDebes implementar get_frame() en la clase hija")

    def setup(self):
        """
        Optional: initialize thigs like camera, image, etc ...
        """
        pass

    def cleanup(self):
        """
        Optional: Used to liberate things like camera, image, etc ...
        """
        pass

    # -----------------------------
    # Other features
    # -----------------------------

    def _toggle_resize(self):
        screen = self.screen()
        screen_geom = screen.availableGeometry()  # util area without taskbar
        max_width = screen_geom.width()
        max_height = screen_geom.height()

        # toogle between actual size and maximum size
        if (self.frame_width, self.frame_height) != (max_width, max_height):
            # save old value
            self.prev_width = self.frame_width
            self.prev_height = self.frame_height

            self.frame_width = max_width
            self.frame_height = max_height
        else:
            # recover old value
            self.frame_width = getattr(self, "prev_width", self.width)
            self.frame_height = getattr(self, "prev_height", self.height)

        self.display_label.setFixedSize(self.frame_width, self.frame_height)
        self.adjustSize()

        # center window
        new_x = screen_geom.x() + (screen_geom.width() - self.width()) // 2
        new_y = screen_geom.y() + (screen_geom.height() - self.height()) // 2
        self.move(new_x, new_y)

    # -----------------------------
    # Events
    # -----------------------------
    def closeEvent(self, event):
        self.cleanup()
        event.accept()

    def mouseReleaseEvent(self, event):
        self.resizing = False
        self.drag_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            margin = BaseWindow.CURSOR_MARGIN
            pos = event.position().toPoint()

            width = self.width()
            height = self.height()

            in_corner = (
                pos.x() >= width - margin and
                pos.y() >= height - margin
            )

            if in_corner:
                # Iniciar resize
                self.resizing = True
                self.resize_start_pos = event.globalPosition().toPoint()
                self.start_width = self.width()
                self.start_height = self.height()
            else:
                # Iniciar movimiento de ventana
                self.drag_pos = event.globalPosition().toPoint()

    """
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            margin = BaseWindow.CURSOR_MARGIN
            rect = self.rect()
            # coordenadas locales para detectar esquina
            pos = event.position()
            if pos.x() >= rect.width() - margin and pos.y() >= rect.height() - margin:
                # resize
                self.resizing = True
                self.resize_start_pos = event.globalPosition().toPoint()
                self.start_width = self.width()
                self.start_height = self.height()
            else:
                # move window (start point)
                self.drag_pos = event.globalPosition().toPoint()
    """

    def mouseMoveEvent(self, event):
        margin = BaseWindow.CURSOR_MARGIN
        pos = event.position().toPoint()

        width = self.width()
        height = self.height()

        in_corner = (
            pos.x() >= width - margin and
            pos.y() >= height - margin
        )

        if in_corner:
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if self.resizing:
            delta = event.globalPosition().toPoint() - self.resize_start_pos
            new_width = max(200, self.start_width + delta.x())
            new_height = max(150, self.start_height + delta.y())

            self.frame_width = new_width
            self.frame_height = new_height
            self.display_label.setFixedSize(self.frame_width, self.frame_height)
            self.adjustSize()
            return

        if event.buttons() == Qt.LeftButton and self.drag_pos:
            delta = event.globalPosition().toPoint() - self.drag_pos
            self.move(self.pos() + delta)
            self.drag_pos = event.globalPosition().toPoint()

    """
    def mouseMoveEvent(self, event):
        margin = BaseWindow.CURSOR_MARGIN
        pos = event.position().toPoint()
        rect = self.rect()

        in_corner = (
            pos.x() >= rect.width() - margin and
            pos.y() >= rect.height() - margin
        )

        if in_corner:
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if self.resizing:
            delta = event.globalPosition().toPoint() - self.resize_start_pos
            new_width = max(200, self.start_width + delta.x())
            new_height = max(150, self.start_height + delta.y())

            self.frame_width = new_width
            self.frame_height = new_height

            self.display_label.setFixedSize(self.frame_width, self.frame_height)
            self.adjustSize()
            return

        if event.buttons() == Qt.LeftButton and self.drag_pos:
            delta = event.globalPosition().toPoint() - self.drag_pos
            self.move(self.pos() + delta)
            self.drag_pos = event.globalPosition().toPoint()

    """

    """
    def mouseMoveEvent(self, event):
        margin = BaseWindow.CURSOR_MARGIN
        pos = event.position()

        # change cursor if we are in the corner
        rect = self.rect()
        if pos.x() >= rect.width() - margin and pos.y() >= rect.height() - margin:
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        # resize if we are moving the corner
        if getattr(self, "resizing", False):
            delta = event.globalPosition().toPoint() - self.resize_start_pos
            new_width = max(200, self.start_width + delta.x())
            new_height = max(150, self.start_height + delta.y())
            self.frame_width = new_width
            self.frame_height = new_height
            self.display_label.setFixedSize(self.frame_width, self.frame_height)
            self.adjustSize()
            return

        # move window if we are moving outside the corner
        if event.buttons() == Qt.LeftButton and self.drag_pos:
            delta = event.globalPosition().toPoint() - self.drag_pos
            self.move(self.pos() + delta)
            self.drag_pos = event.globalPosition().toPoint()
    """

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