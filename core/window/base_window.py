from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QSizePolicy
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PySide6.QtCore import QTimer, Qt
import numpy as np
import time

class BaseWindow(QWidget):

    """
    Base window for displaying video frames or images in a Qt application.

    Provides a reusable widget with a display label and an action button.
    Subclasses must implement `get_frame()` to supply the frame data.
    An optional frame processor can be injected to modify frames before display.
    """

    DEFAULT_TIMER = 30 # Default timer interval for update frame 1000ms/30 # FPS

    def __init__(self, width=800, height=600, frame_processor=None):
        """
        Initialize the BaseWindow.

        :param width: Initial width of the window, defaults to 800
        :type width: int
        :param height: Initial height of the window, defaults to 600
        :type height: int
        :param frame_processor: Optional object with a `process(frame)` method,
                                defaults to None
        :type frame_processor: object, optional
        """
        super().__init__()

        self.frame_processor = frame_processor
        self._processing_frame = False

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
        self.display_label.setMinimumSize(1, 1)
        # self.display_label.setAttribute(Qt.WA_TransparentForMouseEvents)  #avoid events
        main_layout.addWidget(self.display_label)

        self.setLayout(main_layout)

        self.frame_width = width
        self.frame_height = height
        self.resize(width, height)

        #FPS
        self._last_time = time.perf_counter()
        self._fps = 0
        self._frame_count = 0

        # Timer
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(BaseWindow.DEFAULT_TIMER)

        # Hook for extended classes
        self.setup()

    # -----------------------------
    # Main label manage
    # -----------------------------

    def draw_fps(self, scaled_pixmap):
        # Draw FPS overlay
        painter = QPainter(scaled_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        font = QFont("Arial", 14, QFont.Bold)
        painter.setFont(font)

        painter.setPen(QColor(0, 255, 0))
        painter.drawText(10, 25, f"{self._fps} FPS")

        painter.end()

    
    def _update_frame(self):
        '''
        Internal slot called by the timer.
        Retrieves a frame via `get_frame()`, applies the processor if available,
        and displays it. If no frame is returned, shows a black frame.
        '''
        # in case one frame is being processed skip one tick
        if self._processing_frame:
            return

        self._processing_frame = True
        try:
            frame = self.get_frame()

            # FPS CALCULATION
            self._frame_count += 1
            current_time = time.perf_counter()
            elapsed = current_time - self._last_time

            if elapsed >= 1.0:
                self._fps = round(self._frame_count / elapsed)
                self._frame_count = 0
                self._last_time = current_time

            if frame is not None:
                if self.frame_processor is not None:
                    frame = self.frame_processor.process(frame)
                self._show_frame(frame)
            else:
                # default frame
                default_frame = np.zeros((self.display_label.height(), self.display_label.width(), 3), dtype=np.uint8)
                self._show_frame(default_frame)
        finally:
              self._processing_frame = False
        
    def _show_frame(self, frame):
        """
        Convert a numpy RGB frame to QPixmap and display it on the label,
        scaled to fit while preserving aspect ratio.

        :param frame: RGB image as a numpy array of shape (height, width, 3)
        :type frame: np.ndarray
        """
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
            self.draw_fps(scaled_pixmap)
            self.display_label.setPixmap(scaled_pixmap)

    # -----------------------------
    # Abstract methods
    # -----------------------------

    def get_frame(self):
        """
        Retrieve the current frame to be displayed.

        Must be implemented by subclasses.

        :return: RGB frame as a numpy array of shape (height, width, 3)
        :rtype: np.ndarray
        :raises NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Implement get_frame() in the child class")

    def setup(self):
        """
        Optional initialization hook.
        Called once after the widget is constructed.
        Subclasses can override to set up resources like cameras or files.
        """
        pass

    def cleanup(self):
        """
        Optional cleanup hook.
        Called when the window is closed. Subclasses should release resources here.
        """
        pass

    # -----------------------------
    # Events
    # -----------------------------

    def closeEvent(self, event):
        """
        Handle the window close event.
        Calls `cleanup()` before accepting the close event.

        :param event: The close event
        :type event: QCloseEvent
        """
        self.cleanup()
        event.accept()

    def keyPressEvent(self, event):
        """
        Handle key press events.
        Closes the window on Escape key or Ctrl+C.

        :param event: The key event
        :type event: QKeyEvent
        """
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