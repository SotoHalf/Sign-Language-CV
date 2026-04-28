import os

from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QSizePolicy, QInputDialog, QMessageBox
from PySide6.QtGui import QIcon, QImage, QPixmap, QPainter, QColor, QFont
from PySide6.QtCore import QSize, QTimer, Qt
import numpy as np
import time

from core.processors.frame_processor import FrameProcessor

class BaseWindow(QWidget):

    """
    Base window for displaying video frames or images in a Qt application.

    Provides a reusable widget with a display label and an action button.
    Subclasses must implement `get_frame()` to supply the frame data.
    An optional frame processor can be injected to modify frames before display.
    """

    DEFAULT_TIMER = 30 # Default timer interval for update frame 1000ms/30 # FPS
    COLORS = {
        "bg": "#1e1e1e",
        "black": "#000000",
        "white": "#ffffff",

        "blue": "#3a86ff",
        "blue_hover": "#2f6fd6",

        "green": "#52b788",
        "green_hover": "#40916c",

        "orange": "#f4a261",
        "orange_hover": "#f39344",

        "red": "#e76f51",
        "red_hover": "#e25b39",
    }
    
    def __init__(self, width=800, height=600, frame_processor: FrameProcessor=None):
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

        self._frame_processor = frame_processor
        self._processing_frame = False

        # Window Styles
        self.setStyleSheet(f"""
            QWidget {{ background-color: {self.COLORS["bg"]}; }}
            QPushButton {{
                color: {self.COLORS["white"]};
                border: none;
                padding: 5px 10px;
                border-radius: 5px;
            }}
        """)

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        self.action_button_dict = {}
        # in case we need to fix position for some additions to UI
        self.external_resizes = [] 

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

        self.MARGIN_PANEL_X = 20
        self.MARGIN_PANEL_Y = - 10

        self.setup_button_panel()

        #FPS
        self._last_time = time.perf_counter()
        self._fps = 0
        self._frame_count = 0

        # Timer
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(BaseWindow.DEFAULT_TIMER)

        self.overlay_text = ""

        # Hook for extended classes
        self.setup()

    # -----------------------------
    # Append options
    # -----------------------------
    
    def show_input_dialog(self, title: str, label: str, default_text: str = "") -> str | None:
        text, ok = QInputDialog.getText(self, title, label, text=default_text)
        if ok and text:
            return text.strip()
        return None

    def show_dialog(self, title: str, label: str) -> str | None:
        QMessageBox.warning(self, title, label)
        
    def add_button(
        self,
        name,
        text="",
        action=None,
        width=60,
        height=30,
        color=None,
        hover_color=None,
        tooltip=None,
        checkable=False,
        shortcut=None,
        alignment="right"
    ):
        if name in self.action_button_dict:
            raise Exception(f"Button {name} already exists")

        btn = QPushButton(text)
        btn.setFixedSize(width, height)
        btn.setCheckable(checkable)

        if tooltip:
            btn.setToolTip(tooltip)
    
        if action:
            if checkable:
                btn.clicked.connect(lambda checked: action(checked))
            else:
                btn.clicked.connect(lambda: action())

        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color or self.COLORS["blue"]};
                color: {self.COLORS["white"]};
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {hover_color or self.COLORS["blue_hover"]};
            }}
            QToolTip {{
                background-color: {self.COLORS["black"]};
                color: {self.COLORS["white"]};
                border: 1px solid {self.COLORS["white"]};
            }}
        """)

        layout = self.button_panel_layout

        if alignment == "left":
            layout.insertWidget(0, btn)
        else:
            layout.addWidget(btn)

        if shortcut:
            btn.setShortcut(shortcut)

        self.action_button_dict[name] = btn

        return btn
    
    # -----------------------------
    # Main label manage
    # -----------------------------

    def setup_button_panel(self):
        """Create the panel that will contain the buttons"""
        self.button_panel = QWidget(self.display_label)
        self.button_panel.setStyleSheet("background-color: transparent;")
        self.button_panel.setAttribute(Qt.WA_AlwaysShowToolTips)
        self.button_panel_layout = QHBoxLayout(self.button_panel)
        self.button_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.button_panel_layout.setSpacing(5)
        self.button_panel.hide()

    def update_button_panel_position(self):
        """Set the button pannel into the top right image"""
        if not hasattr(self, 'button_panel'):
            return
        
        pixmap = self.display_label.pixmap()
        if pixmap is None or pixmap.isNull():
            self.button_panel.hide()
            return
        
        label_rect = self.display_label.rect()
        pixmap_rect = pixmap.rect()
    
        scaled_rect = pixmap_rect
        scaled_rect.moveCenter(label_rect.center())
        
        # Set the buttons panel at the right side
        panel_width = self.button_panel.sizeHint().width()
        panel_height = self.button_panel.sizeHint().height()
        x = scaled_rect.right() - panel_width - self.MARGIN_PANEL_X
        y = scaled_rect.top() - self.MARGIN_PANEL_Y
        
        self.button_panel.setGeometry(x, y, panel_width, panel_height)
        self.button_panel.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_button_panel_position()
        for _func in self.external_resizes:
            _func()

    def _draw_fps(self, scaled_pixmap):
        # Draw FPS overlay
        painter = QPainter(scaled_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        font = QFont("Arial", 14, QFont.Bold)
        painter.setFont(font)

        painter.setPen(QColor(0, 255, 0))
        painter.drawText(10, 25, f"{self._fps} FPS")

        if hasattr(self, "overlay_text") and self.overlay_text:
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 24, QFont.Bold))
            painter.drawText(self.frame_height//2, self.frame_width//2, self.overlay_text)

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

            if frame is None:
                frame = np.zeros((self.display_label.height(), self.display_label.width(), 3), dtype=np.uint8)
            
            if self._frame_processor is not None:
                frame = self._frame_processor.process(frame)
            self._show_frame(frame)
            
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

        if not hasattr(self, '_last_frame_size') or self._last_frame_size != (w, h):
            self._last_frame_size = (w, h)
            margins = self.layout().contentsMargins()
            new_width = w + margins.left() + margins.right()
            new_height = h + margins.top() + margins.bottom()
            self.resize(new_width, new_height)

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
            self._draw_fps(scaled_pixmap)
            self.display_label.setPixmap(scaled_pixmap)
            self.update_button_panel_position() 
            for _func in self.external_resizes:
                _func()

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