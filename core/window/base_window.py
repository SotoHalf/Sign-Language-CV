import os
import time
from typing import Callable, List, Optional

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
    QHBoxLayout, QSizePolicy, QInputDialog, QMessageBox
)
from PySide6.QtGui import QIcon, QImage, QPixmap, QPainter, QColor, QFont
from PySide6.QtCore import QSize, QTimer, Qt
import numpy as np

from core.processors.frame_processor import FrameProcessor


class BaseWindow(QWidget):
    """
    Base Qt widget for displaying a continuous stream of video frames.

    Provides a display label that refreshes on a timer, an optional frame
    processor injected via dependency injection, an overlay button panel,
    and lifecycle hooks (``setup`` / ``cleanup``) for subclasses.

    Subclasses must implement :meth:`get_frame` to supply frames.
    """

    # Default timer interval in ms — roughly 33 FPS (100ms/30) = 30 FPS
    DEFAULT_TIMER: int = 30

    COLORS = {
        "bg":           "#1e1e1e",
        "black":        "#000000",
        "white":        "#ffffff",
        "blue":         "#3a86ff",
        "blue_hover":   "#2f6fd6",
        "green":        "#52b788",
        "green_hover":  "#40916c",
        "orange":       "#f4a261",
        "orange_hover": "#f39344",
        "red":          "#e76f51",
        "red_hover":    "#e25b39",
    }

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        frame_processor: Optional[FrameProcessor] = None,
        paint_fps: bool = True
    ) -> None:
        """
        Initialize the window, apply the dark theme and start the frame timer.

        :param width: Initial window width in pixels.
        :type width: int
        :param height: Initial window height in pixels.
        :type height: int
        :param frame_processor: Optional processor whose ``process(frame)`` method
            is called on every frame before display.
        :type frame_processor: FrameProcessor, optional
        :param paint_fps: Whether to draw the FPS counter on the frame.
        :type paint_fps: bool
        """
        super().__init__()

        self._frame_processor: Optional[FrameProcessor] = frame_processor
        self._paint_fps: bool = paint_fps
        self._processing_frame: bool = False

        self.finished: bool = False
        self.finished_reason: str = ""

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.COLORS["bg"]};
                color: {self.COLORS["white"]};
            }}
            QLabel {{
                color: {self.COLORS["white"]};
            }}
            QListWidget {{
                background-color: #2a2a2a;
                color: {self.COLORS["white"]};
                border: 1px solid #444;
            }}
            QLineEdit {{
                background-color: #2a2a2a;
                color: {self.COLORS["white"]};
                border: 1px solid #444;
                padding: 4px;
            }}
            QInputDialog {{
                color: {self.COLORS["white"]};
            }}
            QMessageBox {{
                color: {self.COLORS["white"]};
            }}
            QPushButton {{
                color: {self.COLORS["white"]};
                border: none;
                padding: 5px 10px;
                border-radius: 5px;
            }}
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        self.action_button_dict: dict = {}
        # Callbacks registered by subclasses to reposition overlaid widgets on resize
        # in case we need to fix position for some additions to UI
        self.external_resizes: List[Callable] = []

        # Main display
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setMinimumSize(1, 1)
        main_layout.addWidget(self.display_label)

        self.setLayout(main_layout)

        self.frame_width: int = width
        self.frame_height: int = height
        self.resize(width, height)

        self.MARGIN_PANEL_X: int = 20
        self.MARGIN_PANEL_Y: int = -10

        self.setup_button_panel()

        #FPS
        self._last_time: float = time.perf_counter()
        self._fps: int = 0
        self._frame_count: int = 0

        # Timer
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(BaseWindow.DEFAULT_TIMER)

        self.overlay_text: str = ""

        # Hook for extended classes
        self.setup()

    # --------------------------------------------------
    # Dialog helpers
    # --------------------------------------------------

    def show_input_dialog(
        self, title: str, label: str, default_text: str = ""
    ) -> Optional[str]:
        """
        Show a modal text-input dialog and return the entered value.

        :param title: Dialog window title.
        :type title: str
        :param label: Prompt text shown above the input field.
        :type label: str
        :param default_text: Pre-filled text in the input field.
        :type default_text: str
        :return: Stripped input string, or ``None`` if cancelled or empty.
        :rtype: str, optional
        """
        text, ok = QInputDialog.getText(self, title, label, text=default_text)
        if ok and text:
            return text.strip()
        return None

    def show_dialog(self, title: str, label: str) -> None:
        """
        Show a modal warning message box.

        :param title: Dialog window title.
        :type title: str
        :param label: Message body text.
        :type label: str
        """
        QMessageBox.warning(self, title, label)

    # --------------------------------------------------
    # Button panel
    # --------------------------------------------------

    def add_button(
        self,
        name: str,
        text: str = "",
        action: Optional[Callable] = None,
        width: int = 60,
        height: int = 30,
        txt_size: int = 10,
        color: Optional[str] = None,
        hover_color: Optional[str] = None,
        tooltip: Optional[str] = None,
        checkable: bool = False,
        shortcut: Optional[str] = None,
        alignment: str = "right"
    ) -> QPushButton:
        """
        Create a styled button and add it to the overlay panel.

        :param name: Unique key used to look up the button later.
        :type name: str
        :param text: Button label text.
        :type text: str
        :param action: Callable invoked on click.
        :type action: callable, optional
        :param width: Button width in pixels.
        :type width: int
        :param height: Button height in pixels.
        :type height: int
        :param txt_size: Font size in points; bold if > 10.
        :type txt_size: int
        :param color: Background hex colour. Defaults to blue.
        :type color: str, optional
        :param hover_color: Hover-state hex colour. Defaults to blue_hover.
        :type hover_color: str, optional
        :param tooltip: Tooltip string shown on hover.
        :type tooltip: str, optional
        :param checkable: Whether the button is a toggle.
        :type checkable: bool
        :param shortcut: Keyboard shortcut string (e.g. ``"R"``).
        :type shortcut: str, optional
        :param alignment: ``"left"`` inserts at position 0; anything else appends.
        :type alignment: str
        :return: The created ``QPushButton``.
        :rtype: QPushButton
        :raises Exception: If a button with the same ``name`` already exists.
        """
        if name in self.action_button_dict:
            raise Exception(f"Button '{name}' already exists")

        btn = QPushButton(text)
        font = btn.font()
        font.setPointSize(txt_size)
        if txt_size > 10:
            font.setBold(True)
        btn.setFont(font)
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

    def setup_button_panel(self) -> None:
        """
        Create the transparent overlay panel that holds the action buttons.
        The panel floats above the display label and is repositioned on every resize.
        """
        self.button_panel = QWidget(self.display_label)
        self.button_panel.setStyleSheet("background-color: transparent;")
        self.button_panel.setAttribute(Qt.WA_AlwaysShowToolTips)
        self.button_panel_layout = QHBoxLayout(self.button_panel)
        self.button_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.button_panel_layout.setSpacing(5)
        self.button_panel.hide()

    def update_button_panel_position(self) -> None:
        """
        Reposition the button panel to the top-right corner of the displayed image.

        Called after every frame draw and on window resize to keep the panel
        aligned with the scaled pixmap rather than the full label area.
        """
        if not hasattr(self, 'button_panel'):
            return

        pixmap = self.display_label.pixmap()
        if pixmap is None or pixmap.isNull():
            self.button_panel.hide()
            return

        label_rect = self.display_label.rect()
        scaled_rect = pixmap.rect()
        scaled_rect.moveCenter(label_rect.center())

        panel_w = self.button_panel.sizeHint().width()
        panel_h = self.button_panel.sizeHint().height()
        x = scaled_rect.right() - panel_w - self.MARGIN_PANEL_X
        y = scaled_rect.top() - self.MARGIN_PANEL_Y

        self.button_panel.setGeometry(x, y, panel_w, panel_h)
        self.button_panel.show()

    # --------------------------------------------------
    # Frame loop
    # --------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_button_panel_position()
        for func in self.external_resizes:
            func()

    def _draw_fps(self, scaled_pixmap: QPixmap) -> None:
        """
        Paint the FPS counter onto the already-scaled pixmap using a QPainter.

        :param scaled_pixmap: The pixmap currently shown in the display label.
        :type scaled_pixmap: QPixmap
        """
        painter = QPainter(scaled_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.setPen(QColor(0, 255, 0))
        painter.drawText(10, 25, f"{self._fps} FPS")

        if self.overlay_text:
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 24, QFont.Bold))
            painter.drawText(self.frame_height // 2, self.frame_width // 2, self.overlay_text)

        painter.end()

    def _update_frame(self) -> None:
        """
        Timer slot: fetch a frame, apply the processor, and display it.

        Guarded by ``_processing_frame`` to skip ticks if the previous frame
        is still being processed (prevents queue build-up on slow hardware).
        """
        if self._processing_frame:
            return

        self._processing_frame = True
        try:
            frame = self.get_frame()

            self._frame_count += 1
            current_time = time.perf_counter()
            elapsed = current_time - self._last_time
            if elapsed >= 1.0:
                self._fps = round(self._frame_count / elapsed)
                self._frame_count = 0
                self._last_time = current_time

            if frame is None:
                frame = np.zeros(
                    (self.display_label.height(), self.display_label.width(), 3),
                    dtype=np.uint8
                )

            if self._frame_processor is not None and not self._frame_processor.finished:
                frame = self._frame_processor.process(frame)

            # Finished flag can be raised by either the window itself or its processor
            if self.finished or (self._frame_processor and self._frame_processor.finished):
                reason = self.finished_reason or self._frame_processor.finished_reason
                QMessageBox.information(self, "Finished", reason)
                self.timer.stop()
                self.close()

            self._show_frame(frame)
        finally:
            self._processing_frame = False

    def _show_frame(self, frame: np.ndarray) -> None:
        """
        Convert a NumPy RGB frame to a scaled QPixmap and display it.

        Resizes the window to match the frame dimensions the first time a new
        frame size is encountered (e.g. on camera switch).

        :param frame: RGB image of shape ``(H, W, 3)``.
        :type frame: np.ndarray
        """
        h, w, ch = frame.shape
        qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        if not hasattr(self, '_last_frame_size') or self._last_frame_size != (w, h):
            self._last_frame_size = (w, h)
            margins = self.layout().contentsMargins()
            self.resize(w + margins.left() + margins.right(),
                        h + margins.top() + margins.bottom())

        label_w = self.display_label.width()
        label_h = self.display_label.height()
        if label_w > 0 and label_h > 0:
            scaled = pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if self._paint_fps:
                self._draw_fps(scaled)
            self.display_label.setPixmap(scaled)
            self.update_button_panel_position()
            for func in self.external_resizes:
                func()

    # --------------------------------------------------
    # Subclass hooks
    # --------------------------------------------------

    def get_frame(self) -> np.ndarray:
        """
        Return the next RGB frame to display.

        Must be implemented by every subclass.

        :return: RGB image of shape ``(H, W, 3)``.
        :rtype: np.ndarray
        :raises NotImplementedError: Always — subclasses must override this.
        """
        raise NotImplementedError("Implement get_frame() in the subclass")

    def setup(self) -> None:
        """
        Optional post-construction hook called once at the end of ``__init__``.
        Subclasses override this to open cameras, load files, or add buttons.
        """
        pass

    def cleanup(self) -> None:
        """
        Optional teardown hook called just before the window closes.
        Subclasses override this to release cameras or file handles.
        """
        pass

    # --------------------------------------------------
    # Qt events
    # --------------------------------------------------

    def closeEvent(self, event) -> None:
        """
        Call :meth:`cleanup` before accepting the close event.

        :param event: The Qt close event.
        """
        self.cleanup()
        event.accept()

    def keyPressEvent(self, event) -> None:
        """
        Close the window on ``Escape`` or ``Ctrl+C``.

        :param event: The Qt key event.
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
    window = VideoWindow("video_prueba.mp4", width=1280, height=720)
    window.show()
    sys.exit(app.exec())
