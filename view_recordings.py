#!/usr/bin/env python3
"""
view_recordings.py — Browse and manage recorded gesture sequences.

Opens the dataset folder and renders each sequence as an animated hand
skeleton for visual inspection. Supports deleting bad recordings directly
from the viewer.

Keyboard shortcuts:
    Space  — Play / Pause.
    Left   — Previous recording.
    Right  — Next recording.
    S      — Open sign selector dialog.

Usage:
    python view_recordings.py
    OR imported and called as view_recordings.run(app)
"""

import sys
import os
from typing import List, Optional

from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QListWidget,
    QPushButton, QLabel, QMessageBox
)

from core.window.empty_window import EmptyWindow
from core.processors.viewer_processor import ViewerProcessor
from core.utils import AppPaths

AppPaths.load_env()


class SignSelectorDialog(QDialog):
    """
    Modal dialog for jumping directly to a specific gesture label in the viewer.
    Pre-selects the label that is currently displayed.
    """

    def __init__(
        self,
        labels: List[str],
        current_label: Optional[str] = None,
        parent=None
    ) -> None:
        """
        :param labels: Full list of available gesture labels.
        :type labels: list[str]
        :param current_label: The label currently displayed in the viewer
            (used to pre-select the list item).
        :type current_label: str, optional
        :param parent: Optional parent widget.
        :type parent: QWidget, optional
        """
        super().__init__(parent)
        self.setWindowTitle("Select Sign")
        self.setFixedSize(260, 360)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Selecciona un símbolo:"))

        self.list_widget = QListWidget()
        self.list_widget.addItems(labels)
        layout.addWidget(self.list_widget)

        # Pre-select the active label for convenience
        if current_label in labels:
            self.list_widget.setCurrentRow(labels.index(current_label))
        elif labels:
            self.list_widget.setCurrentRow(0)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)

    def selected_label(self) -> Optional[str]:
        """
        Return the label chosen by the user.

        :return: Selected label string, or ``None`` if nothing is selected.
        :rtype: str, optional
        """
        item = self.list_widget.currentItem()
        return item.text() if item else None


def run(app: QApplication) -> Optional[EmptyWindow]:
    """
    Build the viewer window with all navigation and control buttons.

    :param app: The running Qt application instance.
    :type app: QApplication
    :return: The viewer window, or ``None`` if the dataset is empty.
    :rtype: EmptyWindow or None
    """
    DATA_PATH = AppPaths.path(os.getenv("DATA_PATH", "data/processed/"))

    try:
        processor = ViewerProcessor(DATA_PATH, fps=30)
    except RuntimeError as e:
        QMessageBox.critical(None, "Error loading data", str(e))
        return None

    window = EmptyWindow(width=800, height=600, frame_processor=processor)

    # --- Playback controls ---
    window.add_button("play_pause", text="⏸️", action=processor.toggle_play,
                      tooltip="Play / Pause",
                      color=EmptyWindow.COLORS['orange'],
                      hover_color=EmptyWindow.COLORS['orange_hover'],
                      shortcut="Space", alignment="right", width=50)

    window.add_button("reset", text="⟳", action=processor.reset,
                      tooltip="Restart playback",
                      alignment="right", width=50)

    def select_sign() -> None:
        """Open the sign-selector dialog and jump to the chosen label."""
        selector = SignSelectorDialog(
            labels=processor.labels,
            current_label=processor._current_label(),
            parent=window
        )
        if selector.exec():
            selected = selector.selected_label()
            if selected:
                processor.go_to_label(selected)

    window.add_button("select_sign", text="Symbols", action=select_sign,
                      tooltip="View all symbols and choose one",
                      color=EmptyWindow.COLORS['blue'],
                      hover_color=EmptyWindow.COLORS['blue_hover'],
                      shortcut="S", alignment="right", width=90)

    # --- Label navigation ---
    window.add_button("prev_sign", text="<- Sign", action=processor.prev_label,
                      tooltip="Previous sign", alignment="right", width=70)

    window.add_button("next_sign", text="Sign ->", action=processor.next_label,
                      tooltip="Next sign", alignment="right", width=70)

    # --- Recording navigation ---
    window.add_button("prev_recording", text="<- Rec", action=processor.prev_record,
                      tooltip="Previous recording",
                      shortcut="Left", alignment="right", width=70)

    window.add_button("next_recording", text="Rec ->", action=processor.next_record,
                      tooltip="Next recording",
                      shortcut="Right", alignment="right", width=70)

    # --- Destructive action ---
    window.add_button("delete_record", text="Delete",
                      action=processor.delete_current_record,
                      tooltip="Delete current recording",
                      color=EmptyWindow.COLORS['red'],
                      hover_color=EmptyWindow.COLORS['red'],
                      alignment="right", width=70)

    window.show()
    return window


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = run(app)
    sys.exit(app.exec())
