import sys
import os

from PySide6.QtWidgets import QApplication

from core.processors.inference_processor import InferenceProcessor
from core.window.webcam_window import WebcamWindow
from core.utils import AppPaths


def main():
    AppPaths.load_env()

    MODEL_PATH = AppPaths.path(
        os.getenv("SIGN_TRANSLATE_MODEL", "models/sign_lstm.keras")
    )

    ENCODER_OUTPUT = AppPaths.path(
        os.getenv("LABEL_ENCODER_FILE", "models/sign_lstm_encoder.npy")
    )

    app = QApplication(sys.argv)

    processor = InferenceProcessor(MODEL_PATH, ENCODER_OUTPUT)

    window = WebcamWindow(
        width=1280,
        height=720,
        frame_processor=processor
    )

    def toggle():
        processor.toggle_inference()
        print(f"Inference toggled: {processor.inference_enabled}")

    def clear_text():
        processor.clear_text_buffer()
        print("Text buffer cleared")

    window.add_button(
        "toggle_inference",
        text="I",
        tooltip="Start / Stop inference prediction",
        action=toggle,
        color=WebcamWindow.COLORS["orange"],
        hover_color=WebcamWindow.COLORS["orange_hover"],
        shortcut="I",
        width=80
    )

    window.add_button(
        "clear_text",
        text="C",
        tooltip="Clear text buffer",
        action=clear_text,
        color=WebcamWindow.COLORS["red"],
        hover_color=WebcamWindow.COLORS["red_hover"],
        shortcut="C",
        width=80
    )

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()