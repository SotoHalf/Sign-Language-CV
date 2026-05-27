import os
import glob
import time
import cv2 as cv
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from core.processors.frame_processor import FrameProcessor
from core.utils import AppPaths
from core.hand_tracker import HAND_CONNECTIONS

AppPaths.load_env()


class ViewerProcessor(FrameProcessor):
    """
    Replays recorded landmark sequences on a blank canvas for visual inspection.

    Renders the 21-point hand skeleton from pre-processed CSV files.
    Supports navigating between labels, individual recordings, and frames.
    """

    def __init__(
        self,
        data_dir: str,
        frame_size: Tuple[int, int] = (640, 480),
        fps: int = 30
    ) -> None:
        """
        Load all CSV records from the dataset folder and prepare for playback.

        :param data_dir: Root directory containing one subfolder per gesture label.
        :type data_dir: str
        :param frame_size: Output canvas size as ``(width, height)``.
        :type frame_size: tuple[int, int]
        :param fps: Target playback frames per second.
        :type fps: int
        :raises RuntimeError: If no CSV records are found in ``data_dir``.
        """
        super().__init__()
        self.data_dir: str = data_dir
        self.frame_w, self.frame_h = frame_size
        self.fps: int = fps
        self.delay_ms: int = int(1000 / fps) # 33ms para 30fps
        self.size_factor: float = 0.60   # fraction of canvas size used for the hand

        self.total_landmarks: int = int(os.getenv("HAND_TOTAL_LANDMARKS", 21))

        self.records_by_label: Dict[str, List[pd.DataFrame]] = {}
        self.labels: List[str] = []
        self._load_data()

        if not self.labels:
            raise RuntimeError(f"There are no records found in {self.data_dir}")

        self.current_label_idx: int = 0
        self.current_record_idx: int = 0
        self.current_frame_idx: int = 0

        self.is_playing: bool = True
        self.last_update_time: float = 0.0

        # Stable scale/center reference computed once per record (avoids per-frame jitter)
        self._display_ref_center: Optional[np.ndarray] = None
        self._display_ref_scale: Optional[float] = None
        self._init_display_reference()

    # --------------------------------------------------
    # Data loading
    # --------------------------------------------------

    def _load_data(self) -> None:
        """
        Scan ``data_dir`` for label subfolders and load all ``.csv`` files.

        Stores the file path as a DataFrame attribute (``df.attrs["file_path"]``)
        so it can be used during deletion without a separate lookup.

        Expected layout::

            data_dir/
                label1/
                    <uid>.csv
                label2/
                    <uid>.csv
        """
        for label in os.listdir(self.data_dir):
            label_path = os.path.join(self.data_dir, label)
            if not os.path.isdir(label_path):
                continue

            files = sorted(glob.glob(os.path.join(label_path, "*.csv")))
            if not files:
                continue

            records = []
            for f in files:
                df = pd.read_csv(f)
                df.attrs["file_path"] = f
                records.append(df)

            self.records_by_label[label] = records
            self.labels.append(label)

        # Sort by length first so short labels (a, b, c) appear before compound ones
        self.labels.sort(key=lambda lbl: (len(lbl), lbl.lower()))

    def delete_current_record(self) -> None:
        """
        Delete the currently displayed recording from disk and from memory.

        Handles four edge cases in order:
        1. Records still remain for the same label → stay on label.
        2. Label is now empty → remove it.
        3. No more labels at all → set ``finished = True``.
        4. Otherwise → wrap label index to stay in bounds.
        """
        label = self._current_label()
        records = self._current_records()

        if not records:
            return

        df = records[self.current_record_idx]
        file_path: Optional[str] = df.attrs.get("file_path")

        # 1. delete file from disk
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

        # 2. remove from memory
        del records[self.current_record_idx]

        # case 1: still records in label
        if records:
            self.current_record_idx = min(self.current_record_idx, len(records) - 1)
            self.current_frame_idx = 0
            self._init_display_reference()
            return

        # case 2: label is empty remove label
        print(f"Label '{label}' is empty, removing it")
        del self.records_by_label[label]
        self.labels.remove(label)

        # case 3: no more labels end program
        if not self.labels:
            self.finished = True
            self.is_playing = False
            self.finished_reason = (
                "All recordings have been deleted.\n"
                "There are no more signs available to display."
            )
            return

        # case 4: move to next label safely
        self.current_label_idx %= len(self.labels)
        self.current_record_idx = 0
        self.current_frame_idx = 0
        self._init_display_reference()

    # --------------------------------------------------
    # Private getters
    # --------------------------------------------------

    def _current_label(self) -> str:
        return self.labels[self.current_label_idx]

    def _current_records(self) -> List[pd.DataFrame]:
        return self.records_by_label[self._current_label()]

    def _current_df(self) -> pd.DataFrame:
        return self._current_records()[self.current_record_idx]

    def _current_filename(self) -> str:
        return f"{self._current_label()}[{self.current_record_idx}]"

    # --------------------------------------------------
    # Navigation
    # --------------------------------------------------

    def go_to_label(self, label: str) -> None:
        """
        Jump directly to the first record of the specified label.

        :param label: Target label string. No-op if the label does not exist.
        :type label: str
        """
        if label not in self.labels:
            return
        self.current_label_idx = self.labels.index(label)
        self.current_record_idx = 0
        self.current_frame_idx = 0
        self.is_playing = True
        print(f"Label: {self._current_label()}")
        self._init_display_reference()

    def next_label(self) -> None:
        """Advance to the first record of the next label (wraps around)."""
        self.current_label_idx = (self.current_label_idx + 1) % len(self.labels)
        self.current_record_idx = 0
        self.current_frame_idx = 0
        self.is_playing = True
        print(f"Label: {self._current_label()}")
        self._init_display_reference()

    def prev_label(self) -> None:
        """Go back to the first record of the previous label (wraps around)."""
        self.current_label_idx = (self.current_label_idx - 1) % len(self.labels)
        self.current_record_idx = 0
        self.current_frame_idx = 0
        self.is_playing = True
        print(f"Label: {self._current_label()}")
        self._init_display_reference()

    def next_record(self) -> None:
        """Advance to the next recording within the current label (wraps around)."""
        records = self._current_records()
        self.current_record_idx = (self.current_record_idx + 1) % len(records)
        self.current_frame_idx = 0
        self.is_playing = True
        self._init_display_reference()

    def prev_record(self) -> None:
        """Go back to the previous recording within the current label (wraps around)."""
        records = self._current_records()
        self.current_record_idx = (self.current_record_idx - 1) % len(records)
        self.current_frame_idx = 0
        self.is_playing = True
        self._init_display_reference()

    def next_frame(self) -> None:
        """Advance one frame manually and pause playback."""
        df = self._current_df()
        self.current_frame_idx = (self.current_frame_idx + 1) % len(df)
        self.is_playing = False

    def prev_frame(self) -> None:
        """Go back one frame manually and pause playback."""
        df = self._current_df()
        self.current_frame_idx = (self.current_frame_idx - 1) % len(df)
        self.is_playing = False

    def toggle_play(self) -> None:
        """Toggle between play and pause states."""
        self.is_playing = not self.is_playing
        print(f"Playing: {self.is_playing}")

    def reset(self) -> None:
        """Restart playback from the first frame of the current recording."""
        self.current_frame_idx = 0
        self.is_playing = True
        self._init_display_reference()

    def _update_frame(self) -> None:
        """Advance the frame index by one if playback is active."""
        if not self.is_playing:
            return
        df = self._current_df()
        self.current_frame_idx = (self.current_frame_idx + 1) % len(df)

    # --------------------------------------------------
    # Display helpers
    # --------------------------------------------------

    def _init_display_reference(self) -> None:
        """
        Pre-compute a stable bounding-box center and scale for the current recording.

        Computes the center and size across ALL frames of the sequence so that
        the hand appears consistently centered regardless of the frame shown.
        Avoids recalculating every frame, which would cause visible jitter.
        """
        df = self._current_df()
        # flatten
        lm_cols = [f"lm{i}_{ax}" for i in range(self.total_landmarks) for ax in ("x", "y", "z")]

        # 2 dim
        all_pts = np.stack([
            df[lm_cols].iloc[i].values.reshape(-1, 3)[:, :2]
            for i in range(len(df))
        ])

        min_xy = all_pts.min(axis=(0, 1))
        max_xy = all_pts.max(axis=(0, 1))

        self._display_ref_center = (min_xy + max_xy) / 2.0
        self._display_ref_scale = (
            min(self.frame_w, self.frame_h) * self.size_factor
            / max(float(np.max(max_xy - min_xy)), 1e-6)
        ) # pixels by normalized unity

    def _normalize_for_display(
        self, pts: np.ndarray, w: int, h: int
    ) -> np.ndarray:
        """
        Map normalized landmark coordinates to pixel space centered on the canvas.

        Uses the pre-computed reference so all frames of the same recording
        share the same coordinate mapping.

        :param pts: Landmark array of shape ``(21, 3)``.
        :type pts: np.ndarray
        :param w: Canvas width in pixels.
        :type w: int
        :param h: Canvas height in pixels.
        :type h: int
        :return: 2-D pixel coordinates of shape ``(21, 2)``.
        :rtype: np.ndarray
        """
        screen_center = np.array([w / 2, h / 2])
        return (pts[:, :2] - self._display_ref_center) * self._display_ref_scale + screen_center

    def _landmarks_to_frame(self, frame: np.ndarray, row: pd.Series) -> np.ndarray:
        """
        Render one frame of landmarks (skeleton + dots + HUD) onto the canvas.

        The wrist (landmark 0) is drawn larger and in green for orientation reference.

        :param frame: Blank canvas of shape ``(H, W, 3)``.
        :type frame: np.ndarray
        :param row: DataFrame row containing all ``lmX_*`` columns.
        :type row: pd.Series
        :return: Annotated canvas.
        :rtype: np.ndarray
        """
        h, w = frame.shape[:2]
        lm_cols = [f"lm{i}_{ax}" for i in range(self.total_landmarks) for ax in ("x", "y", "z")]
        lms_full = row[lm_cols].values.reshape(-1, 3)
        pts_2d = self._normalize_for_display(lms_full, w, h)

        for start, end in HAND_CONNECTIONS:
            p1 = (int(pts_2d[start][0]), int(pts_2d[start][1]))
            p2 = (int(pts_2d[end][0]), int(pts_2d[end][1]))
            cv.line(frame, p1, p2, (100, 100, 255), 2)

        for i in range(len(lms_full)):
            px, py = int(pts_2d[i][0]), int(pts_2d[i][1])
            color = (0, 255, 0) if i == 0 else (255, 0, 255)
            radius = 8 if i == 0 else 4
            cv.circle(frame, (px, py), radius, color, -1)

        play_symbol = "P" if self.is_playing else "S"
        cv.putText(frame, f"Label: {self._current_label()}", (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(frame, f"Record: {self.current_record_idx}  {play_symbol}", (10, 75),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv.putText(frame, f"Frame: {self.current_frame_idx}", (10, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    # --------------------------------------------------
    # Main process (FrameProcessor override)
    # --------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Advance playback if enough time has elapsed and render the current frame.

        :param frame: Canvas frame (typically black) of shape ``(H, W, 3)``.
        :type frame: np.ndarray
        :return: Canvas with the landmark skeleton and HUD drawn.
        :rtype: np.ndarray
        """
        current_time = time.time() * 1000
        if self.is_playing and (current_time - self.last_update_time >= self.delay_ms):
            self._update_frame()
            self.last_update_time = current_time

        row = self._current_df().iloc[self.current_frame_idx]
        return self._landmarks_to_frame(frame, row)


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QMessageBox
    from core.window.empty_window import EmptyWindow

    DATA_PATH = AppPaths.path(os.getenv("DATA_PATH", "data/processed/"))
    app = QApplication(sys.argv)

    try:
        processor = ViewerProcessor(DATA_PATH, fps=30)
    except RuntimeError as e:
        QMessageBox.critical(None, "Error loading data", str(e))
        sys.exit(1)

    window = EmptyWindow(width=800, height=600, frame_processor=processor)

    window.add_button("play_pause", text="⏸️", action=processor.toggle_play,
                      tooltip="Play / Pause", color=EmptyWindow.COLORS['orange'],
                      hover_color=EmptyWindow.COLORS['orange_hover'],
                      shortcut="Space", alignment="right", width=50)
    window.add_button("reset", text="⟳", action=processor.reset,
                      tooltip="Restart playback", alignment="right", width=50)
    window.add_button("prev_sign", text="🡸 Sign", action=processor.prev_label,
                      tooltip="Previous sign", alignment="right", width=70)
    window.add_button("next_sign", text="🡺 Sign", action=processor.next_label,
                      tooltip="Next sign", alignment="right", width=70)
    window.add_button("prev_recording", text="🡸 Rec", action=processor.prev_record,
                      tooltip="Previous recording", shortcut="Left", alignment="right", width=70)
    window.add_button("next_recording", text="Rec 🡺", action=processor.next_record,
                      tooltip="Next recording", shortcut="Right", alignment="right", width=70)
    window.add_button("prev_frame", text="🡸 Frame", action=processor.prev_frame,
                      tooltip="Previous frame (pause mode)", alignment="right", width=80)
    window.add_button("next_frame", text="Frame 🡺", action=processor.next_frame,
                      tooltip="Next frame (pause mode)", alignment="right", width=80)
    window.add_button("delete_record", text="Delete", action=processor.delete_current_record,
                      tooltip="Delete current recording",
                      color=EmptyWindow.COLORS['red'], hover_color=EmptyWindow.COLORS['red'],
                      alignment="right", width=70)

    window.show()
    sys.exit(app.exec())
