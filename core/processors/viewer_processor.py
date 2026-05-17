import os
import glob
import cv2 as cv
import numpy as np
import pandas as pd

from core.processors.frame_processor import FrameProcessor
from core.utils import AppPaths
from core.hand_tracker import HAND_CONNECTIONS

AppPaths.load_env()

class ViewerProcessor(FrameProcessor):
    def __init__(self, data_dir: str, frame_size=(640, 480), fps: int = 30):
        super().__init__()
        self.data_dir = data_dir
        self.frame_w, self.frame_h = frame_size
        self.fps = fps
        self.delay_ms = int(1000 / fps)  # 33ms para 30fps
        self.size_factor = 0.60

        self.total_landmarks = int(os.getenv("HAND_TOTAL_LANDMARKS", 21))

        # label -> [files]
        self.records_by_label = {}
        self.labels = []

        self._load_data()

        if not self.labels:
            raise RuntimeError(f"There are no records found in {self.data_dir}")

        # navegation
        self.current_label_idx = 0
        self.current_record_idx = 0
        self.current_frame_idx = 0
        
        # play 
        self.is_playing = True
        self.last_update_time = 0

        #center
        self._display_ref_center = None
        self._display_ref_scale = None

        self._init_display_reference()

    # ---------------------------------------
    # DATA
    # ---------------------------------------

    def _load_data(self):
        """
        data/
            label1/
                a.csv
                b.csv
            label2/
                c.csv
        """
        for label in os.listdir(self.data_dir):
            label_path = os.path.join(self.data_dir, label)
            print(label_path)
            if not os.path.isdir(label_path):
                continue

            files = sorted(glob.glob(os.path.join(label_path, "*.csv")))
            if not files:
                continue

            #self.records_by_label[label] = [pd.read_csv(f) for f in files]
            records = []
            for f in files:
                df = pd.read_csv(f)
                df.attrs["file_path"] = f
                records.append(df)

            self.records_by_label[label] = records
            self.labels.append(label)


    def delete_current_record(self):
        label = self._current_label()
        records = self._current_records()

        if not records:
            return

        df = records[self.current_record_idx]
        file_path = df.attrs.get("file_path", None)
        base_dir = os.path.dirname(file_path)

        # 1. delete file from disk
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

        # 2. remove from memory
        del records[self.current_record_idx]

        # case 1: still records in label
        if len(records) > 0:
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
            self.finished_reason = "All recordings have been deleted.\nThere are no more signs available to display."
            return 

        # case 4: move to next label safely
        self.current_label_idx %= len(self.labels)
        self.current_record_idx = 0
        self.current_frame_idx = 0
        self._init_display_reference()

    # ---------------------------------------
    # GETTERS
    # ---------------------------------------

    def _current_label(self):
        return self.labels[self.current_label_idx]

    def _current_records(self):
        return self.records_by_label[self._current_label()]

    def _current_df(self):
        return self._current_records()[self.current_record_idx]

    def _current_filename(self):
        return f"{self._current_label()}[{self.current_record_idx}]"

    # ---------------------------------------
    # NAVIGATION
    # ---------------------------------------

    def next_label(self):
        self.current_label_idx = (self.current_label_idx + 1) % len(self.labels)
        self.current_record_idx = 0
        self.current_frame_idx = 0
        self.is_playing = True
        print(f"Label: {self._current_label()}")
        self._init_display_reference()

    def prev_label(self):
        self.current_label_idx = (self.current_label_idx - 1) % len(self.labels)
        self.current_record_idx = 0
        self.current_frame_idx = 0
        self.is_playing = True
        print(f"Label: {self._current_label()}")
        self._init_display_reference()

    def next_record(self):
        records = self._current_records()
        self.current_record_idx = (self.current_record_idx + 1) % len(records)
        self.current_frame_idx = 0
        self.is_playing = True
        self._init_display_reference()

    def prev_record(self):
        records = self._current_records()
        self.current_record_idx = (self.current_record_idx - 1) % len(records)
        self.current_frame_idx = 0
        self.is_playing = True
        self._init_display_reference()

    def next_frame(self):
        df = self._current_df()
        self.current_frame_idx = (self.current_frame_idx + 1) % len(df)
        self.is_playing = False

    def prev_frame(self):
        df = self._current_df()
        self.current_frame_idx = (self.current_frame_idx - 1) % len(df)
        self.is_playing = False

    def toggle_play(self):
        self.is_playing = not self.is_playing
        print(f"Playing: {self.is_playing}")

    def reset(self):
        self.current_frame_idx = 0
        self.is_playing = True
        self._init_display_reference()

    def _update_frame(self):
        """Advance to next frame if playing"""
        if not self.is_playing:
            return
            
        df = self._current_df()
        self.current_frame_idx = (self.current_frame_idx + 1) % len(df)
        

    def _init_display_reference(self):
        df = self._current_df()

        lm_cols = [f"lm{i}_{axis}" for i in range(self.total_landmarks) for axis in ("x","y","z")]

        all_pts = np.stack([
            df[lm_cols].iloc[i].values.reshape(-1, 3)[:, :2]
            for i in range(len(df))
        ])

        min_xy = all_pts.min(axis=(0, 1))
        max_xy = all_pts.max(axis=(0, 1))

        center = (min_xy + max_xy) / 2.0
        size = np.max(max_xy - min_xy)

        self._display_ref_center = center
        self._display_ref_scale = min(self.frame_w, self.frame_h) * self.size_factor / max(size, 1e-6)

    def _normalize_for_display(self, pts, w, h):
        """
        Scale and center landmarks for consistent visualization for different cameras
        """

        pts_2d = pts[:, :2]

        screen_center = np.array([w / 2, h / 2])

        pts_norm = (
            (pts_2d - self._display_ref_center)
            * self._display_ref_scale
            + screen_center
        )

        return pts_norm

    def _landmarks_to_frame(self, frame, row):
        h, w = frame.shape[:2]
        #cx, cy = w // 2, h // 1.25
        #scale = min(w, h) * 0.2

        lm_cols = [f"lm{i}_{axis}" for i in range(0, self.total_landmarks) for axis in ("x","y","z")]
        lms_full = row[lm_cols].values.reshape(-1, 3)

        pts = lms_full
        pts_2d = self._normalize_for_display(pts, w, h)
        
        for start, end in HAND_CONNECTIONS:
            #p1 = (int(lms_full[start][0] * scale + cx), int(lms_full[start][1] * scale + cy))
            #p2 = (int(lms_full[end][0] * scale + cx), int(lms_full[end][1] * scale + cy))
            p1 = (int(pts_2d[start][0]), int(pts_2d[start][1]))
            p2 = (int(pts_2d[end][0]), int(pts_2d[end][1]))
            cv.line(frame, p1, p2, (100, 100, 255), 2)

        for i, lm in enumerate(lms_full):
            #px = int(lm[0] * scale + cx)
            #py = int(lm[1] * scale + cy)
            px = int(pts_2d[i][0])
            py = int(pts_2d[i][1])
            color = (0, 255, 0) if i == 0 else (255, 0, 255)
            radius = 8 if i == 0 else 4
            cv.circle(frame, (px, py), radius, color, -1)

        play_symbol = "P" if self.is_playing else "S"
        cv.putText(frame, f"Label: {self._current_label()}", (10, 50),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(frame, f"Record: {self.current_record_idx}  {play_symbol}", (10, 75),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if self.is_playing else (0, 0, 255), 2)
        cv.putText(frame, f"Frame: {self.current_frame_idx}", (10, 100),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    # ---------------------------------------
    # MAIN PROCESS (override)
    # ---------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        # Control time to respect fps
        import time
        current_time = time.time() * 1000  # ms
        
        if self.is_playing and (current_time - self.last_update_time >= self.delay_ms):
            self._update_frame()
            self.last_update_time = current_time
        
        df = self._current_df()
        row = df.iloc[self.current_frame_idx]

        return self._landmarks_to_frame(frame, row)
    

if __name__ == "__main__":
    import sys
    import time
    from PySide6.QtWidgets import QApplication
    from core.window.empty_window import EmptyWindow

    DATA_PATH = AppPaths.path(os.getenv("DATA_PATH", "data/processed/"))

    app = QApplication(sys.argv)

    try:
        processor = ViewerProcessor(DATA_PATH, fps=30)
    except RuntimeError as e:
        QMessageBox.critical(
            None,
            "Error loading data",
            str(e)
        )
        sys.exit(1)

    window = EmptyWindow(
        width=800,
        height=600,
        frame_processor=processor,
    )

    # PLAY/PAUSE
    window.add_button(
        "play_pause",
        text="⏸️",
        action=processor.toggle_play,
        tooltip="Play / Pause",
        color=EmptyWindow.COLORS['orange'],
        hover_color=EmptyWindow.COLORS['orange_hover'],
        shortcut="Space",
        alignment="right",
        width=50
    )

    window.add_button(
        "reset",
        text="⟳",
        action=processor.reset,
        tooltip="Restart playback",
        alignment="right",
        width=50
    )

    # SIGN (label)
    window.add_button(
        "prev_sign",
        text="🡸 Sign",
        action=processor.prev_label,
        tooltip="Previous sign",
        alignment="right",
        width=70
    )

    window.add_button(
        "next_sign",
        text="🡺 Sign",
        action=processor.next_label,
        tooltip="Next sign",
        alignment="right",
        width=70
    )

    # RECORDING
    window.add_button(
        "prev_recording",
        text="🡸 Rec",
        action=processor.prev_record,
        tooltip="Previous recording",
        shortcut="Left",
        alignment="right",
        width=70
    )

    window.add_button(
        "next_recording",
        text="Rec 🡺",
        action=processor.next_record,
        tooltip="Next recording",
        shortcut="Right",
        alignment="right",
        width=70
    )

    # FRAME CONTROL (fine control)
    window.add_button(
        "prev_frame",
        text="🡸 Frame",
        action=processor.prev_frame,
        tooltip="Previous frame (pause mode)",
        alignment="right",
        width=80
    )

    window.add_button(
        "next_frame",
        text="Frame 🡺",
        action=processor.next_frame,
        tooltip="Next frame (pause mode)",
        alignment="right",
        width=80
    )

    window.add_button(
        "delete_record",
        text="Delete",
        action=processor.delete_current_record,
        tooltip="Delete current recording",
        color=EmptyWindow.COLORS['red'],
        hover_color=EmptyWindow.COLORS['red'],
        alignment="right",
        width=70
    )

    window.show()
    sys.exit(app.exec())


    window.show()
    sys.exit(app.exec())