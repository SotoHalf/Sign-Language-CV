"""
collect_video.py — Batch landmark extraction from pre-recorded video files.

Processes every ``.mov`` file in ``data/resources/lse_videos`` sequentially.
For each video, the filename stem becomes the gesture label and the processor
records one landmark sequence per video.

Because a video may end before the landmark buffer is full, the script uses
``force_landmark_handler_export()`` to flush a partial buffer when the video
finishes. When all videos are processed, records are saved to disk.

Usage:
    python collect_video.py
"""

import os
import sys
from collections import deque
from typing import Deque

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from core.window.video_window import VideoWindow
from core.processors.recording_processor import RecordingProcessor
from core.utils import AppPaths

VIDEOS_PATH: str = AppPaths.path("data/resources/lse_videos")


def main() -> None:
    """
    Build a deque of video files and process them one by one in a VideoWindow.

    A QTimer polls every 200 ms to detect when the current video has finished
    and advances to the next one without blocking the Qt event loop.
    """
    stack: Deque[str] = deque(
        f for f in os.listdir(VIDEOS_PATH) if f.endswith(".mov")
    )

    if not stack:
        print("There are no videos to process")
        return

    app = QApplication(sys.argv)

    video_file: str = stack.pop()
    processor = RecordingProcessor(max_sequences_per_record=1)

    label: str = os.path.splitext(video_file)[0]
    print(f"Label: {label}")
    processor.current_label = label

    window = VideoWindow(
        video_path=os.path.join(VIDEOS_PATH, video_file),
        width=800,
        height=600,
        frame_processor=processor
    )

    # Start recording immediately — no countdown needed for video files
    processor.start_record(label, countdown=False)
    window.show()
    print(f"Videos remaining: {len(stack)}")

    # Guard flag to prevent re-entering the callback while it is still executing
    running: bool = False

    def check_next_video() -> None:
        """
        Timer callback: advance to the next video when the current one ends.

        If the processor is still mid-recording when the video ends, flush the
        partial buffer first. When the queue is empty, save all records and stop.
        """
        nonlocal running, video_file, label
        if running:
            return
        running = True

        if window.is_finished() and stack:
            if processor.is_recording():
                # Video ended before the buffer was full; save whatever was captured
                processor.force_landmark_handler_export()
            else:
                video_file = stack.pop()
                label = os.path.splitext(video_file)[0]
                print(f"Label: {label}")
                processor.current_label = label
                window.set_video_path(os.path.join(VIDEOS_PATH, video_file))
                processor.start_record(label, countdown=False)

        elif not stack:
            print("No more videos to process")
            processor.stop_record()
            print("Saving all records")
            processor.save_records()
            timer.stop()

        running = False

    timer = QTimer()
    timer.timeout.connect(check_next_video)
    timer.start(200)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
