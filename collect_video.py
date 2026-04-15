import os
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from collections import deque

from core.window.video_window import VideoWindow
from core.processors.recording_processor import RecordingProcessor

VIDEOS_PATH = "data/resources/lse_videos"

def main():
    
    stack = deque()
    for f in os.listdir(VIDEOS_PATH):
        if f.endswith(".mov"):
            stack.append(f)

    if len(stack) <= 0: 
        print("There are no videos to process")

    app = QApplication(sys.argv)
    video_file = stack.pop()
    
    processor = RecordingProcessor(max_sequences_per_record=1)

    label = os.path.splitext(video_file)[0]
    print(f"Label: {label}")
    processor.current_label = label


    window = VideoWindow(
        video_path=os.path.join(VIDEOS_PATH, video_file),
        width=800,
        height=600,
        frame_processor=processor
    )

    processor.start_record(label, countdown=False)

    window.show()
    """
    while len(stack) > 0:
        if window.is_finished():
            video_file = stack.pop()
            video_path = os.path.join(VIDEOS_PATH, video_file)
            window.set_video_path(video_path)
    """
    print(f"Videos to process {len(stack)}")
    running = False
    def check_next_video():
        nonlocal running
        if running:
            return
        else:
            running = True

        if window.is_finished() and stack:
            if processor.is_recording():
                processor.force_landmark_handler_export()
            else:
                video_file = stack.pop()
                video_path = os.path.join(VIDEOS_PATH, video_file)
                window.set_video_path(video_path)

                label = os.path.splitext(video_file)[0]
                print(f"Label: {label}")
                processor.current_label = label
                processor.start_record(label, countdown=False)

        elif len(stack) <= 0:
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