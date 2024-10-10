import cv2
import subprocess
import datetime
import numpy as np
from logger import logger


class RTSPStreamProcessor:
    def __init__(self, source_rtsp_url, target_rtsp_url, temperature_threshold=180):
        """Initializes the RTSP stream processor with source and target URLs and a temperature threshold.
        Args:
            source_rtsp_url (str): RTSP URL of the source video stream.
            target_rtsp_url (str): RTSP URL where the processed video will be streamed.
            temperature_threshold (int): Threshold for temperature detection, defaults to 180.
        Raises:
            ValueError: If the source RTSP stream cannot be opened.
        """
        self.source_rtsp_url = source_rtsp_url
        self.target_rtsp_url = target_rtsp_url
        self.temperature_threshold = temperature_threshold
        self.cap = cv2.VideoCapture(source_rtsp_url)
        self.ffmpeg_process = None
        self.frame_number = 0
        self.logger = logger()

        if not self.cap.isOpened():
            raise ValueError("Error: Could not open the source RTSP stream.")

        # Get original video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def add_text_to_frame(self, frame, frame_number=None):
        """Adds timestamp and optionally a frame number to a given video frame.
        Args:
            frame (np.array): The video frame to annotate.
            frame_number (int, optional): The current frame number to add to the frame.
        """
        cv2.putText(
            frame,
            datetime.datetime.now().strftime("%H:%M:%S"),
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if frame_number is not None:
            cv2.putText(
                frame,
                f"Frame: {frame_number}",
                (10, 60),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def label_fire(self, frame, x1, y1, x2, y2, colorCode):
        """Labels detected fire in the video frame.
        Args:
            frame (np.array): The video frame to label.
            x1, y1, x2, y2 (int): Coordinates for the bounding rectangle around the detected fire.
            colorCode (list): List containing the color codes for text and rectangle.
        """
        cv2.putText(
            frame,
            "Fire",
            (10, 90),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            colorCode[0],
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            frame,
            (x1 - 20, y1 - 20),
            (x2 + 20, y2 + 20),
            colorCode[1],
            1,
        )
        self.logger(frame, [10, 20, 30, 40])

    def process_and_stream(self):
        """Processes video frames from the source RTSP and streams them to the target RTSP.
        Reads frames, processes for fire detection, and streams the output via FFmpeg.
        """
        print("Processing and streaming video frames...")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                continue

            # Additional processing details here...
            # Truncated for brevity, continue your processing logic as originally implemented...

        self._cleanup()

    def write_logs(self):
        self.logger._write_logs_to_file()

    def _cleanup(self):
        """Releases resources and closes the FFmpeg process."""
        self.cap.release()
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()


# End of rtsp_stream_processor.py
