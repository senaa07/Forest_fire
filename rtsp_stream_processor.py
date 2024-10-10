# rtsp_stream_processor.py
import cv2
import subprocess
import datetime
import numpy as np


class RTSPStreamProcessor:
    def __init__(self, source_rtsp_url, target_rtsp_url, temperature_threshold=180):
        self.source_rtsp_url = source_rtsp_url
        self.target_rtsp_url = target_rtsp_url
        self.temperature_threshold = temperature_threshold
        self.cap = cv2.VideoCapture(source_rtsp_url)
        self.ffmpeg_process = None
        self.frame_number = 0

        if not self.cap.isOpened():
            raise ValueError("Error: Could not open the source RTSP stream.")

        # Get original video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Adding date ,time and frame no to the frames list
    def add_text_to_frame(self, frame, frame_number=None):
        """Adds timestamp and frame number to frames."""
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

    def process_and_stream(self):
        """Processes and streams video frames from the source RTSP to the target RTSP."""
        print("Processing and streaming video frames...")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                continue

            right_half_frame = frame[:, self.original_width // 2 :]
            resized_thermal_frame = cv2.resize(
                right_half_frame, (self.original_width // 4, self.original_height // 2)
            )

            # Convert resized_thermal_frame to grayscale and detect fire-like regions
            thermal_gray = cv2.cvtColor(resized_thermal_frame, cv2.COLOR_BGR2GRAY)
            _, fire_mask = cv2.threshold(
                thermal_gray, self.temperature_threshold, 255, cv2.THRESH_BINARY
            )
            contours, _ = cv2.findContours(
                fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Getting the RBG video and resizing it

            left_half = frame[:, : self.original_width // 2]

            left_half = cv2.resize(
                left_half, (self.original_width // 4, self.original_height // 2)
            )
            right_half_frame = frame[:, self.original_width // 2 :]
            height, width = left_half.shape[:2]
            # print(height,width)
            left_half = left_half[
                height // 4 - 20 : (height // 4) * 3 - 20,
                width // 4 + 6 : (width // 4) * 3 + 6,
            ]
            left_half = cv2.resize(
                left_half, (self.original_width // 4, self.original_height // 2)
            )

            self.add_text_to_frame(resized_thermal_frame, self.frame_number)
            self.add_text_to_frame(left_half)

            x1, y1 = 1000, 1000
            x2, y2 = 0, 0

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x1 = min(x, x1)
                y1 = min(y, y1)
                x2 = max(x + w, x2)
                y2 = max(y + h, y2)

            if contours:
                self.label_fire(left_half, x1, y1, x2, y2, [(0, 255, 255), (0, 255, 0)])
                self.label_fire(
                    resized_thermal_frame, x1, y1, x2, y2, [(0, 255, 255), (0, 255, 0)]
                )

            # Drawing contours on the images
            cv2.drawContours(resized_thermal_frame, contours, -1, (0, 255, 0), 2)
            cv2.drawContours(left_half, contours, -1, (0, 0, 255), 2)

        hor = np.concatenate((left_half, resized_thermal_frame), axis=1)
        # Display the video frames
        cv2.imshow("video", resized_thermal_frame)

        thermal_height, thermal_width, _ = hor.shape
        if ffmpeg_process is None:
            ffmpeg_cmd = (
                f"ffmpeg -f rawvideo -pixel_format bgr24 -video_size {thermal_width}x{thermal_height} "
                f"-framerate {self.fps} -i pipe:0 -c:v libx264 -preset veryfast -tune zerolatency "
                f"-f rtsp -rtsp_transport tcp {self.target_rtsp_url}"
            )
            ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE, shell=True
            )

            # Write resized_thermal_frame to FFmpeg stdin
            ffmpeg_process.stdin.write(hor.tobytes())
        self._cleanup()

    def _cleanup(self):
        """Releases resources and closes the FFmpeg process."""
        self.cap.release()
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()


# End of rtsp_stream_processor.py
