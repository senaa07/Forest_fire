import cv2
import subprocess
import threading
import time
import datetime
import numpy as np


def stream_video_to_rtsp(rtsp_url, video_path):
    """Streams a video file to an RTSP server."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define FFmpeg command with reduced latency settings
    ffmpeg_cmd = (
        f"ffmpeg -f rawvideo -pixel_format bgr24 -video_size {video_width}x{video_height} "
        f"-framerate {fps} -i pipe:0 -c:v libx264 -preset veryfast -tune zerolatency "
        f"-f rtsp -rtsp_transport tcp {rtsp_url}"
    )

    # Start FFmpeg process
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, shell=True)

    print(f"Streaming video {video_path} to RTSP server...")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Reached end of video file or failed to read frame.")
            break

        # Check if the frame is valid (not None and correct shape)
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("Invalid frame detected, skipping...")
            continue

        # Resize the frame (e.g., half the size)
        # frame = cv2.resize(frame, (width // 2, height // 2))
        # cv2.imshow("src",frame)

        # Write frame to FFmpeg stdin
        ffmpeg_process.stdin.write(frame.tobytes())

    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()


def process_and_stream_rtsp(source_rtsp_url, target_rtsp_url):
    temperature_threshold = 180
    frame_number = 0
    """Retrieves video frames from an RTSP server, modifies them, resizes the right half, and streams to another RTSP server."""
    cap = cv2.VideoCapture(source_rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open the source RTSP stream.")
        return

    # Get original video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Start FFmpeg process without specifying the output size yet (will be computed dynamically)
    ffmpeg_process = None

    print("Processing and streaming video frames...")

    while cap.isOpened():
        frame_number += 1
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Check if the frame is valid
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("Invalid frame detected, skipping...")
            continue

        right_half_frame = frame[:, original_width // 2 :]

        # Resize the right half frame to half of its size
        resized_thermal_frame = cv2.resize(
            right_half_frame, (original_width // 4, original_height // 2)
        )

        # Get dimensions of resized_thermal_frame
        thermal_height, thermal_width, _ = resized_thermal_frame.shape

        # Convert resized_thermal_frame to grayscale and detect fire-like regions
        thermal_gray = cv2.cvtColor(resized_thermal_frame, cv2.COLOR_BGR2GRAY)
        _, fire_mask = cv2.threshold(
            thermal_gray, temperature_threshold, 255, cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Drawing contours on the resized_thermal_frame
        cv2.drawContours(resized_thermal_frame, contours, -1, (0, 255, 0), 2)
        # cv2.imshow('Fire Detection Thermal', resized_thermal_frame)
        left_half = frame[:, : original_width // 2]

        left_half = cv2.resize(left_half, (original_width // 4, original_height // 2))
        right_half_frame = frame[:, original_width // 2 :]
        height, width = left_half.shape[:2]
        # print(height,width)
        left_half = left_half[
            height // 4 - 20 : (height // 4) * 3 - 20,
            width // 4 + 6 : (width // 4) * 3 + 6,
        ]
        left_half = cv2.resize(left_half, (original_width // 4, original_height // 2))

        # Resize the right half frame to half of its size
        resized_thermal_frame = cv2.resize(
            right_half_frame, (original_width // 4, original_height // 2)
        )

        # Get dimensions of resized_thermal_frame
        thermal_height, thermal_width, _ = resized_thermal_frame.shape

        # Convert resized_thermal_frame to grayscale and detect fire-like regions
        thermal_gray = cv2.cvtColor(resized_thermal_frame, cv2.COLOR_BGR2GRAY)
        _, fire_mask = cv2.threshold(
            thermal_gray, temperature_threshold, 255, cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        x1, y1 = 1000, 1000
        x2, y2 = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x1 = min(x, x1)
            y1 = min(y, y1)
            x2 = max(x + w, x2)
            y2 = max(y + h, y2)

        cv2.putText(
            resized_thermal_frame,
            datetime.datetime.now().strftime("%H:%M:%S"),
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            left_half,
            datetime.datetime.now().strftime("%H:%M:%S"),
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            resized_thermal_frame,
            str(frame_number),
            (10, 60),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if contours:
            cv2.putText(
                resized_thermal_frame,
                "Fire",
                (10, 90),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                resized_thermal_frame,
                (x1 - 20, y1 - 20),
                (x2 + 20, y2 + 20),
                (0, 255, 0),
                1,
            )
            cv2.rectangle(
                left_half, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), (0, 255, 0), 1
            )
            cv2.putText(
                left_half,
                "Fire",
                (10, 90),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            # Drawing contours on the resized_thermal_frame
            cv2.drawContours(resized_thermal_frame, contours, -1, (255, 255, 255), 2)
            cv2.drawContours(left_half, contours, -1, (0, 0, 255), 2)
        # cv2.imshow('Fire Detection Thermal', resized_thermal_frame)
        # cv2.imshow("original",left_half)
        hor = np.concatenate((left_half, resized_thermal_frame), axis=1)
        # cv2.imshow('combinerd',hor)
        thermal_height, thermal_width, _ = hor.shape

        # Initialize FFmpeg process based on the resized thermal frame size (first frame only)
        if ffmpeg_process is None:
            ffmpeg_cmd = (
                f"ffmpeg -f rawvideo -pixel_format bgr24 -video_size {thermal_width}x{thermal_height} "
                f"-framerate {fps} -i pipe:0 -c:v libx264 -preset veryfast -tune zerolatency "
                f"-f rtsp -rtsp_transport tcp {target_rtsp_url}"
            )
            ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE, shell=True
            )

        # Convert resized_thermal_frame to grayscale and detect fire-like regions

        # Write resized_thermal_frame to FFmpeg stdin
        ffmpeg_process.stdin.write(hor.tobytes())

    cap.release()
    if ffmpeg_process:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()


def play_rtsp_stream(rtsp_url):
    """Plays video from an RTSP server."""
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open the RTSP stream.")
        return

    print("Playing RTSP stream...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame from RTSP stream. Exiting...")
            break

        # Display the frame
        cv2.imshow("RTSP Stream", frame)

        # Exit when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    # # Define RTSP URLs and video path

    source_rtsp_url = "rtsp://192.168.144.25:8554/video1"
    target_rtsp_url = "rtsp://192.168.43.135:8554/target"

    # source_rtsp_url = 'rtsp://192.168.43.176:8554/source_stream'

    # source_rtsp_url = 'rtsp://localhost:8554/source_stream'
    # target_rtsp_url = 'rtsp://localhost:8554/target'

    video_path = "test.mp4"  # Replace with your video file path

    # Create and start threads without delays
    print("Starting threads...")

    # Thread for streaming video file to RTSP server
    # video_thread = threading.Thread(target=stream_video_to_rtsp, args=(source_rtsp_url, video_path))
    # video_thread.start()
    # time.sleep(10)

    # Thread for processing and streaming RTSP video
    # processing_thread = threading.Thread(target=process_and_stream_rtsp, args=(source_rtsp_url, target_rtsp_url))
    # processing_thread.start()
    # time.sleep(10)

    # # Thread for playing RTSP video
    # playback_thread = threading.Thread(target=play_rtsp_stream, args=(target_rtsp_url,))
    # playback_thread.start()

    # # Wait for all threads to finish
    # video_thread.join()
    # processing_thread.join()
    # playback_thread.join()
    process_and_stream_rtsp(source_rtsp_url, target_rtsp_url)


if __name__ == "__main__":
    main()
