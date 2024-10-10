import cv2
import subprocess
import threading
import time
from rtsp_stream_processor import RTSPStreamProcessor
from logger import logger


def main():
    source_rtsp_url = "test.MP4"
    # "rtsp://localhost:8554/source_stream"
    target_rtsp_url = "rtsp://localhost:8554/target_stream"
    video_path = "test.mp4"  # Replace with your video file path
    processor = RTSPStreamProcessor(source_rtsp_url, target_rtsp_url)
    # Create and start threads without delays
    print("Starting threads...")

    # Thread for streaming video file to RTSP server
    processing_thread = threading.Thread(
        target=processor.process_and_stream(), args=(source_rtsp_url, target_rtsp_url)
    )
    processing_thread.start()
    processing_thread.join()
    RTSPStreamProcessor.write_logs()
    # time.sleep(5)


if __name__ == "__main__":
    main()
