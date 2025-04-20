import os
import cv2
import ffmpeg
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FrameExtractor:
    """
    A class to handle frame extraction from gameplay videos.
    """

    def __init__(self, video_path: str, output_dir: str, frame_rate: int = 5):
        """
        Initialize the FrameExtractor.

        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory to save extracted frames.
            frame_rate (int): Number of frames to extract per second.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_rate = frame_rate

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"Created output directory: {self.output_dir}")

    def extract_frames(self) -> List[str]:
        """
        Extract frames from the video at the specified frame rate.

        Returns:
            List[str]: List of file paths to the extracted frames.
        """
        logging.info(f"Starting frame extraction from {self.video_path} at {self.frame_rate} FPS.")
        frame_paths = []

        # Open video file
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {self.video_path}")
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, fps // self.frame_rate)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_name = f"frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(self.output_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                logging.debug(f"Extracted frame: {frame_path}")

            frame_count += 1

        cap.release()
        logging.info(f"Extracted {len(frame_paths)} frames to {self.output_dir}.")
        return frame_paths

    @staticmethod
    def is_playable_frame(frame: cv2.Mat) -> bool:
        """
        Determine if a frame is a playable moment based on simple heuristics.

        Args:
            frame (cv2.Mat): The frame to analyze.

        Returns:
            bool: True if the frame is a playable moment, False otherwise.
        """
        # Placeholder for advanced logic (e.g., scene detection, motion analysis)
        # For now, assume all frames are playable
        return True


if __name__ == "__main__":
    # Example usage
    video_path = "sample_video.mp4"
    output_dir = "extracted_frames"
    frame_rate = 5

    try:
        extractor = FrameExtractor(video_path, output_dir, frame_rate)
        extracted_frames = extractor.extract_frames()
        logging.info(f"Frame extraction complete. Total frames: {len(extracted_frames)}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")