import os
import cv2
import ffmpeg
import logging
from pytube import YouTube
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VideoUtils:
    """
    A utility class for video ingestion, frame sampling, and preprocessing.
    """

    @staticmethod
    def download_youtube_video(url: str, output_dir: str) -> str:
        """
        Downloads a YouTube video and saves it to the specified directory.

        Args:
            url (str): The URL of the YouTube video.
            output_dir (str): The directory to save the downloaded video.

        Returns:
            str: The path to the downloaded video file.
        """
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension="mp4").first()
            if not stream:
                raise ValueError("No suitable video stream found.")
            os.makedirs(output_dir, exist_ok=True)
            output_path = stream.download(output_path=output_dir)
            logging.info(f"Downloaded video: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Failed to download video: {e}")
            raise

    @staticmethod
    def extract_frames(video_path: str, output_dir: str, frame_rate: int = 1) -> List[str]:
        """
        Extracts frames from a video at the specified frame rate.

        Args:
            video_path (str): Path to the video file.
            output_dir (str): Directory to save the extracted frames.
            frame_rate (int): Number of frames to extract per second.

        Returns:
            List[str]: List of file paths to the extracted frames.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_interval = max(1, fps // frame_rate)
            frame_paths = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    frame_name = f"frame_{frame_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_name)
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                frame_count += 1

            cap.release()
            logging.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
            return frame_paths
        except Exception as e:
            logging.error(f"Failed to extract frames: {e}")
            raise

    @staticmethod
    def get_video_metadata(video_path: str) -> dict:
        """
        Retrieves metadata of a video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            dict: Metadata of the video (e.g., duration, resolution, frame rate).
        """
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if not video_stream:
                raise ValueError("No video stream found in file.")

            metadata = {
                "duration": float(video_stream.get("duration", 0)),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "frame_rate": eval(video_stream.get("r_frame_rate", "0")),
                "codec": video_stream.get("codec_name", ""),
            }
            logging.info(f"Video metadata: {metadata}")
            return metadata
        except Exception as e:
            logging.error(f"Failed to retrieve video metadata: {e}")
            raise