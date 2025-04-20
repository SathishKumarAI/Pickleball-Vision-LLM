import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import imagehash
from PIL import Image
from pathlib import Path
import yaml
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import mediapipe as mp
import argparse

@dataclass
class FilterConfig:
    """Configuration for frame filtering parameters."""
    blur_threshold: float = 100.0  # Laplacian variance threshold
    duplicate_threshold: int = 4    # Hamming distance for image hash
    motion_threshold: float = 0.02  # Minimum fraction of pixels with motion
    roi_bounds: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    min_players: int = 2           # Minimum number of players in frame

class FrameFilter:
    """Main class for filtering video frames based on various criteria."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the frame filter with optional config file."""
        self.config = FilterConfig()
        if config_path:
            self._load_config(config_path)
            
        self.pose_detector = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.previous_hash = None
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('FrameFilter')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = FilterConfig(**config_dict)

    def is_blurry(self, frame: np.ndarray) -> bool:
        """Check if frame is blurry using Laplacian variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.config.blur_threshold

    def is_duplicate(self, frame: np.ndarray) -> bool:
        """Check if frame is a duplicate using perceptual hashing."""
        current_hash = imagehash.average_hash(Image.fromarray(frame))
        if self.previous_hash is None:
            self.previous_hash = current_hash
            return False
        is_dup = (current_hash - self.previous_hash) < self.config.duplicate_threshold
        self.previous_hash = current_hash
        return is_dup

    def has_motion(self, frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> bool:
        """Detect if there is significant motion between frames."""
        if prev_frame is None:
            return True
        
        diff = cv2.absdiff(frame, prev_frame)
        motion_pixels = np.sum(diff > 30)  # threshold for motion
        total_pixels = frame.shape[0] * frame.shape[1]
        return (motion_pixels / total_pixels) > self.config.motion_threshold

    def has_enough_players(self, frame: np.ndarray) -> bool:
        """Check if frame contains minimum number of players using MediaPipe."""
        results = self.pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return False
        return True  # Simplified check - could be enhanced to count actual players

    def filter_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter frames based on configured criteria.
        
        Args:
            frames: List of input frames as numpy arrays
            
        Returns:
            List of filtered frames
        """
        filtered_frames = []
        prev_frame = None
        
        for frame in tqdm(frames, desc="Filtering frames"):
            if self.is_blurry(frame):
                self.logger.debug("Rejected: Blurry frame")
                continue
                
            if self.is_duplicate(frame):
                self.logger.debug("Rejected: Duplicate frame")
                continue
                
            if not self.has_motion(frame, prev_frame):
                self.logger.debug("Rejected: No motion detected")
                continue
                
            if not self.has_enough_players(frame):
                self.logger.debug("Rejected: Not enough players")
                continue
                
            filtered_frames.append(frame)
            prev_frame = frame.copy()
            
        return filtered_frames

    def process_video(self, input_path: str, output_path: str) -> None:
        """Process video file and save filtered frames."""
        cap = cv2.VideoCapture(input_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        
        filtered_frames = self.filter_frames(frames)
        
        # Save filtered frames or video
        if filtered_frames:
            height, width = filtered_frames[0].shape[:2]
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,  # fps
                (width, height)
            )
            
            for frame in filtered_frames:
                out.write(frame)
            out.release()

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Filter video frames")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--config", help="Path to config YAML file")
    args = parser.parse_args()
    
    filter = FrameFilter(args.config)
    filter.process_video(args.input, args.output)

if __name__ == "__main__":
    main()