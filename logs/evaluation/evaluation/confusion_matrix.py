import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import imagehash
from PIL import Image
import mediapipe as mp
import logging
from enum import Enum

class FilterReason(Enum):
    """Enumeration of reasons why a frame might be filtered out."""
    BLURRY = "blurry"
    DUPLICATE = "duplicate"
    NO_MOTION = "no_motion"
    NO_PLAYERS = "no_players"
    OUTSIDE_ROI = "outside_roi"

@dataclass
class FilterConfig:
    """Configuration parameters for frame filtering."""
    blur_threshold: float = 100.0
    duplicate_threshold: int = 4  # Hash difference threshold
    motion_threshold: float = 0.03
    min_player_detection_confidence: float = 0.5
    roi: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h

class FrameFilter:
    """Main class for filtering video frames based on various criteria."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the frame filter with configuration."""
        self.config = FilterConfig(**config) if config else FilterConfig()
        self.prev_frame_hash = None
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.config.min_player_detection_confidence
        )
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("FrameFilter")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def check_blur(self, frame: np.ndarray) -> bool:
        """Check if frame is too blurry using Laplacian variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.config.blur_threshold

    def check_duplicate(self, frame: np.ndarray) -> bool:
        """Check if frame is a duplicate using perceptual hashing."""
        current_hash = imagehash.average_hash(Image.fromarray(frame))
        if self.prev_frame_hash is None:
            self.prev_frame_hash = current_hash
            return False
        is_duplicate = (current_hash - self.prev_frame_hash) < self.config.duplicate_threshold
        self.prev_frame_hash = current_hash
        return is_duplicate

    def check_motion(self, frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> bool:
        """Check if there's sufficient motion between frames."""
        if prev_frame is None:
            return False
        diff = cv2.absdiff(frame, prev_frame)
        motion_score = np.mean(diff) / 255.0
        return motion_score < self.config.motion_threshold

    def check_players(self, frame: np.ndarray) -> bool:
        """Check if players are present using MediaPipe Pose."""
        results = self.pose.process(frame)
        return results.pose_landmarks is None

    def check_roi(self, frame: np.ndarray) -> bool:
        """Check if main action is within region of interest."""
        if not self.config.roi:
            return False
        x, y, w, h = self.config.roi
        roi = frame[y:y+h, x:x+w]
        return np.mean(roi) < 30  # Simple check for activity in ROI

    def filter_frame(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> Tuple[bool, FilterReason]:
        """Apply all filters to a single frame."""
        if self.check_blur(frame):
            return True, FilterReason.BLURRY
        if self.check_duplicate(frame):
            return True, FilterReason.DUPLICATE
        if prev_frame is not None and self.check_motion(frame, prev_frame):
            return True, FilterReason.NO_MOTION
        if self.check_players(frame):
            return True, FilterReason.NO_PLAYERS
        if self.check_roi(frame):
            return True, FilterReason.OUTSIDE_ROI
        return False, None

    def filter_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Filter a list of frames using all criteria."""
        filtered_frames = []
        filter_stats = {reason: 0 for reason in FilterReason}
        
        with ThreadPoolExecutor() as executor:
            for i, frame in enumerate(tqdm(frames)):
                prev_frame = frames[i-1] if i > 0 else None
                should_filter, reason = self.filter_frame(frame, prev_frame)
                
                if not should_filter:
                    filtered_frames.append(frame)
                else:
                    filter_stats[reason] += 1
                    self.logger.debug(f"Frame {i} filtered: {reason.value}")

        self.logger.info("Filtering statistics:")
        for reason, count in filter_stats.items():
            if count > 0:
                self.logger.info(f"{reason.value}: {count} frames")

        return filtered_frames

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Filter video frames based on quality criteria")
    parser.add_argument("--input", required=True, help="Path to input video or folder")
    parser.add_argument("--output", required=True, help="Path to output filtered frames")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config) if Path(args.config).exists() else None
    frame_filter = FrameFilter(config)

    # Video processing logic here
    cap = cv2.VideoCapture(args.input)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    filtered_frames = frame_filter.filter_frames(frames)
    
    # Save filtered frames
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(filtered_frames):
        cv2.imwrite(str(output_path / f"frame_{i:06d}.jpg"), frame)

if __name__ == "__main__":
    main()