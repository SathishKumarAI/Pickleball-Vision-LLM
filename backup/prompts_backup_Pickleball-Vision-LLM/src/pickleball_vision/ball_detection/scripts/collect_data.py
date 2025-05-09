"""
Data Collection Script for Pickleball Ball Detection

This script processes videos to extract frames for ball detection training.
It implements several techniques to ensure quality data collection:
1. High-resolution frame extraction
2. Frame quality assessment
3. Automatic frame selection based on motion
4. Basic preprocessing for annotation
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import yaml
from typing import Tuple, List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
INPUT_RESOLUTION = (1280, 720)  # Input video resolution
OUTPUT_RESOLUTION = (1280, 1280)  # Processing resolution
FRAME_QUALITY_THRESHOLD = 50  # Minimum frame quality score
MOTION_THRESHOLD = 0.1  # Minimum motion between frames
MAX_FRAMES_PER_VIDEO = 1000  # Maximum frames to extract per video

class DataCollector:
    """Handles the collection and preprocessing of training data."""
    
    def __init__(self, config_path: str):
        """
        Initialize the data collector.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.frame_dir = self.output_dir / 'frames'
        self.frame_dir.mkdir(exist_ok=True)
        self.metadata_dir = self.output_dir / 'metadata'
        self.metadata_dir.mkdir(exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def assess_frame_quality(self, frame: np.ndarray) -> float:
        """
        Assess the quality of a frame.
        
        Checks:
        1. Brightness and contrast
        2. Blur detection
        3. Noise estimation
        
        Args:
            frame: Input frame
            
        Returns:
            Quality score (0-100)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check brightness
        brightness = np.mean(gray)
        if brightness < 30 or brightness > 225:
            return 0
            
        # Check contrast
        contrast = np.std(gray)
        if contrast < 20:
            return 0
            
        # Check blur
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.var(laplacian)
        if blur_score < 100:
            return 0
            
        # Calculate final quality score
        quality = min(100, (blur_score / 500) * 100)
        return quality

    def detect_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate the amount of motion between frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Motion score (0-1)
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate motion magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return np.mean(magnitude)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for ball detection.
        
        Steps:
        1. Resize to target resolution
        2. Enhance contrast
        3. Reduce noise
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Resize
        frame = cv2.resize(frame, OUTPUT_RESOLUTION)
        
        # Enhance contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Reduce noise
        frame = cv2.fastNlMeansDenoisingColored(frame)
        
        return frame

    def process_video(self, video_path: str) -> Dict[str, List]:
        """
        Process a video file to extract training frames.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Dictionary containing frame metadata
        """
        logger.info(f"Processing video: {video_path}")
        metadata = {
            'frames': [],
            'quality_scores': [],
            'motion_scores': []
        }
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        prev_frame = None
        
        while cap.isOpened() and frame_count < MAX_FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            frame = cv2.resize(frame, INPUT_RESOLUTION)
            
            # Assess frame quality
            quality = self.assess_frame_quality(frame)
            
            # Check motion if we have a previous frame
            motion_score = 0
            if prev_frame is not None:
                motion_score = self.detect_motion(prev_frame, frame)
            
            # Save frame if it meets quality criteria
            if quality > FRAME_QUALITY_THRESHOLD and motion_score > MOTION_THRESHOLD:
                # Preprocess frame
                processed = self.preprocess_frame(frame)
                
                # Save frame
                frame_path = self.frame_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), processed)
                
                # Update metadata
                metadata['frames'].append(str(frame_path))
                metadata['quality_scores'].append(float(quality))
                metadata['motion_scores'].append(float(motion_score))
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
            
            prev_frame = frame.copy()
        
        cap.release()
        return metadata

    def save_metadata(self, metadata: Dict[str, List], video_path: str):
        """Save frame metadata to YAML file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_path = self.metadata_dir / f"metadata_{timestamp}.yaml"
        
        # Add processing info
        metadata['video_path'] = video_path
        metadata['timestamp'] = timestamp
        metadata['settings'] = {
            'input_resolution': INPUT_RESOLUTION,
            'output_resolution': OUTPUT_RESOLUTION,
            'quality_threshold': FRAME_QUALITY_THRESHOLD,
            'motion_threshold': MOTION_THRESHOLD
        }
        
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        logger.info(f"Saved metadata to {metadata_path}")

def main():
    """Main function to run data collection."""
    # Load configuration
    config_path = "config/data_collection.yaml"
    collector = DataCollector(config_path)
    
    # Process videos
    video_dir = Path(collector.config['video_dir'])
    for video_path in video_dir.glob("*.mp4"):
        metadata = collector.process_video(str(video_path))
        collector.save_metadata(metadata, str(video_path))

if __name__ == "__main__":
    main() 