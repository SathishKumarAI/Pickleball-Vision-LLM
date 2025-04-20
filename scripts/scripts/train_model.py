import logging
import yaml
import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple
import imagehash
from PIL import Image
from scipy.spatial.distance import hamming
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

#!/usr/bin/env python3

import torchvision.transforms as T

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameFilter:
    """Main class for filtering video frames based on multiple criteria."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the frame filter with configuration.
        
        Args:
            config_path: Path to YAML config file with filter parameters
        """
        # Default configuration
        self.config = {
            'blur_threshold': 100,  # Laplacian variance threshold
            'duplicate_threshold': 0.9,  # Hamming distance threshold
            'motion_threshold': 0.03,  # Minimum motion percentage
            'roi_bounds': None,  # Region of interest [x1, y1, x2, y2]
            'min_resolution': (480, 360),  # Minimum frame resolution
            'batch_size': 32,  # Batch size for processing
            'n_workers': 4  # Number of worker threads
        }
        
        if config_path:
            self._load_config(config_path)
            
        self.previous_hash = None
        self.filter_stats = {
            'blurry': 0,
            'duplicate': 0,
            'low_motion': 0,
            'low_resolution': 0,
            'outside_roi': 0
        }
        
    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                self.config.update(loaded_config)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            
    def _check_blur(self, frame: np.ndarray) -> bool:
        """Check if frame is blurry using Laplacian variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.config['blur_threshold']
    
    def _check_duplicate(self, frame: np.ndarray) -> bool:
        """Check if frame is duplicate using perceptual hashing."""
        current_hash = imagehash.average_hash(Image.fromarray(frame))
        
        if self.previous_hash is None:
            self.previous_hash = current_hash
            return False
            
        similarity = 1 - (current_hash - self.previous_hash) / 64
        self.previous_hash = current_hash
        return similarity > self.config['duplicate_threshold']
    
    def _check_motion(self, frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> bool:
        """Check if frame contains sufficient motion."""
        if prev_frame is None:
            return False
            
        diff = cv2.absdiff(frame, prev_frame)
        motion_ratio = np.count_nonzero(diff) / diff.size
        return motion_ratio < self.config['motion_threshold']
    
    def _check_resolution(self, frame: np.ndarray) -> bool:
        """Check if frame meets minimum resolution requirements."""
        height, width = frame.shape[:2]
        min_width, min_height = self.config['min_resolution']
        return width < min_width or height < min_height

    def process_frame(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """Process a single frame and return if it should be filtered and why."""
        if self._check_resolution(frame):
            self.filter_stats['low_resolution'] += 1
            return True, 'low_resolution'
            
        if self._check_blur(frame):
            self.filter_stats['blurry'] += 1
            return True, 'blurry'
            
        if self._check_duplicate(frame):
            self.filter_stats['duplicate'] += 1
            return True, 'duplicate'
            
        if prev_frame is not None and self._check_motion(frame, prev_frame):
            self.filter_stats['low_motion'] += 1
            return True, 'low_motion'
            
        return False, 'pass'