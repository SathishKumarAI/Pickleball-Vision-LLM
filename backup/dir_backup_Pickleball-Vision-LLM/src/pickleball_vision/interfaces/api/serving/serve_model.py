"""Model serving module for pickleball vision."""
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
import os
import sys

from ...core.config.config import Config
from ...vision.preprocessor import FramePreprocessor
from ...vision.detector import ObjectDetector

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
        
        # Initialize components
        self.preprocessor = FramePreprocessor()
        self.detector = ObjectDetector()

    # ... rest of the existing code ... 