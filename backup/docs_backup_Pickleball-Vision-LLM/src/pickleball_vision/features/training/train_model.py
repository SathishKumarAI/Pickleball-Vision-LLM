"""Model training module for pickleball vision."""
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
import os
import sys
from pathlib import Path
import torchvision.transforms as T

from ...core.config.config import Config
from ...core.logging.logger import setup_logging
from ...vision.preprocessor import FramePreprocessor
from ...vision.detector import ObjectDetector

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model training class for pickleball vision."""
    
    def __init__(self, config_path: str):
        """Initialize model trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config()
        self._load_config(config_path)
        
        # Initialize components
        self.preprocessor = FramePreprocessor()
        self.detector = ObjectDetector()
        
    def _load_config(self, config_path: str) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            self.config.load_yaml_config(config_path)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    # ... rest of the existing code ... 