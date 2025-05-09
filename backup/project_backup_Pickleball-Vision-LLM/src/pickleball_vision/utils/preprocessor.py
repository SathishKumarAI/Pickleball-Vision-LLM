"""Frame preprocessing utilities for video analysis."""

import cv2
import numpy as np
from typing import Tuple, Dict, Any

class FramePreprocessor:
    """Class for preprocessing video frames."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary with quality thresholds
        """
        self.config = config
        self.quality_thresholds = config.get('quality_thresholds', {})
        self.motion_thresholds = config.get('motion_detection', {})
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Preprocessed frame
        """
        # Resize if needed
        if 'frame_size' in self.config:
            frame = cv2.resize(frame, tuple(self.config['frame_size']))
            
        # Convert to RGB
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
        return frame
        
    def check_frame_quality(self, frame: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """Check if frame meets quality thresholds.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (passes_quality_check, quality_metrics)
        """
        metrics = {}
        
        # Brightness
        brightness = np.mean(frame)
        metrics['brightness'] = brightness
        
        # Contrast
        contrast = np.std(frame)
        metrics['contrast'] = contrast
        
        # Blur detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['blur'] = blur
        
        # Check thresholds
        passes_quality = (
            brightness >= self.quality_thresholds.get('brightness', 0) and
            contrast >= self.quality_thresholds.get('contrast', 0) and
            blur >= self.quality_thresholds.get('blur', 0)
        )
        
        return passes_quality, metrics
        
    def compute_motion_score(self, frame: np.ndarray, prev_frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute motion score between consecutive frames.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame
            
        Returns:
            Tuple of (motion_score, flow_visualization)
        """
        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, 
            flags=0
        )
        
        # Compute motion score
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = np.mean(magnitude)
        
        # Create flow visualization
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = cv2.cartToPolar(flow[..., 0], flow[..., 1])[1] * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return motion_score, flow_vis 