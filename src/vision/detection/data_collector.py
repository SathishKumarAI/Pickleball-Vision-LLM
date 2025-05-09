"""
Data collection utilities for ball detection.

This module provides functionality for extracting and preprocessing frames
from video files, with quality assessment and motion detection.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List
import yaml

logger = logging.getLogger(__name__)

class DataCollector:
    """Class for collecting and preprocessing video frames."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the data collector.
        
        Args:
            config_path: Path to YAML config file with collection settings
        """
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            'frame_size': (1280, 1280),
            'quality_thresholds': {
                'brightness': 0.2,
                'contrast': 30.0,
                'blur': 100.0
            },
            'motion_thresholds': {
                'min_flow': 0.5,
                'max_flow': 10.0
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return {**default_config, **config}
        return default_config
        
    def process_video(self, video_path: str) -> Dict:
        """
        Process video file and extract high-quality frames.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Dict containing processed frames and metadata
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        frames = []
        metadata = {
            'total_frames': 0,
            'quality_scores': [],
            'motion_scores': [],
            'selected_frames': []
        }
        
        prev_frame = None
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            frame = cv2.resize(frame, self.config['frame_size'])
            
            # Calculate quality metrics
            quality_score = self._assess_quality(frame)
            motion_score = self._detect_motion(frame, prev_frame) if prev_frame is not None else 0.0
            
            metadata['quality_scores'].append(quality_score)
            metadata['motion_scores'].append(motion_score)
            
            # Store frame if it meets quality criteria
            if self._check_frame_quality(quality_score, motion_score):
                frames.append(frame)
                metadata['selected_frames'].append(frame_idx)
                
            prev_frame = frame.copy()
            frame_idx += 1
            metadata['total_frames'] = frame_idx
            
        cap.release()
        return {'frames': frames, 'metadata': metadata}
        
    def _assess_quality(self, frame: np.ndarray) -> float:
        """
        Assess frame quality based on brightness, contrast and blur.
        
        Args:
            frame: Input frame
            
        Returns:
            Quality score between 0 and 1
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness
        brightness = np.mean(gray) / 255.0
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Calculate blur (Laplacian variance)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Combine metrics into single score
        thresholds = self.config['quality_thresholds']
        brightness_score = brightness > thresholds['brightness']
        contrast_score = contrast > thresholds['contrast']
        blur_score = blur > thresholds['blur']
        
        return float(brightness_score and contrast_score and blur_score)
        
    def _detect_motion(self, frame: np.ndarray, prev_frame: np.ndarray) -> float:
        """
        Detect motion between consecutive frames using optical flow.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame
            
        Returns:
            Motion score between 0 and 1
        """
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate magnitude of flow vectors
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_flow = np.mean(magnitude)
        
        # Normalize flow score
        thresholds = self.config['motion_thresholds']
        if mean_flow < thresholds['min_flow']:
            return 0.0
        elif mean_flow > thresholds['max_flow']:
            return 1.0
        else:
            return (mean_flow - thresholds['min_flow']) / (thresholds['max_flow'] - thresholds['min_flow'])
            
    def _check_frame_quality(self, quality_score: float, motion_score: float) -> bool:
        """Check if frame meets quality criteria."""
        return quality_score > 0.5 and motion_score > 0.2 