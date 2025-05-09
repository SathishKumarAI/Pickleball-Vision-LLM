import cv2
import numpy as np
from typing import Optional
from loguru import logger

class FramePreprocessor:
    """Preprocess video frames for analysis."""
    
    def __init__(self, config):
        """Initialize the frame preprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.prev_frame = None
        self.motion_history = []
        
        logger.info("Initialized FramePreprocessor")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Preprocessed frame
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame input")
        
        # Convert to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Resize if needed
        if self.config.RESIZE_FRAMES:
            frame = cv2.resize(frame, (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT))
        
        # Apply preprocessing steps
        frame = self._normalize(frame)
        frame = self._equalize_histogram(frame)
        frame = self._reduce_noise(frame)
        
        return frame
    
    def adaptive_sampling(self, frame: np.ndarray) -> bool:
        """Check if frame should be processed based on similarity to previous frame.
        
        Args:
            frame: Input frame
        
        Returns:
            True if frame should be processed, False otherwise
        """
        if not self.config.ADAPTIVE_SAMPLING:
            return True
        
        if self.prev_frame is None:
            self.prev_frame = frame
            return True
        
        # Calculate frame similarity
        similarity = self._calculate_frame_similarity(frame, self.prev_frame)
        self.prev_frame = frame
        
        # Update motion history
        self.motion_history.append(similarity)
        if len(self.motion_history) > 10:
            self.motion_history.pop(0)
        
        # Return True if significant change detected
        return similarity > self.config.SAMPLE_THRESHOLD
    
    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame values.
        
        Args:
            frame: Input frame
        
        Returns:
            Normalized frame
        """
        return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    
    def _equalize_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Apply histogram equalization.
        
        Args:
            frame: Input frame
        
        Returns:
            Equalized frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _reduce_noise(self, frame: np.ndarray) -> np.ndarray:
        """Apply noise reduction.
        
        Args:
            frame: Input frame
        
        Returns:
            Denoised frame
        """
        return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    
    def _calculate_frame_similarity(self, frame1: np.ndarray,
                                  frame2: np.ndarray) -> float:
        """Calculate similarity between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
        
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Calculate similarity score
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        return similarity
    
    def reset(self):
        """Reset preprocessor state."""
        self.prev_frame = None
        self.motion_history = [] 