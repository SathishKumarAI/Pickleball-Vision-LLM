"""Frame preprocessing module for pickleball analysis."""
import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging
from ..core.config.config import Config

logger = logging.getLogger(__name__)

class FramePreprocessor:
    """Preprocesses video frames for analysis."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the frame preprocessor.
        
        Args:
            config: Optional configuration object. If None, a new Config instance will be created.
        """
        self.config = config or Config()
        self._setup_preprocessing()
    
    def _setup_preprocessing(self):
        """Setup preprocessing parameters from config."""
        if not self.config.is_preprocessing_enabled:
            return
            
        self.resize_dims = self.config.PREPROCESSING['augmentation'].get('resize_dims', None)
        self.normalize = 'normalize' in self.config.PREPROCESSING['augmentation'].get('methods', [])
        if self.normalize:
            self.normalize_params = self.config.PREPROCESSING['augmentation'].get('normalize_params', {})
            self.mean = np.array(self.normalize_params.get('mean', [0.485, 0.456, 0.406]))
            self.std = np.array(self.normalize_params.get('std', [0.229, 0.224, 0.225]))
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a frame for analysis.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Preprocessed frame
            
        Raises:
            ValueError: If input frame is invalid
        """
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid input frame")
                
            # Convert to RGB if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                
            # Resize if needed
            if frame.shape[:2] != self.config.INPUT_SIZE:
                frame = cv2.resize(frame, self.config.INPUT_SIZE)
                
            # Normalize
            frame = frame.astype(np.float32) / 255.0
            
            return frame
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {str(e)}")
            raise
            
    def adaptive_sampling(self, frame: np.ndarray, 
                         prev_frame: Optional[np.ndarray] = None) -> bool:
        """Determine if frame should be processed based on motion.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame for comparison
            
        Returns:
            True if frame should be processed, False otherwise
        """
        if prev_frame is None:
            return True
            
        try:
            # Convert to grayscale
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate motion
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate motion magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            mean_motion = np.mean(magnitude)
            
            return mean_motion > self.config.MOTION_THRESHOLD
            
        except Exception as e:
            logger.error(f"Error in adaptive sampling: {str(e)}")
            return True
            
    def extract_roi(self, frame: np.ndarray, 
                   bbox: List[int]) -> np.ndarray:
        """Extract region of interest from frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Extracted region of interest
        """
        try:
            x1, y1, x2, y2 = bbox
            return frame[y1:y2, x1:x2]
        except Exception as e:
            logger.error(f"Error extracting ROI: {str(e)}")
            raise
            
    def apply_augmentation(self, frame: np.ndarray) -> np.ndarray:
        """Apply data augmentation to frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Augmented frame
        """
        try:
            # Random horizontal flip
            if np.random.random() < 0.5:
                frame = cv2.flip(frame, 1)
                
            # Random brightness adjustment
            if np.random.random() < 0.5:
                alpha = 1.0 + np.random.uniform(-0.2, 0.2)
                frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
                
            # Random contrast adjustment
            if np.random.random() < 0.5:
                alpha = 1.0 + np.random.uniform(-0.2, 0.2)
                beta = np.random.uniform(-10, 10)
                frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
                
            return frame
            
        except Exception as e:
            logger.error(f"Error applying augmentation: {str(e)}")
            raise
        
    def detect_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Detect motion between frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Motion score between 0 and 1
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate motion score
        motion_score = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
        
        return motion_score
        
    def resize_frame(self, frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize frame to specified dimensions.
        
        Args:
            frame: Input frame
            size: Target size (width, height)
            
        Returns:
            Resized frame
        """
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame values to [0, 1].
        
        Args:
            frame: Input frame
            
        Returns:
            Normalized frame
        """
        return frame.astype(np.float32) / 255.0 