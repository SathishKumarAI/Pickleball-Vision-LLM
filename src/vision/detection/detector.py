"""Object detection module for pickleball analysis."""
import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from ultralytics import YOLO
from ...shared.config.config import Config
from ..utils.preprocessor import FramePreprocessor
from collections import defaultdict

logger = logging.getLogger(__name__)

class ObjectDetector:
    """Detects objects in pickleball video frames."""
    
    # Colors for visualization
    COLORS = {
        'person': (0, 255, 0),      # Green for players
        'sports ball': (0, 0, 255),  # Red for ball
        'chair': (255, 0, 0),       # Blue for coach
        'bench': (128, 128, 128),   # Gray for audience
        'scoreboard': (255, 255, 0)  # Yellow for scoreboard
    }
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the object detector.
        
        Args:
            config: Optional configuration object. If None, a new Config instance will be created.
        """
        self.config = config or Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = FramePreprocessor(self.config)
        
        try:
            # Load model
            self.model = self._load_model()
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def _load_model(self) -> YOLO:
        """Load the YOLOv8 model.
        
        Returns:
            Loaded YOLO model
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            model = YOLO(self.config.MODEL_PATH)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detections, each containing:
            - bbox: Bounding box coordinates [x1, y1, x2, y2]
            - confidence: Detection confidence score
            - class_id: Object class ID
            - class_name: Object class name
            
        Raises:
            ValueError: If input frame is invalid
        """
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid input frame")
                
            # Run inference
            results = self.model(frame, verbose=False)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    
                    if conf >= self.config.CONFIDENCE_THRESHOLD:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': class_name
                        })
                    
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise
            
    def filter_detections(self, detections: List[Dict[str, Any]], 
                         min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Filter detections based on confidence threshold.
        
        Args:
            detections: List of detection dictionaries
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of detections
        """
        return [d for d in detections if d['confidence'] >= min_confidence]
        
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection boxes on frame with improved visualization.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with detection boxes drawn
        """
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Get color for class
            color = self.COLORS.get(class_name, (0, 255, 0))
            
            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Add label with better visibility
            label = f"{class_name}: {conf:.2f}"
            font_scale = 0.8
            thickness = 2
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Create label background
            cv2.rectangle(output, 
                        (x1, y1 - label_height - 10),
                        (x1 + label_width, y1),
                        color, -1)  # Filled rectangle
            
            # Add white text
            cv2.putText(output, label,
                      (x1, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      font_scale, (255, 255, 255), thickness)
                       
        return output
        
    def process_video(self, video_path: str, output_dir: str,
                     max_frames: int = 20, frame_skip: int = 1) -> None:
        """Process video file and save frames with detections.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save processed frames
            max_frames: Maximum number of frames to process
            frame_skip: Process every Nth frame
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        frame_count = 0
        processed_count = 0
        detection_stats = defaultdict(lambda: {'count': 0, 'total_conf': 0})
        
        logger.info("Starting video processing...")
        
        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
                
            # Process frame
            detections = self.detect(frame)
            output_frame = self.draw_detections(frame, detections)
            
            # Update statistics
            for det in detections:
                class_name = det['class_name']
                detection_stats[class_name]['count'] += 1
                detection_stats[class_name]['total_conf'] += det['confidence']
            
            # Save frame
            frame_path = output_path / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(frame_path), output_frame)
            
            frame_count += 1
            processed_count += 1
            
            if processed_count % 5 == 0:
                logger.info(f"Processed {processed_count}/{max_frames} frames")
        
        # Cleanup
        cap.release()
        
        # Print statistics
        logger.info("\nDetection Statistics:")
        logger.info("-" * 50)
        for class_name, stats in detection_stats.items():
            avg_conf = stats['total_conf'] / stats['count'] if stats['count'] > 0 else 0
            logger.info(f"{class_name}:")
            logger.info(f"  Total detections: {stats['count']}")
            logger.info(f"  Average confidence: {avg_conf:.2f}")
            logger.info("-" * 50)
        
        logger.info(f"\nProcessing complete. Frames saved to {output_path}")
        logger.info(f"Total frames processed: {processed_count}") 