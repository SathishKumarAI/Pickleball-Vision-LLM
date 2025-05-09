"""
Script to process a video using YOLOv8 model and save individual frames.
"""

import os
import cv2
import torch
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VIDEO_URL = "https://youtu.be/Osha_slBRkc?si=2QkF-sBK4Ol0PIe6"
MODEL_NAME = "yolov8x.pt"  # Using YOLOv8x for best accuracy
CONFIDENCE_THRESHOLD = 0.3  # Lowered threshold to catch more potential ball detections
FRAME_SKIP = 1  # Process every frame
TARGET_FRAMES = 100  # Process 100 frames
OUTPUT_DIR = Path("src/pickleball_vision/data/frames_v8")  # Changed to v8

# Ball-related classes to track
BALL_CLASSES = ['sports ball', 'tennis ball', 'ball']

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_video(url: str, output_path: Path) -> None:
    """Download video from URL if not already present."""
    if output_path.exists():
        logger.info(f"Video already exists at {output_path}")
        return
    
    try:
        import yt_dlp
        logger.info(f"Downloading video from {url}")
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': str(output_path)
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info(f"Video downloaded successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        raise

def load_model(model_name: str) -> YOLO:
    """Load YOLOv8 model."""
    logger.info(f"Loading YOLOv8 model: {model_name}")
    try:
        model = YOLO(model_name)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def process_frame(frame: np.ndarray, model: YOLO) -> tuple:
    """Process a single frame with YOLOv8 model."""
    try:
        # Run inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        
        # Get detections
        boxes = results.boxes
        detections = []
        ball_detections = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            detection = {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': confidence,
                'class': class_name
            }
            
            detections.append(detection)
            
            # Track ball detections separately
            if class_name.lower() in BALL_CLASSES:
                ball_detections.append(detection)
                logger.info(f"Ball detected: {class_name} with confidence {confidence:.2f}")
        
        return frame, detections, ball_detections
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return frame, [], []

def draw_detections(frame: np.ndarray, detections: list, ball_detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    # Draw regular detections in green
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cls = det['class']
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{cls}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw ball detections in red
    for det in ball_detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cls = det['class']
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw label
        label = f"{cls}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame

def main():
    """Main function to process video."""
    # Setup paths
    video_path = Path("src/pickleball_vision/data/videos/input.mp4")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download video if needed
    download_video(VIDEO_URL, video_path)
    
    # Load model
    model = load_model(MODEL_NAME)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Error opening video file")
        return
    
    # Initialize counters and stats
    frame_count = 0
    processed_count = 0
    detection_stats = defaultdict(lambda: {'count': 0, 'total_conf': 0})
    ball_detection_stats = defaultdict(lambda: {'count': 0, 'total_conf': 0})
    start_time = datetime.now()
    
    try:
        while processed_count < TARGET_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % FRAME_SKIP == 0:
                # Process frame
                processed_frame, detections, ball_detections = process_frame(frame, model)
                
                # Update stats
                for det in detections:
                    cls = det['class']
                    conf = det['confidence']
                    detection_stats[cls]['count'] += 1
                    detection_stats[cls]['total_conf'] += conf
                
                # Update ball detection stats
                for det in ball_detections:
                    cls = det['class']
                    conf = det['confidence']
                    ball_detection_stats[cls]['count'] += 1
                    ball_detection_stats[cls]['total_conf'] += conf
                
                # Draw detections
                processed_frame = draw_detections(processed_frame, detections, ball_detections)
                
                # Save frame
                frame_path = OUTPUT_DIR / f"frame_{processed_count:04d}.jpg"
                cv2.imwrite(str(frame_path), processed_frame)
                
                processed_count += 1
                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count}/{TARGET_FRAMES} frames")
            
            frame_count += 1
        
        # Calculate and log statistics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        fps = processed_count / processing_time
        
        logger.info("\nDetection Statistics:")
        for cls, stats in detection_stats.items():
            avg_conf = stats['total_conf'] / stats['count'] if stats['count'] > 0 else 0
            logger.info(f"{cls}: {stats['count']} detections, avg confidence: {avg_conf:.2f}")
        
        logger.info("\nBall Detection Statistics:")
        for cls, stats in ball_detection_stats.items():
            avg_conf = stats['total_conf'] / stats['count'] if stats['count'] > 0 else 0
            logger.info(f"{cls}: {stats['count']} detections, avg confidence: {avg_conf:.2f}")
        
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"Total frames processed: {processed_count}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Average FPS: {fps:.2f}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
    finally:
        cap.release()
        logger.info("Video processing completed")

if __name__ == "__main__":
    main() 