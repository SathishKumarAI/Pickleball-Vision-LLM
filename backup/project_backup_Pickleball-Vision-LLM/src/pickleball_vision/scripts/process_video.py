import cv2
import torch
import numpy as np
import yt_dlp
import logging
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
VIDEO_URL = "https://youtu.be/Osha_slBRkc?si=2QkF-sBK4Ol0PIe6"
OUTPUT_DIR = Path("src/pickleball_vision/data")
MODEL_NAME = "yolov8x.pt"  # Using YOLOv8x for best accuracy
CONFIDENCE_THRESHOLD = 0.4
FRAME_SKIP = 1
TARGET_FRAMES = 20

# Colors for visualization
COLORS = {
    'person': (0, 255, 0),    # Green for players
    'sports ball': (0, 0, 255),  # Red for ball
    'chair': (255, 0, 0),     # Blue for coach/referee
    'bench': (128, 128, 128), # Gray for audience
    'scoreboard': (0, 255, 255)  # Yellow for scoreboard
}

def download_video(url: str, output_path: Path) -> Path:
    """Download video from YouTube URL."""
    if output_path.exists():
        logger.info(f"Video already exists at {output_path}")
        return output_path
    
    logger.info(f"Downloading video from {url}")
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': str(output_path)
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def load_model(model_name: str) -> YOLO:
    """Load YOLOv8 model."""
    logger.info(f"Loading YOLOv8 model: {model_name}")
    model = YOLO(model_name)
    return model

def process_frame(frame, model, detection_stats):
    """Process a single frame with YOLOv8."""
    # Run inference
    results = model(frame, verbose=False)
    
    # Process detections
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue
                
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Update detection statistics
            detection_stats[class_name]['count'] += 1
            detection_stats[class_name]['total_confidence'] += confidence
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw bounding box
            color = COLORS.get(class_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2)
            })
    
    return frame, detections

def main():
    # Create output directories
    output_dir = OUTPUT_DIR / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download video
    video_path = OUTPUT_DIR / "videos" / "input.mp4"
    video_path = download_video(VIDEO_URL, video_path)
    
    # Load model
    model = load_model(MODEL_NAME)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Error opening video file")
        return
    
    # Initialize detection statistics
    detection_stats = defaultdict(lambda: {'count': 0, 'total_confidence': 0.0})
    
    # Process frames
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    logger.info("Starting video processing...")
    
    while cap.isOpened() and processed_count < TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue
            
        # Process frame
        processed_frame, detections = process_frame(frame, model, detection_stats)
        
        # Save frame
        frame_path = output_dir / f"frame_{processed_count:04d}.jpg"
        cv2.imwrite(str(frame_path), processed_frame)
        
        processed_count += 1
        if processed_count % 5 == 0:
            logger.info(f"Processed {processed_count}/{TARGET_FRAMES} frames")
    
    # Calculate performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    fps = processed_count / total_time
    
    # Print detection statistics
    logger.info("\nDetection Statistics:")
    logger.info("-" * 50)
    for class_name, stats in detection_stats.items():
        if stats['count'] > 0:
            avg_confidence = stats['total_confidence'] / stats['count']
            logger.info(f"{class_name}:")
            logger.info(f"  Total detections: {stats['count']}")
            logger.info(f"  Average confidence: {avg_confidence:.2f}")
            logger.info("-" * 50)
    
    # Print performance metrics
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"Total frames processed: {processed_count}")
    logger.info(f"Processing time: {total_time:.2f} seconds")
    logger.info(f"Average FPS: {fps:.2f}")
    logger.info(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Cleanup
    cap.release()
    logger.info(f"\nProcessing complete. Frames saved to {output_dir}")

if __name__ == "__main__":
    main() 