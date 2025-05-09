"""FastAPI application for serving ball detection model."""

import os
from typing import List, Dict, Any
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from ...core.config.config import Config
from ...models.detector import PickleballDetector
from ...utils.preprocessor import FramePreprocessor
from ...utils.visualizer import Visualizer
from ...monitoring.metrics import MetricsCollector

# Initialize FastAPI app
app = FastAPI(
    title="Pickleball Vision API",
    description="API for pickleball ball detection and tracking",
    version="0.1.0"
)

# Initialize components
config = Config()
detector = PickleballDetector(config)
preprocessor = FramePreprocessor(config)
visualizer = Visualizer("output/api")
metrics = MetricsCollector("output/metrics")

class Detection(BaseModel):
    """Detection result schema."""
    frame_id: int
    bbox: List[int]
    confidence: float
    
class DetectionResponse(BaseModel):
    """API response schema."""
    detections: List[Detection]
    metrics: Dict[str, float]
    
@app.post("/detect", response_model=DetectionResponse)
async def detect_ball(video: UploadFile = File(...)):
    """Detect ball in video frames.
    
    Args:
        video: Input video file
        
    Returns:
        List of detections and metrics
    """
    # Save uploaded file
    video_path = Path("temp") / video.filename
    video_path.parent.mkdir(exist_ok=True)
    
    with open(video_path, "wb") as f:
        f.write(await video.read())
        
    try:
        # Process video
        cap = cv2.VideoCapture(str(video_path))
        frame_id = 0
        detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess frame
            frame = preprocessor.preprocess_frame(frame)
            
            # Check quality
            passes_quality, quality_metrics = preprocessor.check_frame_quality(frame)
            
            if passes_quality:
                # Run detection
                detection = detector.detect(frame)
                
                if detection is not None:
                    bbox, confidence = detection
                    detections.append(Detection(
                        frame_id=frame_id,
                        bbox=bbox.tolist(),
                        confidence=float(confidence)
                    ))
                    
                    # Record metrics
                    metrics.record_metric("confidence", confidence)
                    
            frame_id += 1
            
        cap.release()
        
        # Get latest metrics
        response_metrics = {
            "confidence": metrics.get_latest("confidence") or 0.0,
            "frames_processed": frame_id,
            "detections_found": len(detections)
        }
        
        return DetectionResponse(
            detections=detections,
            metrics=response_metrics
        )
        
    finally:
        # Cleanup
        if video_path.exists():
            os.remove(video_path)
            
@app.get("/metrics")
async def get_metrics():
    """Get current metrics."""
    return JSONResponse({
        name: series.values
        for name, series in metrics.metrics.items()
    })
    
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 