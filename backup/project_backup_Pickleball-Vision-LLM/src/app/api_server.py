from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any
import cv2
import numpy as np
from src.data_collection.frame_sampler import sample_frames_from_video
from src.models.pose_extractor import extract_pose_from_frame
from src.models.llm_clip_integration import analyze_gameplay_with_llm
from src.utils.visualizer import draw_pose_on_frame
import uvicorn

# Import your custom modules (adjust paths as needed)

app = FastAPI(title="Pickleball Vision API",
             description="API for analyzing pickleball gameplay videos using AI",
             version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to Pickleball Vision API"}

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze uploaded pickleball video and return insights
    """
    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            video_path = tmp_file.name

        # Process video
        results = {
            "frames": [],
            "poses": [],
            "analysis": None
        }

        # Sample frames
        sampled_frames = sample_frames_from_video(video_path, every_nth=30)
        
        # Extract poses
        for frame in sampled_frames:
            pose = extract_pose_from_frame(frame)
            vis_frame = draw_pose_on_frame(frame.copy(), pose)
            
            # Convert frames to base64 for API response
            _, buffer = cv2.imencode('.jpg', vis_frame)
            frame_b64 = buffer.tobytes().hex()
            
            results["frames"].append(frame_b64)
            results["poses"].append(pose)

        # Run LLM analysis
        llm_results = analyze_gameplay_with_llm(video_path)
        results["analysis"] = llm_results

        # Cleanup
        os.unlink(video_path)
        
        return results

    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)