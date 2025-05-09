"""Setup script for initializing data directory structure and downloading required files."""
import os
from pathlib import Path
import shutil
import logging
import yt_dlp
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "pickleball_vision" / "data"
VIDEO_URL = "https://youtu.be/Osha_slBRkc?si=2QkF-sBK4Ol0PIe6"
MODEL_NAME = "yolov8x.pt"

def create_directory_structure():
    """Create the required directory structure."""
    dirs = [
        DATA_DIR / "videos",
        DATA_DIR / "frames",
        DATA_DIR / "models",
        DATA_DIR / "temp"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def download_video():
    """Download the sample video."""
    output_path = DATA_DIR / "videos" / "input.mp4"
    
    if output_path.exists():
        logger.info(f"Video already exists at {output_path}")
        return output_path
    
    logger.info(f"Downloading video from {VIDEO_URL}")
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': str(output_path)
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([VIDEO_URL])
    
    return output_path

def download_model():
    """Download the YOLOv8 model."""
    model_path = DATA_DIR / "models" / MODEL_NAME
    
    if model_path.exists():
        logger.info(f"Model already exists at {model_path}")
        return model_path
    
    logger.info(f"Downloading YOLOv8 model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    shutil.copy(MODEL_NAME, model_path)
    
    return model_path

def main():
    """Main setup function."""
    logger.info("Starting data directory setup...")
    
    # Create directory structure
    create_directory_structure()
    
    # Download required files
    video_path = download_video()
    model_path = download_model()
    
    logger.info("\nSetup complete!")
    logger.info(f"Video saved to: {video_path}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"\nData directory structure created at: {DATA_DIR}")

if __name__ == "__main__":
    main() 