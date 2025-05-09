"""
Script to download a sample pickleball video for testing.

This script downloads a sample pickleball video from a public source
and prepares it for testing the data collection pipeline.
"""

import os
import requests
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample video URL (pickleball match)
SAMPLE_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with actual pickleball video URL

def download_file(url: str, output_path: Path):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
    """
    try:
        # Use yt-dlp to download YouTube video
        import yt_dlp
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': str(output_path),
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
    except ImportError:
        logger.error("yt-dlp not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'yt-dlp'])
        
        # Try again after installation
        import yt_dlp
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': str(output_path),
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

def main():
    """Main function to download sample video."""
    # Create test directory
    test_dir = Path("test_videos")
    test_dir.mkdir(exist_ok=True)
    
    output_path = test_dir / "sample_pickleball.mp4"
    
    if output_path.exists():
        logger.info("Sample video already exists!")
        return
        
    logger.info("Downloading sample pickleball video...")
    try:
        download_file(SAMPLE_VIDEO_URL, output_path)
        logger.info(f"Downloaded sample video to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to download sample video: {e}")
        logger.info("\nPlease manually download a pickleball video and place it in the test_videos directory.")
        logger.info("You can find pickleball videos on platforms like YouTube or sports websites.")

if __name__ == "__main__":
    main() 