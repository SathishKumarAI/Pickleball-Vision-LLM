"""
Script to find pickleball videos in the workspace.

This script searches for video files that might contain pickleball footage
and helps prepare them for testing.
"""

import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_videos(root_dir: str = None) -> list:
    """
    Find video files in the workspace.
    
    Args:
        root_dir: Optional root directory to search from
        
    Returns:
        List of paths to video files
    """
    if root_dir is None:
        root_dir = os.getcwd()
        
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    videos = []
    
    for ext in video_extensions:
        videos.extend(Path(root_dir).rglob(f"*{ext}"))
        
    return videos

def main():
    """Main function to find and list videos."""
    logger.info("Searching for video files...")
    
    videos = find_videos()
    
    if not videos:
        logger.warning("No video files found!")
        return
        
    logger.info(f"\nFound {len(videos)} video files:")
    for i, video in enumerate(videos, 1):
        logger.info(f"{i}. {video}")
        
    # Create test directory if needed
    test_dir = Path("test_videos")
    test_dir.mkdir(exist_ok=True)
    
    # Ask user which video to use for testing
    choice = input("\nEnter the number of the video to use for testing (or 'q' to quit): ")
    
    if choice.lower() == 'q':
        return
        
    try:
        video_path = videos[int(choice) - 1]
        logger.info(f"\nSelected video: {video_path}")
        
        # Create symbolic link in test directory
        test_path = test_dir / video_path.name
        if not test_path.exists():
            try:
                os.symlink(video_path, test_path)
                logger.info(f"Created symbolic link: {test_path}")
            except OSError:
                logger.warning("Could not create symbolic link. Copying file instead...")
                import shutil
                shutil.copy2(video_path, test_path)
                logger.info(f"Copied video to: {test_path}")
                
        logger.info(f"\nYou can now use this video for testing:")
        logger.info(f"Path: {test_path}")
        
    except (ValueError, IndexError):
        logger.error("Invalid selection!")

if __name__ == "__main__":
    main() 