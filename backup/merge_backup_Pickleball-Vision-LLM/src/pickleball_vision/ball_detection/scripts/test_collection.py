"""
Test script for data collection pipeline.

This script tests the data collection pipeline with a sample video
and verifies the quality of extracted frames.
"""

import os
import yaml
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import unified utilities
from pickleball_vision.utils.logging_utils import setup_logging, get_log_file_path, log_progress
from pickleball_vision.utils.visualization_utils import draw_detection, display_frames, plot_metrics, save_visualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_frame_distribution(metadata: dict, output_dir: Path):
    """
    Analyze and plot frame distribution metrics.
    
    Args:
        metadata: Frame metadata dictionary
        output_dir: Directory to save plots
    """
    # Create quality score histogram
    plt.figure(figsize=(10, 6))
    plt.hist(metadata['quality_scores'], bins=20, alpha=0.7)
    plt.title('Frame Quality Score Distribution')
    plt.xlabel('Quality Score')
    plt.ylabel('Number of Frames')
    plt.savefig(output_dir / 'quality_distribution.png')
    plt.close()
    
    # Create motion score histogram
    plt.figure(figsize=(10, 6))
    plt.hist(metadata['motion_scores'], bins=20, alpha=0.7)
    plt.title('Motion Score Distribution')
    plt.xlabel('Motion Score')
    plt.ylabel('Number of Frames')
    plt.savefig(output_dir / 'motion_distribution.png')
    plt.close()

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_video(video_path: str, config: Dict) -> Dict:
    """Process video and extract frames based on configuration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    frame_metrics = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate frame metrics
        metrics = calculate_frame_metrics(frame)
        frame_metrics.append(metrics)

        # Check if frame meets quality criteria
        if (metrics['brightness'] > config['quality_thresholds']['brightness'] and
            metrics['contrast'] > config['quality_thresholds']['contrast'] and
            metrics['blur'] < config['quality_thresholds']['blur']):
            frames.append(frame)
            frame_count += 1

        if frame_count >= config['max_frames_per_video']:
            break

    cap.release()
    return {
        'frames': frames,
        'metrics': frame_metrics,
        'total_processed': len(frame_metrics),
        'frames_kept': len(frames)
    }

def calculate_frame_metrics(frame: np.ndarray) -> Dict:
    """Calculate quality metrics for a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {
        'brightness': brightness,
        'contrast': contrast,
        'blur': blur
    }

def test_data_collection(video_path: str, config_path: str):
    """
    Test the data collection pipeline with a sample video.
    
    Args:
        video_path: Path to test video
        config_path: Path to configuration file
    """
    logger.info(f"Testing data collection with video: {video_path}")
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Process video
        logger.info("Processing video...")
        result = process_video(video_path, config)
        
        # Calculate statistics
        logger.info("Calculating statistics...")
        stats = {
            'total_frames': result['total_processed'],
            'frames_kept': result['frames_kept'],
            'quality': {
                'mean': float(np.mean([m['brightness'] for m in result['metrics']])),
                'std': float(np.std([m['brightness'] for m in result['metrics']])),
                'min': float(np.min([m['brightness'] for m in result['metrics']])),
                'max': float(np.max([m['brightness'] for m in result['metrics']])),
                'median': float(np.median([m['brightness'] for m in result['metrics']]))
            }
        }
        
        # Print detailed statistics
        logger.info("\nDetailed Statistics:")
        logger.info(f"Total frames processed: {stats['total_frames']}")
        logger.info(f"Frames kept: {stats['frames_kept']}")
        logger.info("\nQuality Scores:")
        logger.info(f"  Mean: {stats['quality']['mean']:.2f}")
        logger.info(f"  Std: {stats['quality']['std']:.2f}")
        logger.info(f"  Min: {stats['quality']['min']:.2f}")
        logger.info(f"  Max: {stats['quality']['max']:.2f}")
        logger.info(f"  Median: {stats['quality']['median']:.2f}")
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("test_results") / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\nCreated results directory: {results_dir}")
        
        # Save frames
        logger.info("Saving sample frames...")
        for i, frame in enumerate(result['frames'][:5]):  # Save first 5 frames
            frame_path = results_dir / f"frame_{i+1}.jpg"
            cv2.imwrite(str(frame_path), frame)
            logger.info(f"Saved frame to: {frame_path}")
        
        # Generate and save plots
        logger.info("Generating analysis plots...")
        plot_metrics(
            {
                'Brightness': [m['brightness'] for m in result['metrics']],
                'Contrast': [m['contrast'] for m in result['metrics']],
                'Blur': [m['blur'] for m in result['metrics']]
            },
            title="Frame Quality Metrics",
            save_path=results_dir / 'metrics_plot.jpg'
        )
        
        # Save detailed results
        results = {
            'video_path': str(video_path),
            'timestamp': timestamp,
            'statistics': stats,
            'config': config
        }
        
        results_file = results_dir / "test_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f)
        logger.info(f"Saved test results to: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during test execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    # Setup logging
    logger = setup_logging(
        'test_collection',
        log_file=get_log_file_path('ball_detection')
    )

    # Load configuration
    config_path = Path('config/data_collection.yaml')
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return

    config = load_config(config_path)
    logger.info("Configuration loaded successfully")

    # Process video
    video_path = "test_videos/sample_pickleball.mp4"
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return

    logger.info(f"Processing video: {video_path}")
    results = process_video(video_path, config)

    # Log results
    logger.info(f"Total frames processed: {results['total_processed']}")
    logger.info(f"Frames kept: {results['frames_kept']}")

    # Save results
    output_dir = Path('test_results')
    output_dir.mkdir(exist_ok=True)

    # Save frames
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(results['frames']):
        cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), frame)

    # Save metrics
    metrics_path = output_dir / 'metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.dump(results['metrics'], f)

    # Visualize results
    if results['frames']:
        sample_frames = results['frames'][:5]  # First 5 frames
        display_frames(
            sample_frames,
            titles=[f"Frame {i}" for i in range(len(sample_frames))],
            save_path=output_dir / 'sample_frames.jpg'
        )

    # Plot metrics
    plot_metrics(
        {
            'Brightness': [m['brightness'] for m in results['metrics']],
            'Contrast': [m['contrast'] for m in results['metrics']],
            'Blur': [m['blur'] for m in results['metrics']]
        },
        title="Frame Quality Metrics",
        save_path=output_dir / 'metrics_plot.jpg'
    )

    logger.info("Test collection completed successfully")

if __name__ == "__main__":
    main() 