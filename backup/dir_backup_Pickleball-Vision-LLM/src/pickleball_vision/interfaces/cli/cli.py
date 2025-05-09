import click
import os
from pathlib import Path
from .config.config import Config
from .processors.video_processor import VideoProcessor

@click.group()
def cli():
    """Pickleball Vision System CLI"""
    pass

@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--frame-skip', '-f', type=int, help='Number of frames to skip')
@click.option('--max-frames', '-m', type=int, help='Maximum frames to process')
@click.option('--confidence', '-t', type=float, help='Detection confidence threshold')
def process(video_path, config, output_dir, frame_skip, max_frames, confidence):
    """Process a video file for pickleball detection."""
    # Load config
    config_path = config or 'config/config.yaml'
    cfg = Config(config_path)
    
    # Override config with CLI options
    if output_dir:
        cfg.OUTPUT_DIR = output_dir
    if frame_skip:
        cfg.FRAME_SKIP = frame_skip
    if max_frames:
        cfg.MAX_FRAMES = max_frames
    if confidence:
        cfg.CONFIDENCE_THRESHOLD = confidence
    
    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Process video
    processor = VideoProcessor(cfg)
    processor.process_video(video_path)

@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
def extract_frames(video_path, output_dir):
    """Extract frames from a video file."""
    import cv2
    
    # Setup output directory
    output_dir = output_dir or 'data/frames'
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame
        frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            click.echo(f'Extracted {frame_count} frames...')
    
    cap.release()
    click.echo(f'Extracted {frame_count} frames to {output_dir}')

@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
def analyze(video_path, output_dir):
    """Analyze video and generate statistics."""
    from .utils.analyzer import VideoAnalyzer
    
    # Setup output directory
    output_dir = output_dir or 'data/analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    cfg = Config()
    
    # Analyze video
    analyzer = VideoAnalyzer(cfg)
    stats = analyzer.analyze_video(video_path)
    
    # Save statistics
    import json
    stats_path = os.path.join(output_dir, 'video_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    click.echo(f'Analysis saved to {stats_path}')

if __name__ == '__main__':
    cli() 