"""
Unified visualization utilities for the pickleball vision project.

This module provides consistent visualization functions for frames,
detections, tracking, and analysis results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import json

def draw_detection(
    frame: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    score: Optional[float] = None,
    label: Optional[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw detection bounding box and information on frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        score: Detection confidence score
        label: Optional label text
        color: Box color in BGR format
        thickness: Line thickness
        
    Returns:
        Frame with visualizations
    """
    vis_frame = frame.copy()
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Add label and score
        text = []
        if label:
            text.append(label)
        if score is not None:
            text.append(f"{score:.2f}")
            
        if text:
            text = " ".join(text)
            cv2.putText(vis_frame, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            
    return vis_frame

def draw_trajectory(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    max_points: int = 30
) -> np.ndarray:
    """
    Draw trajectory path on frame.
    
    Args:
        frame: Input frame
        points: List of (x, y) coordinates
        color: Line color in BGR format
        thickness: Line thickness
        max_points: Maximum number of points to draw
        
    Returns:
        Frame with trajectory visualization
    """
    vis_frame = frame.copy()
    
    if len(points) < 2:
        return vis_frame
        
    # Draw lines between points
    points = points[-max_points:]  # Keep only recent points
    for i in range(len(points) - 1):
        cv2.line(vis_frame, points[i], points[i+1], color, thickness)
        
    return vis_frame

def display_frames(
    frames: List[np.ndarray],
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Path] = None
) -> None:
    """
    Display multiple frames in a row.
    
    Args:
        frames: List of frames to display
        titles: Optional list of titles for each frame
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    n = len(frames)
    if titles is None:
        titles = [f"Frame {i+1}" for i in range(n)]
        
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
        
    for ax, frame, title in zip(axes, frames, titles):
        if len(frame.shape) == 2:
            ax.imshow(frame, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Metrics Over Time",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot multiple metrics over time.
    
    Args:
        metrics: Dictionary of metric names to lists of values
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    
    for name, values in metrics.items():
        plt.plot(values, label=name)
        
    plt.title(title)
    plt.xlabel("Frame Number")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def save_visualization(
    data: Dict[str, Union[np.ndarray, Dict, List]],
    output_dir: Path,
    prefix: str = ""
) -> Dict[str, Path]:
    """
    Save visualization data to files.
    
    Args:
        data: Dictionary of visualization data
        output_dir: Directory to save files
        prefix: Optional prefix for filenames
        
    Returns:
        Dictionary mapping data keys to saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # Save image
            path = output_dir / f"{prefix}{key}.jpg"
            cv2.imwrite(str(path), value)
            saved_paths[key] = path
            
        elif isinstance(value, (dict, list)):
            # Save JSON data
            path = output_dir / f"{prefix}{key}.json"
            with open(path, 'w') as f:
                json.dump(value, f, indent=2)
            saved_paths[key] = path
            
    return saved_paths 