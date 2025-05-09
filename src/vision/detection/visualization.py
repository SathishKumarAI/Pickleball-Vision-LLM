"""
Visualization utilities for ball detection.

This module provides functions for visualizing frames, detections,
and analysis results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def draw_detection(frame: np.ndarray, bbox: Tuple[int, int, int, int] = None,
                  score: float = None, color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box and score on frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        score: Detection confidence score
        color: Box color in BGR format
        thickness: Line thickness
        
    Returns:
        Frame with visualizations
    """
    vis_frame = frame.copy()
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
        
        if score is not None:
            text = f"{score:.2f}"
            cv2.putText(vis_frame, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            
    return vis_frame
    
def display_frames(frames: List[np.ndarray], titles: List[str] = None,
                  figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Display multiple frames in a row.
    
    Args:
        frames: List of frames to display
        titles: Optional list of titles for each frame
        figsize: Figure size (width, height)
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
    plt.show()
    
def plot_scores(quality_scores: List[float], motion_scores: List[float],
                figsize: Tuple[int, int] = (10, 5)) -> None:
    """
    Plot quality and motion scores over time.
    
    Args:
        quality_scores: List of frame quality scores
        motion_scores: List of motion detection scores
        figsize: Figure size (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    frames = range(len(quality_scores))
    ax1.plot(frames, quality_scores, 'b-', label='Quality')
    ax1.set_ylabel('Quality Score')
    ax1.set_title('Frame Quality Over Time')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(frames, motion_scores, 'r-', label='Motion')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Motion Score')
    ax2.set_title('Motion Detection Over Time')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show() 