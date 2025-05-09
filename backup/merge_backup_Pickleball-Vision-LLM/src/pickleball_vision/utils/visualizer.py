"""Visualization utilities for displaying and saving results."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

class Visualizer:
    """Class for visualizing frames and detections."""
    
    def __init__(self, output_dir: str):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def draw_detection(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                      label: str = "", score: float = None) -> np.ndarray:
        """Draw bounding box and label on frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            label: Optional label text
            score: Optional confidence score
            
        Returns:
            Frame with detection visualization
        """
        x1, y1, x2, y2 = bbox
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label and score
        if label or score is not None:
            label_text = label
            if score is not None:
                label_text += f" {score:.2f}"
                
            cv2.putText(frame, label_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                       
        return frame
        
    def display_frames(self, frames: List[np.ndarray], titles: Optional[List[str]] = None,
                      save_path: Optional[str] = None) -> None:
        """Display multiple frames in a grid.
        
        Args:
            frames: List of frames to display
            titles: Optional list of titles for each frame
            save_path: Optional path to save the visualization
        """
        n = len(frames)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        
        plt.figure(figsize=(4*cols, 4*rows))
        
        for i, frame in enumerate(frames):
            plt.subplot(rows, cols, i+1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            if titles and i < len(titles):
                plt.title(titles[i])
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(str(self.output_dir / save_path))
        else:
            plt.show()
            
        plt.close()
        
    def plot_metrics(self, metrics: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
        """Plot multiple metrics over time.
        
        Args:
            metrics: Dictionary of metric name to list of values
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        for name, values in metrics.items():
            plt.plot(values, label=name)
            
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.title('Metrics Over Time')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(str(self.output_dir / save_path))
        else:
            plt.show()
            
        plt.close()
        
    def save_visualization(self, data: Any, filename: str) -> None:
        """Save visualization data to file.
        
        Args:
            data: Data to save (image or JSON)
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        if isinstance(data, np.ndarray):
            cv2.imwrite(str(output_path), data)
        else:
            import json
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2) 