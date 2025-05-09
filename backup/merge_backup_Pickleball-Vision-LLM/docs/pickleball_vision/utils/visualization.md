# Visualization Utilities

The `Visualizer` class provides tools for visualizing frames, detections, and metrics.

## Overview

The visualizer supports:
- Drawing bounding boxes and labels
- Displaying multiple frames in a grid
- Plotting metrics over time
- Saving visualizations to files

## Usage

```python
from pickleball_vision.utils.visualizer import Visualizer

# Initialize with output directory
visualizer = Visualizer("output/visualizations")

# Draw detection on frame
bbox = (x1, y1, x2, y2)
frame_with_detection = visualizer.draw_detection(
    frame, bbox, label="ball", score=0.95
)

# Display multiple frames
frames = [frame1, frame2, frame3]
titles = ["Original", "Preprocessed", "Detection"]
visualizer.display_frames(frames, titles, save_path="frames.png")

# Plot metrics
metrics = {
    "confidence": confidence_scores,
    "quality": quality_scores
}
visualizer.plot_metrics(metrics, save_path="metrics.png")
```

## Detection Visualization

The `draw_detection` method:
- Draws bounding boxes in green
- Adds optional label text
- Shows confidence score if provided
- Returns modified frame

## Frame Display

The `display_frames` method:
- Arranges frames in a grid (max 4 columns)
- Supports optional frame titles
- Can save to file or display interactively
- Automatically handles color space conversion

## Metrics Plotting

The `plot_metrics` method:
- Creates line plots for multiple metrics
- Adds legend and grid
- Labels axes automatically
- Supports saving to file

## Saving Visualizations

The `save_visualization` method:
- Handles both images and JSON data
- Creates output directory if needed
- Uses standard file formats (PNG for images)
- Preserves data structure in JSON 