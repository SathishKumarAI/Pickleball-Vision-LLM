# Configuration System

The Pickleball Vision project uses a centralized configuration management system through the `Config` class.

## Overview

The configuration system is designed to:
- Load settings from YAML files
- Provide easy access to configuration values
- Support default values for missing settings
- Organize settings by component (frame extraction, quality thresholds, etc.)

## Usage

```python
from pickleball_vision.core.config.config import Config

# Load default configuration
config = Config()

# Load specific configuration file
config = Config("path/to/config.yaml")

# Access configuration values
frame_size = config.get("frame_size", (1280, 720))
quality_threshold = config.quality_thresholds["brightness"]
```

## Configuration Properties

The `Config` class provides the following property accessors:

### Frame Extraction
```python
frame_extraction = config.frame_extraction
```
Settings for video frame extraction:
- `frame_size`: Target frame dimensions
- `min_frames_per_video`: Minimum frames to extract
- `max_frames_per_video`: Maximum frames to extract

### Quality Thresholds
```python
quality_thresholds = config.quality_thresholds
```
Frame quality settings:
- `brightness`: Minimum average pixel value
- `contrast`: Minimum standard deviation
- `blur`: Minimum Laplacian variance

### Motion Detection
```python
motion_detection = config.motion_detection
```
Motion analysis settings:
- `min_flow`: Minimum optical flow magnitude
- `max_flow`: Maximum optical flow magnitude

### Output
```python
output = config.output
```
Output directory settings:
- `frames_dir`: Directory for saved frames
- `metrics_dir`: Directory for metrics data 