# Frame Preprocessing

The `FramePreprocessor` class provides utilities for preprocessing video frames before ball detection.

## Overview

The preprocessor handles:
- Frame resizing
- Color space conversion
- Quality assessment
- Motion detection

## Usage

```python
from pickleball_vision.utils.preprocessor import FramePreprocessor

# Initialize with configuration
preprocessor = FramePreprocessor(config)

# Preprocess a frame
processed_frame = preprocessor.preprocess_frame(frame)

# Check frame quality
passes_quality, metrics = preprocessor.check_frame_quality(frame)

# Compute motion between frames
motion_score, flow_vis = preprocessor.compute_motion_score(frame, prev_frame)
```

## Frame Quality Metrics

The quality check returns several metrics:

### Brightness
- Computed as mean pixel value
- Higher values indicate brighter frames
- Threshold defined in configuration

### Contrast
- Computed as pixel value standard deviation
- Higher values indicate more contrast
- Threshold defined in configuration

### Blur Detection
- Uses Laplacian variance
- Higher values indicate sharper images
- Threshold defined in configuration

## Motion Detection

The motion detection system:
- Uses Farneback optical flow
- Computes flow magnitude
- Provides flow visualization
- Returns average motion score

### Parameters
- `pyr_scale`: Image pyramid scale
- `levels`: Number of pyramid levels
- `winsize`: Average window size
- `iterations`: Number of iterations
- `poly_n`: Polynomial expansion neighborhood
- `poly_sigma`: Gaussian sigma for polynomial expansion 