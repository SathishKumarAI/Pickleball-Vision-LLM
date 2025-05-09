# Pickleball Vision System Architecture

## System Overview

```ascii
┌─────────────────────────────────────────────────────────────┐
│                     VideoProcessor                          │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │  Detector   │  │ Visualizer  │  │ FramePreprocessor │   │
│  │  (YOLOv8)   │◄─┤  (Drawing)  │◄─┤  (Preprocessing)  │   │
│  └──────┬──────┘  └─────────────┘  └─────────┬─────────┘   │
│         │                                    │             │
│         ▼                                    ▼             │
│  ┌─────────────┐                    ┌───────────────────┐   │
│  │ CacheManager│                    │    Config        │   │
│  │  (Storage)  │                    │  (Settings)      │   │
│  └─────────────┘                    └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. VideoProcessor
- **Role**: Orchestrates the entire video processing pipeline
- **Key Features**:
  - Frame-by-frame processing
  - Progress tracking and logging
  - Output generation (video, frames, detection logs)
- **Dependencies**: All other components

### 2. PickleballDetector
- **Role**: Handles object detection using YOLOv8
- **Key Features**:
  - GPU acceleration when available
  - Instance segmentation
  - Confidence thresholding
- **Dependencies**: Config, YOLOv8 model

### 3. Visualizer
- **Role**: Renders detection results
- **Key Features**:
  - Bounding box drawing
  - Mask overlays
  - Frame information display
- **Dependencies**: Config

### 4. FramePreprocessor
- **Role**: Prepares frames for detection
- **Key Features**:
  - Resizing
  - Normalization
  - Adaptive sampling
- **Dependencies**: Config

### 5. CacheManager
- **Role**: Manages detection caching
- **Key Features**:
  - TTL-based caching
  - Size-based cleanup
  - Compression support
- **Dependencies**: Config

### 6. Config
- **Role**: Central configuration management
- **Key Features**:
  - YAML-based configuration
  - Runtime settings
  - Directory management
- **Dependencies**: None

## Data Flow

1. **Input**: Video file
2. **Processing**:
   - Frame extraction
   - Preprocessing
   - Detection
   - Visualization
   - Caching
3. **Output**:
   - Annotated video
   - Detection logs
   - Sample frames
   - Masks

## Configuration

The system is configured through `config.yaml` with settings for:
- Model parameters
- Processing options
- Output directories
- Visualization settings
- Cache management
- Error handling

## Next Steps

1. **Feature Additions**:
   - Ball trajectory prediction
   - Player tracking
   - Scoring detection
   - Shot classification

2. **Testing**:
   - Unit tests for each component
   - Integration tests
   - Performance benchmarks

3. **User Interface**:
   - CLI interface
   - Streamlit dashboard
   - Real-time processing

4. **Deployment**:
   - Docker containerization
   - API endpoints
   - Cloud deployment 