# Ball Detection Implementation - Phase 1

## Overview
The first phase of ball detection focuses on collecting and preprocessing high-quality training data for pickleball detection. This phase implements sophisticated frame extraction and preprocessing techniques to ensure optimal data quality for model training.

## Components

### 1. Data Collection Script (`collect_data.py`)
The script implements several key features for extracting high-quality frames:

#### Frame Quality Assessment
- **Brightness Analysis**: Ensures frames are well-lit (30-225 range)
- **Contrast Check**: Validates frame contrast (threshold: 20)
- **Blur Detection**: Removes blurry frames using Laplacian variance
- **Quality Scoring**: Combines metrics into a 0-100 score

#### Motion Detection
- Uses Optical Flow to detect frame-to-frame motion
- Helps identify frames with active play
- Configured for small, fast-moving objects
- Parameters optimized for pickleball speed

#### Frame Preprocessing
- **Resolution**: Standardizes to 1280x1280
- **Contrast Enhancement**: CLAHE algorithm for better detail
- **Noise Reduction**: Non-local means denoising
- **Format Standardization**: Consistent output format

### 2. Configuration System
The `data_collection.yaml` file provides flexible configuration:

```yaml
# Key Configuration Parameters
video_dir: "data/raw/videos"
output_dir: "data/processed"
max_frames_per_video: 1000
motion_threshold: 0.1
```

### 3. Metadata Tracking
- Tracks frame quality scores
- Records motion detection results
- Maintains processing history
- Enables dataset analysis

### 4. Testing Framework
A comprehensive testing framework has been implemented to evaluate the data collection pipeline. See [Ball Detection Testing Framework](ball_detection_testing.md) for detailed documentation.

Key testing components:
- Individual test cases with detailed metrics
- Test suite with multiple configurations
- Video file management utilities
- Results visualization and analysis

## Directory Structure
```
ball_detection/
├── data/
│   ├── raw/          # Original videos
│   └── processed/    # Extracted frames
├── config/
│   └── data_collection.yaml
├── scripts/
│   ├── collect_data.py
│   ├── test_collection.py
│   ├── run_test_suite.py
│   └── find_videos.py
└── test_results/     # Test output directory
```

## Usage

1. **Setup Configuration**
   ```bash
   # Create necessary directories
   mkdir -p data/raw/videos data/processed
   
   # Copy configuration file
   cp config/data_collection.yaml config/data_collection.local.yaml
   ```

2. **Prepare Videos**
   - Place pickleball videos in `data/raw/videos`
   - Supported formats: MP4, AVI, MOV

3. **Run Data Collection**
   ```bash
   python scripts/collect_data.py
   ```

4. **Test the Pipeline**
   ```bash
   # Run individual test
   python scripts/test_collection.py
   
   # Run full test suite
   python scripts/run_test_suite.py
   ```

5. **Output**
   - Processed frames in `data/processed/frames`
   - Metadata in `data/processed/metadata`
   - Test results in `test_results/`

## Next Steps

### Phase 2 Planning
1. **Annotation Tool Development**
   - GUI for ball position marking
   - Automatic tracking assistance
   - Quality control features

2. **Dataset Preparation**
   - Frame selection criteria
   - Annotation guidelines
   - Dataset splitting strategy

3. **Model Architecture**
   - YOLOv8 customization
   - Small object detection optimization
   - Motion prediction integration

## Performance Metrics

### Frame Processing
- Target resolution: 1280x1280
- Processing speed: ~5 frames/second
- Storage requirement: ~500KB per frame

### Quality Metrics
- Minimum quality score: 50/100
- Minimum motion threshold: 0.1
- Maximum frames per video: 1000

## Dependencies
```python
opencv-python>=4.8.0
numpy>=1.24.0
pyyaml>=6.0.0
matplotlib>=3.7.0  # For test visualization
```

## Notes
- Frame quality assessment is critical for model training
- Motion detection helps focus on active play
- Preprocessing pipeline ensures consistent data quality
- Metadata tracking enables dataset analysis and cleanup
- Testing framework helps optimize pipeline parameters 