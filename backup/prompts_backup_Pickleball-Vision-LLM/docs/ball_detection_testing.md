# Ball Detection Testing Framework

## Overview
The ball detection testing framework provides tools and utilities to evaluate the data collection pipeline's performance. It includes both individual test cases and a comprehensive test suite that evaluates the pipeline under different configurations.

## Components

### 1. Test Collection Script (`test_collection.py`)
The main test script that evaluates the data collection pipeline with detailed metrics and visualizations.

#### Features
- Frame quality assessment
- Motion detection analysis
- Sample frame visualization
- Statistical analysis
- Results visualization

#### Metrics Collected
- **Quality Scores**
  - Mean, standard deviation
  - Minimum and maximum values
  - Median score
  - Distribution histogram

- **Motion Scores**
  - Mean, standard deviation
  - Minimum and maximum values
  - Median score
  - Distribution histogram

#### Output
- Timestamped test results directory containing:
  - Sample frames
  - Quality distribution plot
  - Motion distribution plot
  - Detailed statistics in YAML format

### 2. Test Suite (`run_test_suite.py`)
A comprehensive test suite that evaluates the pipeline under different configurations.

#### Test Configurations
1. **Default Configuration**
   ```yaml
   brightness_range: {min: 30, max: 225}
   contrast_threshold: 20
   blur_threshold: 100
   motion_threshold: 0.1
   ```

2. **High Quality Configuration**
   ```yaml
   brightness_range: {min: 50, max: 200}
   contrast_threshold: 30
   blur_threshold: 150
   ```

3. **High Motion Configuration**
   ```yaml
   motion_threshold: 0.2
   optical_flow:
     levels: 4
     winsize: 20
     iterations: 4
   ```

4. **Balanced Configuration**
   ```yaml
   brightness_range: {min: 40, max: 210}
   contrast_threshold: 25
   blur_threshold: 120
   motion_threshold: 0.15
   ```

### 3. Video Finder (`find_videos.py`)
Utility script to locate and prepare video files for testing.

#### Features
- Recursive search for video files
- Support for multiple formats (MP4, AVI, MOV, MKV)
- Automatic test directory setup
- Video file preparation for testing

## Usage

### Running Individual Tests
```bash
cd src/pickleball_vision/ball_detection
python scripts/test_collection.py
```
When prompted, enter the path to your test video.

### Running the Test Suite
```bash
cd src/pickleball_vision/ball_detection
python scripts/run_test_suite.py
```
When prompted, enter the path to your test video.

### Finding Test Videos
```bash
cd src/pickleball_vision/ball_detection
python scripts/find_videos.py
```

## Test Results

### Directory Structure
```
test_suite_results/
├── YYYYMMDD_HHMMSS/           # Timestamped test run
│   ├── sample_1.jpg          # Sample frame 1
│   ├── sample_2.jpg          # Sample frame 2
│   ├── sample_3.jpg          # Sample frame 3
│   ├── quality_distribution.png
│   ├── motion_distribution.png
│   └── test_results.yaml     # Detailed test results
├── default_config.yaml       # Default configuration
├── high_quality_config.yaml  # High quality configuration
├── high_motion_config.yaml   # High motion configuration
├── balanced_config.yaml      # Balanced configuration
└── test_suite_results.yaml   # Combined test results
```

### Results Format
```yaml
video_path: "path/to/video.mp4"
timestamp: "YYYYMMDD_HHMMSS"
statistics:
  total_frames: 1000
  quality:
    mean: 75.5
    std: 12.3
    min: 45.0
    max: 95.0
    median: 78.0
  motion:
    mean: 0.15
    std: 0.05
    min: 0.05
    max: 0.25
    median: 0.14
frame_paths:
  - "path/to/frame1.jpg"
  - "path/to/frame2.jpg"
  - "path/to/frame3.jpg"
```

## Best Practices

1. **Test Video Selection**
   - Use high-quality video recordings
   - Ensure good lighting conditions
   - Include various playing scenarios
   - Test with different camera angles

2. **Configuration Tuning**
   - Start with default configuration
   - Adjust thresholds based on video quality
   - Use high-quality config for well-lit videos
   - Use high-motion config for fast-paced games

3. **Results Analysis**
   - Check quality distribution for consistency
   - Verify motion detection sensitivity
   - Review sample frames for visual quality
   - Compare results across configurations

## Troubleshooting

### Common Issues
1. **No Frames Extracted**
   - Check video file format
   - Verify video quality
   - Adjust quality thresholds
   - Check motion threshold

2. **Poor Quality Frames**
   - Increase brightness range
   - Adjust contrast threshold
   - Check blur threshold
   - Verify preprocessing settings

3. **Missing Motion**
   - Lower motion threshold
   - Adjust optical flow parameters
   - Check frame interval
   - Verify video frame rate

## Future Improvements

1. **Additional Metrics**
   - Frame rate analysis
   - Memory usage tracking
   - Processing time metrics
   - GPU utilization (if applicable)

2. **Enhanced Visualization**
   - Interactive result viewer
   - Frame comparison tools
   - Quality score heatmaps
   - Motion trajectory plots

3. **Automated Testing**
   - CI/CD integration
   - Regression testing
   - Performance benchmarking
   - Configuration optimization 