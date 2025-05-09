import pytest
import numpy as np
import cv2
from pathlib import Path
from src.pickleball_vision.utils.visualizer import Visualizer
from src.pickleball_vision.config.config import Config
from loguru import logger

@pytest.fixture
def config():
    """Create a test configuration."""
    logger.debug("Creating test configuration")
    config = Config(
        USE_GPU=False,
        MIN_CONFIDENCE=0.5,
        SHOW_FPS=True,
        SHOW_FRAME_COUNT=True,
        SHOW_DETECTIONS=True,
        SHOW_TRACKING=True
    )
    logger.debug(f"Created config: {config}")
    return config

@pytest.fixture
def visualizer(config):
    """Create a Visualizer instance."""
    logger.debug("Initializing Visualizer")
    visualizer = Visualizer(config)
    logger.debug("Visualizer initialized successfully")
    return visualizer

@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    logger.debug("Creating sample test frame")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    logger.debug(f"Created frame with shape: {frame.shape}")
    return frame

@pytest.fixture
def sample_detections():
    """Create sample detections for testing."""
    logger.debug("Creating sample detections")
    detections = [
        {
            'class': 'player',
            'confidence': 0.95,
            'bbox': [100, 100, 200, 200],
            'mask': np.ones((100, 100), dtype=np.uint8)
        },
        {
            'class': 'ball',
            'confidence': 0.90,
            'bbox': [300, 300, 320, 320],
            'mask': np.ones((20, 20), dtype=np.uint8)
        }
    ]
    logger.debug(f"Created {len(detections)} sample detections")
    return detections

def test_visualizer_initialization(visualizer, config):
    """Test Visualizer initialization."""
    logger.info("Testing Visualizer initialization")
    assert visualizer.config == config
    logger.debug("Config verification passed")
    
    assert visualizer.colors is not None
    logger.debug("Colors initialization verified")
    
    assert visualizer.font is not None
    logger.debug("Font initialization verified")
    
    assert visualizer.font_scale is not None
    assert visualizer.thickness is not None
    logger.debug("Visualization parameters verified")

def test_draw_detections(visualizer, sample_frame, sample_detections):
    """Test drawing detections on frame."""
    logger.info("Testing detection drawing")
    
    # Log input details
    logger.debug(f"Input frame shape: {sample_frame.shape}")
    logger.debug(f"Number of detections: {len(sample_detections)}")
    
    # Draw detections
    output_frame = visualizer.draw_detections(sample_frame, sample_detections)
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    logger.debug("Output is numpy array")
    
    assert output_frame.shape == sample_frame.shape
    logger.debug(f"Output shape matches input: {output_frame.shape}")
    
    assert output_frame.dtype == np.uint8
    logger.debug("Output dtype is uint8")

def test_draw_detections_without_confidence(visualizer, sample_frame, sample_detections):
    """Test drawing detections without confidence scores."""
    logger.info("Testing detection drawing without confidence scores")
    
    # Remove confidence from detections
    detections_without_conf = [
        {k: v for k, v in d.items() if k != 'confidence'}
        for d in sample_detections
    ]
    logger.debug(f"Removed confidence from {len(detections_without_conf)} detections")
    
    # Draw detections
    output_frame = visualizer.draw_detections(sample_frame, detections_without_conf)
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8
    logger.debug("Successfully drew detections without confidence scores")

def test_draw_detections_with_masks(visualizer, sample_frame, sample_detections):
    """Test drawing detections with masks."""
    # Draw detections
    output_frame = visualizer.draw_detections(sample_frame, sample_detections)
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_draw_detections_with_tracking(visualizer, sample_frame, sample_detections):
    """Test drawing detections with tracking information."""
    # Add tracking IDs
    detections_with_tracking = [
        {**d, 'track_id': i} for i, d in enumerate(sample_detections)
    ]
    
    # Draw detections
    output_frame = visualizer.draw_detections(sample_frame, detections_with_tracking)
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_add_frame_info(visualizer, sample_frame):
    """Test adding frame information."""
    # Add frame info
    output_frame = visualizer.add_frame_info(sample_frame, frame_count=1, fps=30.0)
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_add_frame_info_without_fps(visualizer, sample_frame):
    """Test adding frame information without FPS."""
    # Add frame info
    output_frame = visualizer.add_frame_info(sample_frame, frame_count=1)
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_add_frame_info_without_frame_count(visualizer, sample_frame):
    """Test adding frame information without frame count."""
    # Add frame info
    output_frame = visualizer.add_frame_info(sample_frame, fps=30.0)
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_draw_trajectory(visualizer, sample_frame):
    """Test drawing trajectory."""
    # Create trajectory points
    trajectory = [(100, 100), (150, 150), (200, 200)]
    
    # Draw trajectory
    output_frame = visualizer.draw_trajectory(sample_frame, trajectory)
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_draw_trajectory_with_color(visualizer, sample_frame):
    """Test drawing trajectory with custom color."""
    # Create trajectory points
    trajectory = [(100, 100), (150, 150), (200, 200)]
    
    # Draw trajectory with custom color
    output_frame = visualizer.draw_trajectory(sample_frame, trajectory, color=(0, 255, 0))
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_draw_trajectory_with_thickness(visualizer, sample_frame):
    """Test drawing trajectory with custom thickness."""
    # Create trajectory points
    trajectory = [(100, 100), (150, 150), (200, 200)]
    
    # Draw trajectory with custom thickness
    output_frame = visualizer.draw_trajectory(sample_frame, trajectory, thickness=3)
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_draw_text(visualizer, sample_frame):
    """Test drawing text on frame."""
    # Draw text
    output_frame = visualizer.draw_text(sample_frame, "Test Text", (100, 100))
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_draw_text_with_custom_style(visualizer, sample_frame):
    """Test drawing text with custom style."""
    # Draw text with custom style
    output_frame = visualizer.draw_text(
        sample_frame,
        "Test Text",
        (100, 100),
        color=(0, 255, 0),
        scale=2.0,
        thickness=2
    )
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_draw_bbox(visualizer, sample_frame):
    """Test drawing bounding box."""
    # Draw bounding box
    output_frame = visualizer.draw_bbox(sample_frame, [100, 100, 200, 200])
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_draw_bbox_with_custom_style(visualizer, sample_frame):
    """Test drawing bounding box with custom style."""
    # Draw bounding box with custom style
    output_frame = visualizer.draw_bbox(
        sample_frame,
        [100, 100, 200, 200],
        color=(0, 255, 0),
        thickness=2
    )
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_draw_mask(visualizer, sample_frame):
    """Test drawing mask."""
    # Create mask
    mask = np.ones((100, 100), dtype=np.uint8)
    
    # Draw mask
    output_frame = visualizer.draw_mask(sample_frame, mask, [100, 100, 200, 200])
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_draw_mask_with_custom_style(visualizer, sample_frame):
    """Test drawing mask with custom style."""
    # Create mask
    mask = np.ones((100, 100), dtype=np.uint8)
    
    # Draw mask with custom style
    output_frame = visualizer.draw_mask(
        sample_frame,
        mask,
        [100, 100, 200, 200],
        color=(0, 255, 0),
        alpha=0.5
    )
    
    # Verify results
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape
    assert output_frame.dtype == np.uint8

def test_invalid_frame(visualizer):
    """Test handling of invalid frame input."""
    logger.info("Testing invalid frame handling")
    
    # Test with None frame
    logger.debug("Testing None frame")
    with pytest.raises(ValueError) as exc_info:
        visualizer.draw_detections(None, [])
    logger.debug(f"None frame error: {exc_info.value}")
    
    # Test with empty frame
    logger.debug("Testing empty frame")
    with pytest.raises(ValueError) as exc_info:
        visualizer.draw_detections(np.array([]), [])
    logger.debug(f"Empty frame error: {exc_info.value}")
    
    # Test with invalid frame shape
    logger.debug("Testing invalid frame shape")
    invalid_frame = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    with pytest.raises(ValueError) as exc_info:
        visualizer.draw_detections(invalid_frame, [])
    logger.debug(f"Invalid shape error: {exc_info.value}")

def test_invalid_detections(visualizer, sample_frame):
    """Test handling of invalid detections input."""
    logger.info("Testing invalid detections handling")
    
    # Test with None detections
    logger.debug("Testing None detections")
    with pytest.raises(ValueError) as exc_info:
        visualizer.draw_detections(sample_frame, None)
    logger.debug(f"None detections error: {exc_info.value}")
    
    # Test with invalid detection format
    logger.debug("Testing invalid detection format")
    with pytest.raises(ValueError) as exc_info:
        visualizer.draw_detections(sample_frame, [{'invalid': 'format'}])
    logger.debug(f"Invalid format error: {exc_info.value}") 