import pytest
import numpy as np
import cv2
from src.pickleball_vision.processors.preprocessor import FramePreprocessor
from src.pickleball_vision.config.config import Config

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config(
        FRAME_SKIP=2,
        MAX_FRAMES=10,
        MIN_CONFIDENCE=0.5,
        ADAPTIVE_SAMPLING=True,
        SAMPLE_THRESHOLD=0.1
    )

@pytest.fixture
def preprocessor(config):
    """Create a FramePreprocessor instance."""
    return FramePreprocessor(config)

@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

def test_preprocessor_initialization(preprocessor, config):
    """Test FramePreprocessor initialization."""
    assert preprocessor.config == config
    assert preprocessor.frame_count == 0
    assert preprocessor.last_frame is None
    assert preprocessor.motion_history is not None

def test_preprocess_frame(preprocessor, sample_frame):
    """Test frame preprocessing."""
    # Preprocess frame
    processed_frame = preprocessor.preprocess(sample_frame)
    
    # Verify results
    assert isinstance(processed_frame, np.ndarray)
    assert processed_frame.shape == sample_frame.shape
    assert processed_frame.dtype == np.uint8

def test_preprocess_frame_with_resize(preprocessor):
    """Test frame preprocessing with resize."""
    # Create large frame
    large_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # Preprocess frame
    processed_frame = preprocessor.preprocess(large_frame)
    
    # Verify results
    assert isinstance(processed_frame, np.ndarray)
    assert processed_frame.shape == (480, 640, 3)
    assert processed_frame.dtype == np.uint8

def test_preprocess_frame_with_normalization(preprocessor, sample_frame):
    """Test frame preprocessing with normalization."""
    # Enable normalization
    preprocessor.config.NORMALIZE_FRAMES = True
    
    # Preprocess frame
    processed_frame = preprocessor.preprocess(sample_frame)
    
    # Verify results
    assert isinstance(processed_frame, np.ndarray)
    assert processed_frame.shape == sample_frame.shape
    assert processed_frame.dtype == np.float32
    assert np.min(processed_frame) >= 0.0
    assert np.max(processed_frame) <= 1.0

def test_preprocess_frame_with_histogram_equalization(preprocessor, sample_frame):
    """Test frame preprocessing with histogram equalization."""
    # Enable histogram equalization
    preprocessor.config.USE_HISTOGRAM_EQUALIZATION = True
    
    # Preprocess frame
    processed_frame = preprocessor.preprocess(sample_frame)
    
    # Verify results
    assert isinstance(processed_frame, np.ndarray)
    assert processed_frame.shape == sample_frame.shape
    assert processed_frame.dtype == np.uint8

def test_preprocess_frame_with_noise_reduction(preprocessor, sample_frame):
    """Test frame preprocessing with noise reduction."""
    # Enable noise reduction
    preprocessor.config.USE_NOISE_REDUCTION = True
    
    # Preprocess frame
    processed_frame = preprocessor.preprocess(sample_frame)
    
    # Verify results
    assert isinstance(processed_frame, np.ndarray)
    assert processed_frame.shape == sample_frame.shape
    assert processed_frame.dtype == np.uint8

def test_adaptive_sampling(preprocessor, sample_frame):
    """Test adaptive sampling."""
    # Process first frame
    preprocessor.preprocess(sample_frame)
    
    # Create similar frame
    similar_frame = sample_frame.copy()
    similar_frame[0, 0] = 0  # Small change
    
    # Test adaptive sampling
    should_process = preprocessor.adaptive_sampling(similar_frame)
    assert not should_process  # Should skip similar frame
    
    # Create different frame
    different_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test adaptive sampling
    should_process = preprocessor.adaptive_sampling(different_frame)
    assert should_process  # Should process different frame

def test_adaptive_sampling_without_enabled(preprocessor, sample_frame):
    """Test adaptive sampling when disabled."""
    # Disable adaptive sampling
    preprocessor.config.ADAPTIVE_SAMPLING = False
    
    # Process first frame
    preprocessor.preprocess(sample_frame)
    
    # Create similar frame
    similar_frame = sample_frame.copy()
    
    # Test adaptive sampling
    should_process = preprocessor.adaptive_sampling(similar_frame)
    assert should_process  # Should process all frames when disabled

def test_frame_skip(preprocessor, sample_frame):
    """Test frame skipping."""
    # Process frames
    for _ in range(5):
        preprocessor.preprocess(sample_frame)
    
    # Verify frame count
    assert preprocessor.frame_count == 5

def test_max_frames_limit(preprocessor, sample_frame):
    """Test maximum frames limit."""
    # Process frames beyond limit
    for _ in range(preprocessor.config.MAX_FRAMES + 1):
        preprocessor.preprocess(sample_frame)
    
    # Verify frame count
    assert preprocessor.frame_count == preprocessor.config.MAX_FRAMES

def test_motion_detection(preprocessor, sample_frame):
    """Test motion detection."""
    # Process first frame
    preprocessor.preprocess(sample_frame)
    
    # Create frame with motion
    motion_frame = sample_frame.copy()
    motion_frame[100:200, 100:200] = 0  # Add motion
    
    # Process motion frame
    processed_frame = preprocessor.preprocess(motion_frame)
    
    # Verify motion was detected
    assert preprocessor.motion_detected

def test_motion_history(preprocessor, sample_frame):
    """Test motion history tracking."""
    # Process frames
    for _ in range(5):
        preprocessor.preprocess(sample_frame)
    
    # Verify motion history
    assert len(preprocessor.motion_history) == 5
    assert all(isinstance(x, bool) for x in preprocessor.motion_history)

def test_reset(preprocessor, sample_frame):
    """Test resetting the preprocessor."""
    # Process some frames
    for _ in range(5):
        preprocessor.preprocess(sample_frame)
    
    # Reset preprocessor
    preprocessor.reset()
    
    # Verify reset
    assert preprocessor.frame_count == 0
    assert preprocessor.last_frame is None
    assert len(preprocessor.motion_history) == 0
    assert not preprocessor.motion_detected

def test_invalid_frame(preprocessor):
    """Test handling of invalid frame input."""
    # Test with None frame
    with pytest.raises(ValueError):
        preprocessor.preprocess(None)
    
    # Test with empty frame
    with pytest.raises(ValueError):
        preprocessor.preprocess(np.array([]))
    
    # Test with invalid frame shape
    with pytest.raises(ValueError):
        preprocessor.preprocess(np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8))

def test_preprocess_frame_with_roi(preprocessor, sample_frame):
    """Test frame preprocessing with region of interest."""
    # Set ROI
    preprocessor.config.USE_ROI = True
    preprocessor.config.ROI = [100, 100, 300, 300]
    
    # Preprocess frame
    processed_frame = preprocessor.preprocess(sample_frame)
    
    # Verify results
    assert isinstance(processed_frame, np.ndarray)
    assert processed_frame.shape == (200, 200, 3)
    assert processed_frame.dtype == np.uint8

def test_preprocess_frame_with_rotation(preprocessor, sample_frame):
    """Test frame preprocessing with rotation."""
    # Set rotation angle
    preprocessor.config.ROTATE_FRAMES = True
    preprocessor.config.ROTATION_ANGLE = 90
    
    # Preprocess frame
    processed_frame = preprocessor.preprocess(sample_frame)
    
    # Verify results
    assert isinstance(processed_frame, np.ndarray)
    assert processed_frame.shape == (640, 480, 3)
    assert processed_frame.dtype == np.uint8 