import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from src.pickleball_vision.models.detector import PickleballDetector
from src.pickleball_vision.config.config import Config

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config(
        USE_GPU=False,
        MIN_CONFIDENCE=0.5,
        DETECTOR_MODEL="yolov8n.pt"
    )

@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_model():
    """Create a mock YOLO model."""
    model = Mock()
    model.return_value = [
        Mock(
            boxes=Mock(
                xyxy=torch.tensor([[100, 100, 200, 200], [300, 300, 320, 320]]),
                conf=torch.tensor([0.95, 0.90]),
                cls=torch.tensor([0, 1])
            ),
            masks=Mock(
                data=torch.ones((2, 100, 100))
            )
        )
    ]
    return model

@pytest.fixture
def detector(config, mock_model):
    """Create a PickleballDetector instance with mocked model."""
    with patch('ultralytics.YOLO', return_value=mock_model):
        return PickleballDetector(config)

def test_detector_initialization(detector, config):
    """Test PickleballDetector initialization."""
    assert detector.config == config
    assert detector.model is not None
    assert detector.device == 'cpu'
    assert detector.class_names is not None

def test_detect_objects(detector, sample_frame):
    """Test object detection."""
    # Detect objects
    detections = detector.detect(sample_frame)
    
    # Verify results
    assert isinstance(detections, list)
    assert len(detections) == 2
    for detection in detections:
        assert 'class' in detection
        assert 'confidence' in detection
        assert 'bbox' in detection
        assert 'mask' in detection
        assert isinstance(detection['bbox'], list)
        assert len(detection['bbox']) == 4
        assert isinstance(detection['confidence'], float)
        assert isinstance(detection['mask'], np.ndarray)

def test_detect_objects_with_confidence_threshold(detector, sample_frame):
    """Test object detection with confidence threshold."""
    # Set high confidence threshold
    detector.config.MIN_CONFIDENCE = 0.99
    
    # Detect objects
    detections = detector.detect(sample_frame)
    
    # Verify results
    assert isinstance(detections, list)
    assert len(detections) == 0  # No detections should pass threshold

def test_detect_objects_with_gpu(config, mock_model):
    """Test object detection with GPU."""
    # Enable GPU
    config.USE_GPU = True
    
    # Mock CUDA availability
    with patch('torch.cuda.is_available', return_value=True):
        with patch('ultralytics.YOLO', return_value=mock_model):
            detector = PickleballDetector(config)
            assert detector.device == 'cuda'

def test_detect_objects_without_gpu(config, mock_model):
    """Test object detection without GPU."""
    # Disable GPU
    config.USE_GPU = False
    
    # Create detector
    with patch('ultralytics.YOLO', return_value=mock_model):
        detector = PickleballDetector(config)
        assert detector.device == 'cpu'

def test_detect_objects_with_invalid_frame(detector):
    """Test object detection with invalid frame."""
    # Test with None frame
    with pytest.raises(ValueError):
        detector.detect(None)
    
    # Test with empty frame
    with pytest.raises(ValueError):
        detector.detect(np.array([]))
    
    # Test with invalid frame shape
    with pytest.raises(ValueError):
        detector.detect(np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8))

def test_detect_objects_with_model_error(detector, sample_frame):
    """Test object detection with model error."""
    # Mock model to raise exception
    detector.model.side_effect = Exception("Model error")
    
    # Test error handling
    with pytest.raises(Exception) as exc_info:
        detector.detect(sample_frame)
    assert "Model error" in str(exc_info.value)

def test_detect_objects_with_empty_results(detector, sample_frame):
    """Test object detection with empty results."""
    # Mock model to return empty results
    detector.model.return_value = [Mock(boxes=Mock(xyxy=torch.empty((0, 4))))]
    
    # Detect objects
    detections = detector.detect(sample_frame)
    
    # Verify results
    assert isinstance(detections, list)
    assert len(detections) == 0

def test_detect_objects_with_custom_classes(detector, sample_frame):
    """Test object detection with custom classes."""
    # Set custom classes
    detector.class_names = ['custom1', 'custom2']
    
    # Detect objects
    detections = detector.detect(sample_frame)
    
    # Verify results
    assert isinstance(detections, list)
    for detection in detections:
        assert detection['class'] in detector.class_names

def test_detect_objects_with_batch_processing(detector):
    """Test batch processing of frames."""
    # Create batch of frames
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
    
    # Process batch
    batch_detections = detector.detect_batch(frames)
    
    # Verify results
    assert isinstance(batch_detections, list)
    assert len(batch_detections) == len(frames)
    for detections in batch_detections:
        assert isinstance(detections, list)

def test_detect_objects_with_tracking(detector, sample_frame):
    """Test object detection with tracking."""
    # Enable tracking
    detector.config.USE_TRACKING = True
    
    # Detect objects
    detections = detector.detect(sample_frame)
    
    # Verify results
    assert isinstance(detections, list)
    for detection in detections:
        assert 'track_id' in detection
        assert isinstance(detection['track_id'], int)

def test_detect_objects_with_roi(detector, sample_frame):
    """Test object detection with region of interest."""
    # Set ROI
    detector.config.USE_ROI = True
    detector.config.ROI = [100, 100, 300, 300]
    
    # Detect objects
    detections = detector.detect(sample_frame)
    
    # Verify results
    assert isinstance(detections, list)
    for detection in detections:
        bbox = detection['bbox']
        assert bbox[0] >= detector.config.ROI[0]
        assert bbox[1] >= detector.config.ROI[1]
        assert bbox[2] <= detector.config.ROI[2]
        assert bbox[3] <= detector.config.ROI[3]

def test_detect_objects_with_augmentation(detector, sample_frame):
    """Test object detection with augmentation."""
    # Enable augmentation
    detector.config.USE_AUGMENTATION = True
    
    # Detect objects
    detections = detector.detect(sample_frame)
    
    # Verify results
    assert isinstance(detections, list)
    for detection in detections:
        assert 'augmented' in detection
        assert isinstance(detection['augmented'], bool)

def test_cleanup(detector):
    """Test cleanup of resources."""
    # Mock model cleanup
    detector.model = Mock()
    
    # Call cleanup
    detector.__del__()
    
    # Verify cleanup was called
    assert detector.model is None 