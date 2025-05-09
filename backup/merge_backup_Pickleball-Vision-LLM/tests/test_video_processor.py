import pytest
import cv2
import numpy as np
from pathlib import Path
from pickleball_vision.config.config import Config
from pickleball_vision.processors.video_processor import VideoProcessor
from unittest.mock import Mock, patch
from pickleball_vision.utils.logger import pickleball_logger

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config(
        FRAME_SKIP=2,
        MAX_FRAMES=10,
        MIN_CONFIDENCE=0.5,
        USE_GPU=False
    )

@pytest.fixture
def mock_detector():
    """Create a mock detector."""
    detector = Mock()
    detector.detect.return_value = [
        {
            'class': 'player',
            'confidence': 0.95,
            'bbox': [100, 100, 200, 200],
            'mask': np.ones((100, 100), dtype=np.uint8)
        }
    ]
    return detector

@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = Mock()
    embedder.generate.return_value = [
        {
            'frame': 0,
            'class': 'player',
            'confidence': 0.95,
            'clip_embedding': np.random.rand(512).tolist(),
            'scene_description': 'Player in ready position'
        }
    ]
    return embedder

@pytest.fixture
def mock_visualizer():
    """Create a mock visualizer."""
    visualizer = Mock()
    visualizer.draw_detections.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    visualizer.add_frame_info.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return visualizer

@pytest.fixture
def mock_preprocessor():
    """Create a mock preprocessor."""
    preprocessor = Mock()
    preprocessor.preprocess.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    preprocessor.adaptive_sampling.return_value = True
    return preprocessor

@pytest.fixture
def mock_cache():
    """Create a mock cache."""
    cache = Mock()
    cache.get.return_value = None
    return cache

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    vector_store = Mock()
    return vector_store

@pytest.fixture
def video_processor(config, mock_detector, mock_embedder, mock_visualizer,
                   mock_preprocessor, mock_cache, mock_vector_store):
    """Create a VideoProcessor instance with mocked dependencies."""
    with patch('src.pickleball_vision.processors.video_processor.PickleballDetector',
               return_value=mock_detector), \
         patch('src.pickleball_vision.processors.video_processor.EmbeddingGenerator',
               return_value=mock_embedder), \
         patch('src.pickleball_vision.processors.video_processor.Visualizer',
               return_value=mock_visualizer), \
         patch('src.pickleball_vision.processors.video_processor.FramePreprocessor',
               return_value=mock_preprocessor), \
         patch('src.pickleball_vision.processors.video_processor.CacheManager',
               return_value=mock_cache), \
         patch('src.pickleball_vision.processors.video_processor.VectorStore',
               return_value=mock_vector_store):
        return VideoProcessor(config)

@pytest.fixture
def sample_frame():
    # Create a sample frame (100x100 RGB image)
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

def test_video_processor_initialization(video_processor, config):
    """Test VideoProcessor initialization."""
    assert video_processor.config == config
    assert len(video_processor.detection_log) == 0
    assert len(video_processor.embedding_log) == 0

def test_frame_preprocessing(video_processor, sample_frame):
    """Test frame preprocessing."""
    processed_frame = video_processor.preprocessor.preprocess(sample_frame)
    assert processed_frame is not None
    assert isinstance(processed_frame, np.ndarray)

def test_detection(video_processor, sample_frame):
    """Test object detection."""
    detections = video_processor.detector.detect(sample_frame)
    assert isinstance(detections, list)

def test_visualization(video_processor, sample_frame):
    """Test visualization of detections."""
    detections = video_processor.detector.detect(sample_frame)
    output_frame = video_processor.visualizer.draw_detections(sample_frame, detections)
    assert output_frame is not None
    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == sample_frame.shape

def test_cache_operations(video_processor):
    """Test cache operations."""
    key = "test_key"
    value = {"test": "data"}
    
    # Test setting cache
    video_processor.cache.set(key, value)
    
    # Test getting cache
    cached_value = video_processor.cache.get(key)
    assert cached_value == value

def test_process_video_success(video_processor, tmp_path):
    """Test successful video processing."""
    # Create test video
    video_path = tmp_path / "test.mp4"
    create_test_video(video_path)
    
    # Process video
    result = video_processor.process_video(str(video_path))
    
    # Verify results
    assert result['total_frames'] == 10
    assert result['processed_frames'] > 0
    assert result['detections'] > 0
    assert result['embeddings'] > 0
    assert result['processing_time'] > 0
    assert Path(result['output_path']).exists()

def test_process_video_file_not_found(video_processor):
    """Test video processing with non-existent file."""
    with pytest.raises(FileNotFoundError):
        video_processor.process_video("nonexistent.mp4")

def test_process_video_empty(video_processor, tmp_path):
    """Test processing an empty video."""
    # Create empty video
    video_path = tmp_path / "empty.mp4"
    create_test_video(video_path, num_frames=0)
    
    # Process video
    result = video_processor.process_video(str(video_path))
    
    # Verify results
    assert result['total_frames'] == 0
    assert result['processed_frames'] == 0
    assert result['detections'] == 0
    assert result['embeddings'] == 0

def test_process_video_corrupted(video_processor, tmp_path):
    """Test processing a corrupted video."""
    # Create corrupted video file
    video_path = tmp_path / "corrupted.mp4"
    with open(video_path, 'w') as f:
        f.write("This is not a valid video file")
    
    # Process video
    with pytest.raises(Exception):
        video_processor.process_video(str(video_path))

def test_log_frame_results(video_processor):
    """Test logging frame results."""
    frame_count = 1
    detections = [
        {
            'class': 'player',
            'confidence': 0.95,
            'bbox': [100, 100, 200, 200]
        }
    ]
    embeddings = [
        {
            'class': 'player',
            'confidence': 0.95,
            'clip_embedding': [0.1, 0.2, 0.3]
        }
    ]
    
    # Log results
    video_processor._log_frame_results(frame_count, detections, embeddings)
    
    # Verify logs
    assert len(video_processor.detection_log) == 1
    assert len(video_processor.embedding_log) == 1
    assert video_processor.detection_log[0]['frame'] == frame_count
    assert video_processor.embedding_log[0]['frame'] == frame_count

def test_save_sample_frame(video_processor, tmp_path):
    """Test saving sample frames."""
    # Create test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame_count = 1
    
    # Save frame
    video_processor._save_sample_frame(frame, frame_count)
    
    # Verify frame was saved
    frame_path = video_processor.config.FRAMES_DIR / f"frame_{frame_count:03d}.jpg"
    assert frame_path.exists()

def test_save_results(video_processor):
    """Test saving processing results."""
    # Add some test data
    video_processor.detection_log = [
        {'frame': 1, 'class': 'player', 'confidence': 0.95}
    ]
    video_processor.embedding_log = [
        {'frame': 1, 'class': 'player', 'confidence': 0.95}
    ]
    
    # Save results
    video_processor._save_results()
    
    # Verify files were created
    assert (video_processor.config.OUTPUT_DIR / "detections.csv").exists()
    assert (video_processor.config.OUTPUT_DIR / "embeddings.csv").exists()

def test_print_summary(video_processor, caplog):
    """Test printing processing summary."""
    # Add some test data
    video_processor.detection_log = [
        {'frame': 1, 'class': 'player', 'confidence': 0.95},
        {'frame': 2, 'class': 'ball', 'confidence': 0.90}
    ]
    
    # Print summary
    video_processor._print_summary(processed_count=2, elapsed_time=1.0)
    
    # Verify log output
    assert "Processing Summary" in caplog.text
    assert "Total frames processed: 2" in caplog.text
    assert "Average FPS: 2.0" in caplog.text
    assert "Detection Summary" in caplog.text

def create_test_video(output_path: Path, num_frames: int = 10):
    """Create a test video file.
    
    Args:
        output_path: Path to save the video
        num_frames: Number of frames to generate
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (640, 480))
    
    # Generate frames
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release() 