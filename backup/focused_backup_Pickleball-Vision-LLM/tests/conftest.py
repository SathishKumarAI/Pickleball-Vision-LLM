import os
import pytest
import numpy as np
from pathlib import Path
from pickleball_vision.config.config import Config
from pickleball_vision.utils.colors import Colors
import logging
import sys
from loguru import logger as loguru_logger

@pytest.fixture(scope="session")
def test_config():
    """Create a test configuration for all tests."""
    loguru_logger.debug(Colors.debug("Creating test configuration"))
    config = Config(
        # Base paths
        BASE_DIR=Path(__file__).parent.parent,
        DATA_DIR=Path(__file__).parent / "data",
        OUTPUT_DIR=Path(__file__).parent / "output",
        FRAMES_DIR=Path(__file__).parent / "output" / "frames",
        VECTOR_STORE_DIR=Path(__file__).parent / "output" / "vector_store",
        MODEL_DIR=Path(__file__).parent / "models",
        
        # Video processing
        FRAME_SKIP=2,
        MAX_FRAMES=10,
        MIN_CONFIDENCE=0.5,
        ADAPTIVE_SAMPLING=True,
        SAMPLE_THRESHOLD=0.1,
        
        # Model settings
        USE_GPU=False,
        USE_LLAVA=False,  # Disable LLaVA for tests
        EMBEDDING_DIM=512,
        DETECTOR_MODEL="yolov8n.pt",
        
        # Vector store settings
        VECTOR_STORE_TYPE="faiss",
        VECTOR_STORE_HOST="localhost",
        VECTOR_STORE_PORT=5000,
        
        # Cache settings
        CACHE_ENABLED=True,
        CACHE_DIR=Path(__file__).parent / "output" / "cache",
        
        # Logging settings
        LOG_LEVEL="INFO",
        LOG_FILE=Path(__file__).parent / "output" / "test.log",
        USE_MLFLOW=False  # Disable MLflow for tests
    )
    loguru_logger.debug(Colors.debug(f"Created test config: {config}"))
    return config

@pytest.fixture(scope="session")
def sample_frame():
    """Create a sample frame for testing."""
    loguru_logger.debug(Colors.debug("Creating sample test frame"))
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    loguru_logger.debug(Colors.debug(f"Created frame with shape: {frame.shape}"))
    return frame

@pytest.fixture(scope="session")
def sample_detections():
    """Create sample detections for testing."""
    loguru_logger.debug(Colors.debug("Creating sample detections"))
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
    loguru_logger.debug(Colors.debug(f"Created {len(detections)} sample detections"))
    return detections

@pytest.fixture(scope="session")
def sample_embeddings():
    """Create sample embeddings for testing."""
    loguru_logger.debug(Colors.debug("Creating sample embeddings"))
    embeddings = [
        {
            'frame': 1,
            'class': 'player',
            'confidence': 0.95,
            'clip_embedding': np.random.rand(512).tolist()
        },
        {
            'frame': 2,
            'class': 'ball',
            'confidence': 0.90,
            'clip_embedding': np.random.rand(512).tolist()
        }
    ]
    loguru_logger.debug(Colors.debug(f"Created {len(embeddings)} sample embeddings"))
    return embeddings

@pytest.fixture(scope="session")
def test_video_path(tmp_path_factory):
    """Create a test video file."""
    loguru_logger.debug(Colors.debug("Creating test video file"))
    video_path = tmp_path_factory.mktemp("data") / "test.mp4"
    
    # Create video writer
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    # Generate frames
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
        loguru_logger.debug(Colors.debug(f"Generated frame {i+1}/10"))
    
    out.release()
    loguru_logger.debug(Colors.success(f"Created test video at: {video_path}"))
    return video_path

@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    """Create a test output directory."""
    loguru_logger.debug(Colors.debug("Creating test output directory"))
    output_dir = tmp_path_factory.mktemp("output")
    loguru_logger.debug(Colors.success(f"Created output directory: {output_dir}"))
    return output_dir

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a test data directory."""
    loguru_logger.debug(Colors.debug("Creating test data directory"))
    data_dir = tmp_path_factory.mktemp("data")
    loguru_logger.debug(Colors.success(f"Created data directory: {data_dir}"))
    return data_dir

@pytest.fixture(scope="session")
def test_model_dir(tmp_path_factory):
    """Create a test model directory."""
    loguru_logger.debug(Colors.debug("Creating test model directory"))
    model_dir = tmp_path_factory.mktemp("models")
    loguru_logger.debug(Colors.success(f"Created model directory: {model_dir}"))
    return model_dir

@pytest.fixture(scope="session")
def test_cache_dir(tmp_path_factory):
    """Create a test cache directory."""
    loguru_logger.debug(Colors.debug("Creating test cache directory"))
    cache_dir = tmp_path_factory.mktemp("cache")
    loguru_logger.debug(Colors.success(f"Created cache directory: {cache_dir}"))
    return cache_dir

@pytest.fixture(scope="session")
def test_vector_store_dir(tmp_path_factory):
    """Create a test vector store directory."""
    loguru_logger.debug(Colors.debug("Creating test vector store directory"))
    vector_store_dir = tmp_path_factory.mktemp("vector_store")
    loguru_logger.debug(Colors.success(f"Created vector store directory: {vector_store_dir}"))
    return vector_store_dir

@pytest.fixture(scope="session")
def test_frames_dir(tmp_path_factory):
    """Create a test frames directory."""
    loguru_logger.debug(Colors.debug("Creating test frames directory"))
    frames_dir = tmp_path_factory.mktemp("frames")
    loguru_logger.debug(Colors.success(f"Created frames directory: {frames_dir}"))
    return frames_dir

@pytest.fixture(scope="session")
def test_log_file(tmp_path_factory):
    """Create a test log file."""
    loguru_logger.debug(Colors.debug("Creating test log file"))
    log_file = tmp_path_factory.mktemp("logs") / "test.log"
    loguru_logger.debug(Colors.success(f"Created log file: {log_file}"))
    return log_file

# Configure logging for tests
@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for all tests."""
    # Remove all handlers
    loguru_logger.remove()
    
    # Add console handler with detailed format and colors
    loguru_logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level="DEBUG",
        colorize=True
    )
    
    # Add file handler for test logs
    log_dir = Path("test_logs")
    log_dir.mkdir(exist_ok=True)
    loguru_logger.add(
        log_dir / "test_{time}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
               "{name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="1 MB",
        retention="1 week"
    )
    
    # Intercept standard logging
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    
    yield
    
    # Cleanup after tests
    loguru_logger.remove()

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()

@pytest.fixture
def test_data_dir():
    """Create a test data directory."""
    return Path("tests/data")

@pytest.fixture
def output_dir():
    """Create a test output directory."""
    return Path("tests/output")

@pytest.fixture
def logger():
    """Return a logger instance for tests."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger 