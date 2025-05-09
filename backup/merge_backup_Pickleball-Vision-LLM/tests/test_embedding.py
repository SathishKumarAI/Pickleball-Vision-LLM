import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from src.pickleball_vision.config.config import Config

# Import EmbeddingGenerator only if needed
try:
    from src.pickleball_vision.models.embedding import EmbeddingGenerator
except ImportError:
    EmbeddingGenerator = None

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config(
        USE_GPU=False,
        USE_LLAVA=False,  # Disable LLaVA for tests
        EMBEDDING_DIM=512
    )

@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def sample_detections():
    """Create sample detections for testing."""
    return [
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

@pytest.fixture
def mock_clip_model():
    """Create a mock CLIP model."""
    model = Mock()
    model.get_image_features.return_value = torch.randn(1, 512)
    return model

@pytest.fixture
def mock_clip_processor():
    """Create a mock CLIP processor."""
    processor = Mock()
    processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}
    return processor

@pytest.fixture
def mock_llava_model():
    """Create a mock LLaVA model."""
    model = Mock()
    model.generate.return_value = ["A player in ready position with a ball in play"]
    return model

@pytest.fixture
def mock_llava_conv():
    """Create a mock LLaVA conversation."""
    conv = Mock()
    return conv

@pytest.fixture
def embedding_generator(config, mock_clip_model, mock_clip_processor):
    """Create an EmbeddingGenerator instance with mocked dependencies."""
    if EmbeddingGenerator is None:
        pytest.skip("EmbeddingGenerator not available")
    
    with patch('transformers.CLIPModel.from_pretrained',
               return_value=mock_clip_model), \
         patch('transformers.CLIPProcessor.from_pretrained',
               return_value=mock_clip_processor):
        return EmbeddingGenerator(config)

@pytest.mark.skipif(EmbeddingGenerator is None,
                    reason="EmbeddingGenerator not available")
def test_embedding_generator_initialization(embedding_generator, config):
    """Test EmbeddingGenerator initialization."""
    assert embedding_generator.config == config
    assert embedding_generator.device == 'cpu'
    assert embedding_generator.clip_model is not None
    assert embedding_generator.clip_processor is not None
    assert embedding_generator.llava_model is None  # LLaVA disabled
    assert embedding_generator.llava_conv is None

@pytest.mark.skipif(EmbeddingGenerator is None,
                    reason="EmbeddingGenerator not available")
def test_generate_embeddings(embedding_generator, sample_frame, sample_detections):
    """Test generating embeddings for a frame and its detections."""
    # Generate embeddings
    embeddings = embedding_generator.generate(sample_frame, sample_detections)
    
    # Verify results
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_detections)
    for embedding in embeddings:
        assert 'frame' in embedding
        assert 'class' in embedding
        assert 'confidence' in embedding
        assert 'clip_embedding' in embedding
        assert isinstance(embedding['clip_embedding'], list)
        assert len(embedding['clip_embedding']) == embedding_generator.config.EMBEDDING_DIM
        assert 'scene_description' not in embedding  # LLaVA disabled

@pytest.mark.skipif(EmbeddingGenerator is None,
                    reason="EmbeddingGenerator not available")
def test_generate_clip_embeddings(embedding_generator, sample_frame):
    """Test generating CLIP embeddings for a frame."""
    # Generate CLIP embeddings
    embeddings = embedding_generator._generate_clip_embeddings(sample_frame)
    
    # Verify results
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (512,)  # Default CLIP dimension

@pytest.mark.skipif(EmbeddingGenerator is None,
                    reason="EmbeddingGenerator not available")
def test_generate_scene_description_without_llava(embedding_generator,
                                                sample_frame, sample_detections):
    """Test generating scene description without LLaVA."""
    # Generate scene description
    description = embedding_generator._generate_scene_description(
        sample_frame, sample_detections[0])
    
    # Verify results
    assert description == ""

@pytest.mark.skipif(EmbeddingGenerator is None,
                    reason="EmbeddingGenerator not available")
def test_cleanup(embedding_generator):
    """Test cleanup of resources."""
    # Mock the cleanup methods
    embedding_generator.clip_model = Mock()
    embedding_generator.llava_model = None
    
    # Call cleanup
    embedding_generator.__del__()
    
    # Verify cleanup was called
    assert embedding_generator.clip_model is None

@pytest.mark.skipif(EmbeddingGenerator is None,
                    reason="EmbeddingGenerator not available")
def test_error_handling(embedding_generator, sample_frame, sample_detections):
    """Test error handling in embedding generation."""
    # Mock CLIP model to raise an exception
    embedding_generator.clip_model.get_image_features.side_effect = Exception("CLIP error")
    
    # Test error handling
    with pytest.raises(Exception) as exc_info:
        embedding_generator.generate(sample_frame, sample_detections)
    assert "CLIP error" in str(exc_info.value)

@pytest.mark.skipif(EmbeddingGenerator is None,
                    reason="EmbeddingGenerator not available")
def test_gpu_initialization(config):
    """Test initialization with GPU support."""
    # Enable GPU
    config.USE_GPU = True
    
    # Mock torch.cuda.is_available
    with patch('torch.cuda.is_available', return_value=True), \
         patch('transformers.CLIPModel.from_pretrained'), \
         patch('transformers.CLIPProcessor.from_pretrained'):
        generator = EmbeddingGenerator(config)
        assert generator.device == 'cuda'

@pytest.mark.skipif(EmbeddingGenerator is None,
                    reason="EmbeddingGenerator not available")
def test_invalid_frame(embedding_generator, sample_detections):
    """Test handling of invalid frame input."""
    # Test with None frame
    with pytest.raises(ValueError):
        embedding_generator.generate(None, sample_detections)
    
    # Test with empty frame
    with pytest.raises(ValueError):
        embedding_generator.generate(np.array([]), sample_detections)

@pytest.mark.skipif(EmbeddingGenerator is None,
                    reason="EmbeddingGenerator not available")
def test_invalid_detections(embedding_generator, sample_frame):
    """Test handling of invalid detections input."""
    # Test with None detections
    with pytest.raises(ValueError):
        embedding_generator.generate(sample_frame, None)
    
    # Test with empty detections
    with pytest.raises(ValueError):
        embedding_generator.generate(sample_frame, []) 