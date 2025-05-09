import pytest
import numpy as np
import faiss
from pathlib import Path
from unittest.mock import Mock, patch
from src.pickleball_vision.database.vector_store import VectorStore
from src.pickleball_vision.config.config import Config

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config(
        USE_GPU=False,
        EMBEDDING_DIM=512,
        VECTOR_STORE_TYPE='faiss'
    )

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return [
        {
            'frame': 1,
            'class': 'player',
            'confidence': 0.95,
            'clip_embedding': np.random.rand(512).tolist(),
            'scene_description': 'Player in ready position'
        },
        {
            'frame': 2,
            'class': 'ball',
            'confidence': 0.90,
            'clip_embedding': np.random.rand(512).tolist(),
            'scene_description': 'Ball in play'
        }
    ]

@pytest.fixture
def vector_store(config):
    """Create a VectorStore instance."""
    return VectorStore(config)

def test_vector_store_initialization(vector_store, config):
    """Test VectorStore initialization."""
    assert vector_store.config == config
    assert isinstance(vector_store.index, faiss.IndexFlatL2)
    assert vector_store.index.d == config.EMBEDDING_DIM
    assert len(vector_store.metadata) == 0

def test_add_embeddings(vector_store, sample_embeddings):
    """Test adding embeddings to the vector store."""
    # Add embeddings
    vector_store.add(sample_embeddings)
    
    # Verify results
    assert vector_store.index.ntotal == len(sample_embeddings)
    assert len(vector_store.metadata) == len(sample_embeddings)
    for i, embedding in enumerate(sample_embeddings):
        assert vector_store.metadata[i]['frame'] == embedding['frame']
        assert vector_store.metadata[i]['class'] == embedding['class']

def test_search_embeddings(vector_store, sample_embeddings):
    """Test searching for similar embeddings."""
    # Add embeddings
    vector_store.add(sample_embeddings)
    
    # Create query vector
    query_vector = np.random.rand(512)
    
    # Search for similar vectors
    results = vector_store.search(query_vector, k=2)
    
    # Verify results
    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert 'vector' in result
        assert 'metadata' in result
        assert 'distance' in result
        assert isinstance(result['vector'], np.ndarray)
        assert isinstance(result['metadata'], dict)
        assert isinstance(result['distance'], float)

def test_search_with_empty_store(vector_store):
    """Test searching in an empty vector store."""
    # Create query vector
    query_vector = np.random.rand(512)
    
    # Search for similar vectors
    results = vector_store.search(query_vector, k=2)
    
    # Verify results
    assert isinstance(results, list)
    assert len(results) == 0

def test_save_and_load(vector_store, sample_embeddings, tmp_path):
    """Test saving and loading the vector store."""
    # Add embeddings
    vector_store.add(sample_embeddings)
    
    # Save vector store
    save_path = tmp_path / "vector_store"
    vector_store.save(save_path)
    
    # Create new vector store
    new_vector_store = VectorStore(vector_store.config)
    
    # Load saved vector store
    new_vector_store.load(save_path)
    
    # Verify results
    assert new_vector_store.index.ntotal == vector_store.index.ntotal
    assert len(new_vector_store.metadata) == len(vector_store.metadata)
    for i in range(len(vector_store.metadata)):
        assert new_vector_store.metadata[i] == vector_store.metadata[i]

def test_clear(vector_store, sample_embeddings):
    """Test clearing the vector store."""
    # Add embeddings
    vector_store.add(sample_embeddings)
    
    # Clear vector store
    vector_store.clear()
    
    # Verify results
    assert vector_store.index.ntotal == 0
    assert len(vector_store.metadata) == 0

def test_gpu_initialization(config):
    """Test initialization with GPU support."""
    # Enable GPU
    config.USE_GPU = True
    
    # Mock faiss.StandardGpuResources
    with patch('faiss.StandardGpuResources') as mock_gpu_resources:
        with patch('faiss.index_cpu_to_gpu') as mock_index_cpu_to_gpu:
            # Create vector store
            vector_store = VectorStore(config)
            
            # Verify GPU initialization
            mock_gpu_resources.assert_called_once()
            mock_index_cpu_to_gpu.assert_called_once()

def test_invalid_embeddings(vector_store):
    """Test handling of invalid embeddings."""
    # Test with None embeddings
    with pytest.raises(ValueError):
        vector_store.add(None)
    
    # Test with empty embeddings
    with pytest.raises(ValueError):
        vector_store.add([])
    
    # Test with invalid embedding format
    with pytest.raises(ValueError):
        vector_store.add([{'invalid': 'format'}])

def test_invalid_search_params(vector_store):
    """Test handling of invalid search parameters."""
    # Test with invalid k value
    with pytest.raises(ValueError):
        vector_store.search(np.random.rand(512), k=0)
    
    # Test with invalid query vector
    with pytest.raises(ValueError):
        vector_store.search(None)
    
    # Test with invalid query vector shape
    with pytest.raises(ValueError):
        vector_store.search(np.random.rand(256))

def test_save_without_directory(vector_store, sample_embeddings):
    """Test saving without creating directory."""
    # Add embeddings
    vector_store.add(sample_embeddings)
    
    # Save vector store
    with pytest.raises(FileNotFoundError):
        vector_store.save(Path("/nonexistent/path"))

def test_load_nonexistent(vector_store):
    """Test loading nonexistent vector store."""
    with pytest.raises(FileNotFoundError):
        vector_store.load(Path("/nonexistent/path")) 