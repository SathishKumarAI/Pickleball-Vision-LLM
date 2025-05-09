import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from pickleball_vision.utils.logger import PickleballLogger, setup_logger, get_logger, ColoredFormatter, LogConfig
import os
import logging
from pickleball_vision.utils.colors import Colors

@pytest.fixture
def config():
    """Create a test configuration."""
    return LogConfig(
        LOG_LEVEL='INFO',
        LOG_FILE=Path('test.log'),
        USE_MLFLOW=True,
        MLFLOW_TRACKING_URI='http://localhost:5000'
    )

@pytest.fixture
def logger(config):
    """Create a PickleballLogger instance."""
    return PickleballLogger(config)

def test_logger_initialization(logger, config):
    """Test PickleballLogger initialization."""
    assert logger.config == config
    assert logger.logger is not None
    assert logger.enable_mlflow == config.USE_MLFLOW

def test_log_metrics(logger):
    """Test logging metrics."""
    # Mock mlflow.log_metrics
    with patch('mlflow.log_metrics') as mock_log_metrics:
        # Log metrics
        metrics = {'accuracy': 0.95, 'fps': 30.0}
        logger.log_metrics(metrics)
        
        # Verify mlflow was called
        mock_log_metrics.assert_called_once_with(metrics)

def test_log_metrics_without_mlflow(logger):
    """Test logging metrics without MLflow."""
    # Disable MLflow
    logger.enable_mlflow = False
    
    # Log metrics
    metrics = {'accuracy': 0.95, 'fps': 30.0}
    logger.log_metrics(metrics)  # Should not raise any errors

def test_log_parameters(logger):
    """Test logging parameters."""
    # Mock mlflow.log_params
    with patch('mlflow.log_params') as mock_log_params:
        # Log parameters
        params = {'batch_size': 32, 'learning_rate': 0.001}
        logger.log_parameters(params)
        
        # Verify mlflow was called
        mock_log_params.assert_called_once_with(params)

def test_log_artifacts(logger, tmp_path):
    """Test logging artifacts."""
    # Create test artifact
    artifact_path = tmp_path / "test.txt"
    with open(artifact_path, 'w') as f:
        f.write("Test artifact")
    
    # Mock mlflow.log_artifacts
    with patch('mlflow.log_artifacts') as mock_log_artifacts:
        # Log artifact
        logger.log_artifacts(str(artifact_path))
        
        # Verify mlflow was called
        mock_log_artifacts.assert_called_once_with(str(artifact_path))

def test_log_error(logger):
    """Test logging errors."""
    # Mock logger.error
    with patch.object(logger.logger, 'error') as mock_error:
        # Log error
        error_msg = "Test error"
        logger.log_error(error_msg)
        
        # Verify logger was called
        mock_error.assert_called_once()

def test_log_performance(logger):
    """Test logging performance metrics."""
    # Mock logger.info
    with patch.object(logger.logger, 'info') as mock_info:
        # Log performance
        metrics = {'fps': 30.0, 'memory_usage': '1.5GB'}
        logger.log_performance(metrics)
        
        # Verify logger was called
        mock_info.assert_called_once()

def test_start_mlflow_run(logger):
    """Test starting MLflow run."""
    # Mock mlflow.start_run
    with patch('mlflow.start_run') as mock_start_run:
        # Start run
        run_name = "test_run"
        logger.start_mlflow_run(run_name)
        
        # Verify mlflow was called
        mock_start_run.assert_called_once_with(run_name=run_name)

def test_end_mlflow_run(logger):
    """Test ending MLflow run."""
    # Mock mlflow.end_run
    with patch('mlflow.end_run') as mock_end_run:
        # End run
        logger.end_mlflow_run()
        
        # Verify mlflow was called
        mock_end_run.assert_called_once()

def test_logger_without_mlflow(config):
    """Test logger initialization without MLflow."""
    # Disable MLflow
    config.USE_MLFLOW = False
    
    # Create logger
    logger = PickleballLogger(config)
    
    # Verify MLflow is disabled
    assert not logger.enable_mlflow

def test_logger_with_custom_log_file(config, tmp_path):
    """Test logger with custom log file."""
    # Set custom log file
    log_file = tmp_path / "custom.log"
    config.LOG_FILE = log_file
    
    # Create logger
    logger = PickleballLogger(config)
    
    # Log a message
    test_message = "Test message"
    logger.info(test_message)
    
    # Verify log file was created
    assert log_file.exists()
    with open(log_file, 'r') as f:
        content = f.read()
        assert test_message in content

def test_logger_with_different_log_levels(config):
    """Test logger with different log levels."""
    # Test each log level
    for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        config.LOG_LEVEL = level
        logger = PickleballLogger(config)
        assert logger.logger.level == getattr(logging, level)

def test_logger_with_invalid_mlflow_uri(config):
    """Test logger with invalid MLflow tracking URI."""
    # Set invalid MLflow URI
    config.MLFLOW_TRACKING_URI = 'invalid://uri'
    
    # Create logger (should not raise error)
    logger = PickleballLogger(config)
    
    # Verify MLflow is disabled
    assert not logger.enable_mlflow

def test_logger_with_nonexistent_log_directory(config):
    """Test logger with nonexistent log directory."""
    # Set log file in nonexistent directory
    config.LOG_FILE = Path('/nonexistent/directory/test.log')
    
    # Create logger (should not raise error)
    logger = PickleballLogger(config)
    
    # Verify logger was created
    assert logger.logger is not None 

def test_setup_logger(config):
    """Test logger setup and basic functionality."""
    logger = setup_logger(config)
    assert isinstance(logger, PickleballLogger)
    assert logger.logger.level == getattr(logging, config.LOG_LEVEL)
    assert len(logger.logger.handlers) > 0

def test_logger_output(tmp_path):
    """Test logger output to file."""
    log_file = tmp_path / "test.log"
    logger = setup_logger(log_file=str(log_file))
    
    test_message = "Test log message"
    logger.info(test_message)
    
    assert log_file.exists()
    with open(log_file, 'r') as f:
        content = f.read()
        assert test_message in content

def test_colored_formatter():
    """Test colored formatter functionality."""
    formatter = ColoredFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    formatted = formatter.format(record)
    assert "Test message" in formatted
    assert Colors.GREEN in formatted  # INFO level uses green color

def test_get_logger():
    """Test get_logger function."""
    logger = get_logger()
    assert isinstance(logger, PickleballLogger)
    assert logger.logger.name == "pickleball_vision"

def test_logger_levels():
    """Test different logging levels."""
    logger = setup_logger()
    
    # Test all logging levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

def test_logger_rotation(tmp_path):
    """Test log rotation functionality."""
    log_file = tmp_path / "rotation_test.log"
    logger = setup_logger(
        log_file=str(log_file),
        max_bytes=1000,
        backup_count=3
    )
    
    # Write enough logs to trigger rotation
    for i in range(100):
        logger.info(f"Test message {i}" * 10)
    
    # Check if rotation files were created
    log_files = list(tmp_path.glob("rotation_test*.log*"))
    assert len(log_files) > 1

@pytest.mark.asyncio
async def test_async_logging():
    """Test logging in async context."""
    logger = setup_logger()
    
    async def async_function():
        logger.info("Async log message")
    
    await async_function() 