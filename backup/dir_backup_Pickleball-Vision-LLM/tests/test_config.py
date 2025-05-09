"""Tests for configuration module."""
import os
import pytest
from pathlib import Path
import yaml
from pickleball_vision.core.config.config import Config

@pytest.fixture
def config():
    """Create a test configuration instance."""
    return Config()

@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file."""
    config_data = {
        'MODEL_PATH': 'models/yolov5s.pt',
        'INPUT_SIZE': [640, 640],
        'CONFIDENCE_THRESHOLD': 0.5,
        'CLASS_NAMES': ['player', 'ball', 'court'],
        'LOG_LEVEL': 'INFO',
        'LOG_DIR': 'logs',
        'LOG_FILE': 'app.log',
        'FRAME_HISTORY': 30,
        'MOTION_THRESHOLD': 0.1
    }
    
    config_file = tmp_path / 'test_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
        
    return config_file

def test_config_initialization(config):
    """Test configuration initialization."""
    assert config is not None
    assert isinstance(config, Config)
    
def test_default_values(config):
    """Test default configuration values."""
    assert config.MODEL_PATH == 'models/yolov5s.pt'
    assert config.INPUT_SIZE == [640, 640]
    assert config.CONFIDENCE_THRESHOLD == 0.5
    assert config.CLASS_NAMES == ['player', 'ball', 'court']
    assert config.LOG_LEVEL == 'INFO'
    assert config.LOG_DIR == 'logs'
    assert config.LOG_FILE == 'app.log'
    assert config.FRAME_HISTORY == 30
    assert config.MOTION_THRESHOLD == 0.1
    
def test_load_yaml_config(config, temp_config_file):
    """Test loading configuration from YAML file."""
    config.load_yaml_config(temp_config_file)
    
    assert config.MODEL_PATH == 'models/yolov5s.pt'
    assert config.INPUT_SIZE == [640, 640]
    assert config.CONFIDENCE_THRESHOLD == 0.5
    assert config.CLASS_NAMES == ['player', 'ball', 'court']
    assert config.LOG_LEVEL == 'INFO'
    assert config.LOG_DIR == 'logs'
    assert config.LOG_FILE == 'app.log'
    assert config.FRAME_HISTORY == 30
    assert config.MOTION_THRESHOLD == 0.1
    
def test_load_invalid_yaml_config(config, tmp_path):
    """Test loading invalid YAML configuration."""
    invalid_file = tmp_path / 'invalid.yaml'
    with open(invalid_file, 'w') as f:
        f.write('invalid: yaml: content')
        
    with pytest.raises(Exception):
        config.load_yaml_config(invalid_file)
        
def test_load_nonexistent_yaml_config(config):
    """Test loading nonexistent YAML configuration."""
    with pytest.raises(FileNotFoundError):
        config.load_yaml_config('nonexistent.yaml')
        
def test_to_dict(config):
    """Test converting configuration to dictionary."""
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict['MODEL_PATH'] == config.MODEL_PATH
    assert config_dict['INPUT_SIZE'] == config.INPUT_SIZE
    assert config_dict['CONFIDENCE_THRESHOLD'] == config.CONFIDENCE_THRESHOLD
    assert config_dict['CLASS_NAMES'] == config.CLASS_NAMES
    assert config_dict['LOG_LEVEL'] == config.LOG_LEVEL
    assert config_dict['LOG_DIR'] == config.LOG_DIR
    assert config_dict['LOG_FILE'] == config.LOG_FILE
    assert config_dict['FRAME_HISTORY'] == config.FRAME_HISTORY
    assert config_dict['MOTION_THRESHOLD'] == config.MOTION_THRESHOLD
    
def test_from_dict(config):
    """Test creating configuration from dictionary."""
    config_dict = {
        'MODEL_PATH': 'custom_model.pt',
        'INPUT_SIZE': [320, 320],
        'CONFIDENCE_THRESHOLD': 0.7,
        'CLASS_NAMES': ['custom1', 'custom2'],
        'LOG_LEVEL': 'DEBUG',
        'LOG_DIR': 'custom_logs',
        'LOG_FILE': 'custom.log',
        'FRAME_HISTORY': 60,
        'MOTION_THRESHOLD': 0.2
    }
    
    config.from_dict(config_dict)
    
    assert config.MODEL_PATH == 'custom_model.pt'
    assert config.INPUT_SIZE == [320, 320]
    assert config.CONFIDENCE_THRESHOLD == 0.7
    assert config.CLASS_NAMES == ['custom1', 'custom2']
    assert config.LOG_LEVEL == 'DEBUG'
    assert config.LOG_DIR == 'custom_logs'
    assert config.LOG_FILE == 'custom.log'
    assert config.FRAME_HISTORY == 60
    assert config.MOTION_THRESHOLD == 0.2
    
def test_invalid_input_size(config):
    """Test invalid input size configuration."""
    with pytest.raises(ValueError):
        config.INPUT_SIZE = [0, 0]
        
def test_invalid_confidence_threshold(config):
    """Test invalid confidence threshold configuration."""
    with pytest.raises(ValueError):
        config.CONFIDENCE_THRESHOLD = -0.1
        
    with pytest.raises(ValueError):
        config.CONFIDENCE_THRESHOLD = 1.1
        
def test_invalid_log_level(config):
    """Test invalid log level configuration."""
    with pytest.raises(ValueError):
        config.LOG_LEVEL = 'INVALID'
        
def test_invalid_frame_history(config):
    """Test invalid frame history configuration."""
    with pytest.raises(ValueError):
        config.FRAME_HISTORY = -1
        
def test_invalid_motion_threshold(config):
    """Test invalid motion threshold configuration."""
    with pytest.raises(ValueError):
        config.MOTION_THRESHOLD = -0.1
        
    with pytest.raises(ValueError):
        config.MOTION_THRESHOLD = 1.1 