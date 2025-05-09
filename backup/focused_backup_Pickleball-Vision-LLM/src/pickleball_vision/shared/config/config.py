"""Configuration management for pickleball vision system."""
import os
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from loguru import logger

@dataclass
class Config:
    """Configuration settings for the pickleball vision system."""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    FRAMES_DIR: Path = OUTPUT_DIR / "frames"
    VECTOR_STORE_DIR: Path = OUTPUT_DIR / "vector_store"
    MODEL_DIR: Path = BASE_DIR / "models"
    
    # Video processing
    FRAME_SKIP: int = 5
    MAX_FRAMES: int = 1000
    MIN_CONFIDENCE: float = 0.5
    ADAPTIVE_SAMPLING: bool = True
    SAMPLE_THRESHOLD: float = 0.1
    
    # Model settings
    DETECTOR_MODEL: str = "yolov8x-seg.pt"
    EMBEDDING_DIM: int = 512
    USE_GPU: bool = True
    USE_LLAVA: bool = True
    USE_WHISPER: bool = False
    
    # Vector store
    VECTOR_STORE_TYPE: str = "faiss"  # or "weaviate"
    VECTOR_STORE_HOST: str = "localhost"
    VECTOR_STORE_PORT: int = 8080
    
    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_DIR: Path = OUTPUT_DIR / "cache"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[Path] = OUTPUT_DIR / "app.log"
    USE_MLFLOW: bool = False
    MLFLOW_TRACKING_URI: Optional[str] = None
    
    # Analysis settings
    ANALYSIS_INTERVAL: float = 0.1  # seconds
    BATCH_SIZE: int = 1024
    WINDOW_SIZE: int = 5
    TIME_WINDOW: float = 1.0
    
    # GPU settings
    GPU_MEMORY_LIMIT: int = 4096  # MB
    GPU_BATCH_SIZE: int = 1024
    
    # ML settings
    HIDDEN_SIZE: int = 64
    DROPOUT_RATE: float = 0.2
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 100
    
    # Visualization settings
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        "team1": "#FF0000",
        "team2": "#0000FF",
        "neutral": "#808080",
        "success": "#00FF00",
        "failure": "#FF0000"
    })
    
    # Court dimensions
    COURT_WIDTH: float = 6.1  # meters
    COURT_LENGTH: float = 13.4  # meters
    NET_HEIGHT: float = 0.91  # meters
    
    # Shot types
    SHOT_TYPES: List[str] = field(default_factory=lambda: [
        "serve",
        "return",
        "drive",
        "drop",
        "lob",
        "smash",
        "dink"
    ])
    
    # Player positions
    POSITIONS: Dict[str, List[float]] = field(default_factory=lambda: {
        "server": [0.0, 0.0],
        "receiver": [0.0, 13.4],  # COURT_LENGTH
        "partner1": [3.05, 0.0],  # COURT_WIDTH/2
        "partner2": [3.05, 13.4]  # COURT_WIDTH/2, COURT_LENGTH
    })
    
    # Analysis thresholds
    THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "min_speed": 5.0,  # m/s
        "max_speed": 50.0,  # m/s
        "min_spin": 100.0,  # rpm
        "max_spin": 2000.0,  # rpm
        "min_effectiveness": 0.0,
        "max_effectiveness": 1.0
    })
    
    # Detection colors (BGR format)
    CONFIDENCE_THRESHOLD: float = 0.3
    MASK_OPACITY: float = 0.3
    
    # Preprocessing settings
    PREPROCESSING: Dict = None
    
    # Cache settings
    CACHING: Dict = None
    
    # Error handling settings
    ERROR_HANDLING: Dict = None
    
    # Distributed processing settings
    DISTRIBUTED: Dict = None
    
    # Model serving settings
    SERVING: Dict = None

    def __post_init__(self):
        """Initialize configuration after dataclass creation."""
        try:
            # Create directories
            self._setup_directories()
            
            # Validate settings
            self._validate_settings()
            
            # Load environment variables
            self._load_from_env()
            
            # Setup logging
            self._setup_logging()
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            raise
    
    def _setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.DATA_DIR, self.OUTPUT_DIR, self.FRAMES_DIR, 
                        self.VECTOR_STORE_DIR, self.CACHE_DIR]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")
                raise
    
    def _validate_settings(self):
        """Validate configuration settings."""
        try:
            # Check model file exists
            model_path = self.MODEL_DIR / self.DETECTOR_MODEL
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            # Validate vector store settings
            if self.VECTOR_STORE_TYPE not in ["faiss", "weaviate"]:
                raise ValueError(f"Invalid vector store type: {self.VECTOR_STORE_TYPE}")
                
            # Check GPU availability
            if self.USE_GPU:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("GPU requested but not available, falling back to CPU")
                        self.USE_GPU = False
                except ImportError:
                    logger.warning("PyTorch not installed, GPU support disabled")
                    self.USE_GPU = False
                    
            # Validate log level
            valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if self.LOG_LEVEL not in valid_log_levels:
                raise ValueError(f"Invalid log level: {self.LOG_LEVEL}")
                
            # Validate numeric parameters
            assert self.ANALYSIS_INTERVAL > 0, "Analysis interval must be positive"
            assert self.BATCH_SIZE > 0, "Batch size must be positive"
            assert self.WINDOW_SIZE > 0, "Window size must be positive"
            assert self.TIME_WINDOW > 0, "Time window must be positive"
            assert self.GPU_MEMORY_LIMIT > 0, "GPU memory limit must be positive"
            assert self.GPU_BATCH_SIZE > 0, "GPU batch size must be positive"
            assert 0 < self.DROPOUT_RATE < 1, "Dropout rate must be between 0 and 1"
            assert self.LEARNING_RATE > 0, "Learning rate must be positive"
            assert self.NUM_EPOCHS > 0, "Number of epochs must be positive"
            assert self.COURT_WIDTH > 0, "Court width must be positive"
            assert self.COURT_LENGTH > 0, "Court length must be positive"
            assert self.NET_HEIGHT > 0, "Net height must be positive"
            assert len(self.SHOT_TYPES) > 0, "Shot types list cannot be empty"
            
            # Validate thresholds
            assert self.THRESHOLDS["min_speed"] >= 0, "Min speed must be non-negative"
            assert self.THRESHOLDS["max_speed"] > self.THRESHOLDS["min_speed"], "Max speed must be greater than min speed"
            assert self.THRESHOLDS["min_spin"] >= 0, "Min spin must be non-negative"
            assert self.THRESHOLDS["max_spin"] > self.THRESHOLDS["min_spin"], "Max spin must be greater than min spin"
            assert 0 <= self.THRESHOLDS["min_effectiveness"] <= 1, "Min effectiveness must be between 0 and 1"
            assert 0 <= self.THRESHOLDS["max_effectiveness"] <= 1, "Max effectiveness must be between 0 and 1"
            assert self.THRESHOLDS["max_effectiveness"] > self.THRESHOLDS["min_effectiveness"], "Max effectiveness must be greater than min effectiveness"
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_vars = {
            'FRAME_SKIP': int,
            'MAX_FRAMES': int,
            'MIN_CONFIDENCE': float,
            'USE_GPU': bool,
            'USE_LLAVA': bool,
            'USE_WHISPER': bool,
            'VECTOR_STORE_TYPE': str,
            'VECTOR_STORE_HOST': str,
            'VECTOR_STORE_PORT': int,
            'CACHE_TTL': int,
            'LOG_LEVEL': str
        }
        
        for var, type_ in env_vars.items():
            env_var = f"PICKLEBALL_{var}"
            if env_var in os.environ:
                try:
                    value = os.environ[env_var]
                    if type_ == bool:
                        value = value.lower() in ['true', '1', 'yes']
                    else:
                        value = type_(value)
                    setattr(self, var, value)
                except Exception as e:
                    logger.warning(f"Failed to load environment variable {env_var}: {e}")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        from ..logging.logger import setup_logging
        setup_logging()
    
    def load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dictionary containing configuration values
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            ValueError: If config values are invalid
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                config = {}
                
            # Update preprocessing settings
            if 'preprocessing' in config:
                self.PREPROCESSING = config['preprocessing']
                
            # Update cache settings
            if 'caching' in config:
                self.CACHING = config['caching']
                
            # Update error handling settings
            if 'error_handling' in config:
                self.ERROR_HANDLING = config['error_handling']
                
            # Update distributed settings
            if 'distributed' in config:
                self.DISTRIBUTED = config['distributed']
                
            # Update serving settings
            if 'serving' in config:
                self.SERVING = config['serving']
                
            return config
                
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def save_yaml_config(self, config_path: str):
        """Save configuration to YAML file.
        
        Args:
            config_path: Path to save YAML configuration file
            
        Raises:
            IOError: If file cannot be written
            yaml.YAMLError: If configuration cannot be serialized
        """
        try:
            config = {
                'preprocessing': self.PREPROCESSING,
                'caching': self.CACHING,
                'error_handling': self.ERROR_HANDLING,
                'distributed': self.DISTRIBUTED,
                'serving': self.SERVING
            }
            
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
                
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    @property
    def target_classes(self) -> list:
        """Get list of target classes for detection."""
        return list(self.COLORS.keys())
    
    @property
    def is_distributed(self) -> bool:
        """Check if distributed processing is enabled."""
        return self.DISTRIBUTED and self.DISTRIBUTED.get('enabled', False)
    
    @property
    def is_caching_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.CACHING and self.CACHING.get('enabled', False)
    
    @property
    def is_preprocessing_enabled(self) -> bool:
        """Check if preprocessing is enabled."""
        return self.PREPROCESSING and self.PREPROCESSING.get('frame_sampling', {}).get('enabled', False)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls() 