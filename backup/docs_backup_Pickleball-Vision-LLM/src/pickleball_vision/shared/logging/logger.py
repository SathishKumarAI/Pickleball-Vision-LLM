"""Logging utilities for pickleball analysis."""
import logging
import sys
from pathlib import Path
from typing import Optional
from ..core.config.config import Config

def setup_logger(config: Optional[Config] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        config: Optional configuration object. If None, a new Config instance will be created.
        
    Returns:
        Configured logger instance
    """
    try:
        config = config or Config()
        
        # Create logger
        logger = logging.getLogger('pickleball_vision')
        logger.setLevel(config.LOG_LEVEL)
        
        # Create handlers
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.LOG_LEVEL)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        handlers.append(console_handler)
        
        # File handler
        if config.LOG_FILE:
            log_dir = Path(config.LOG_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / config.LOG_FILE,
                mode='a'
            )
            file_handler.setLevel(config.LOG_LEVEL)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_format)
            handlers.append(file_handler)
            
        # Add handlers to logger
        for handler in handlers:
            logger.addHandler(handler)
            
        return logger
        
    except Exception as e:
        print(f"Error setting up logger: {str(e)}")
        # Return basic logger as fallback
        logger = logging.getLogger('pickleball_vision')
        logger.setLevel(logging.INFO)
        return logger
        
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.
    
    Args:
        name: Module name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'pickleball_vision.{name}') 