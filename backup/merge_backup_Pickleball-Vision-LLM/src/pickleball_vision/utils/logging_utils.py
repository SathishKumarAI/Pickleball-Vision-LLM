"""
Unified logging utilities for the pickleball vision project.

This module provides consistent logging configuration and utilities
across all components of the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""
    
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)

def setup_logging(
    name: str,
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up logging with consistent formatting and output options.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Optional path to log file
        console: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_log_file_path(component: str) -> Path:
    """
    Get standard log file path for a component.
    
    Args:
        component: Component name (e.g., 'ball_detection', 'tracking')
        
    Returns:
        Path to log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / component / f"{timestamp}.log"

def log_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 50,
    fill: str = "█"
) -> None:
    """
    Log progress bar to console.
    
    Args:
        logger: Logger instance
        current: Current progress value
        total: Total value
        prefix: Prefix string
        suffix: Suffix string
        decimals: Number of decimal places
        length: Progress bar length
        fill: Progress bar fill character
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    logger.info(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if current == total:
        logger.info() 