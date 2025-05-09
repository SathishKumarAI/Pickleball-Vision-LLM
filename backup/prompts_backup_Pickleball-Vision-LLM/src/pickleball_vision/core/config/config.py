"""Configuration management for the Pickleball Vision project."""

import os
from pathlib import Path
import yaml
from typing import Dict, Any, Optional

class Config:
    """Configuration class for managing project settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default.
        """
        self.config_path = config_path or os.path.join(
            Path(__file__).parent.parent.parent,
            'ball_detection/config/data_collection.yaml'
        )
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary access."""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration contains key."""
        return key in self.config
        
    @property
    def frame_extraction(self) -> Dict[str, Any]:
        """Get frame extraction configuration."""
        return self.config.get('frame_extraction', {})
        
    @property 
    def quality_thresholds(self) -> Dict[str, Any]:
        """Get quality threshold configuration."""
        return self.config.get('quality_thresholds', {})
        
    @property
    def motion_detection(self) -> Dict[str, Any]:
        """Get motion detection configuration."""
        return self.config.get('motion_detection', {})
        
    @property
    def output(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get('output', {}) 