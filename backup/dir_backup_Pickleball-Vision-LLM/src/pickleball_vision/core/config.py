"""
Configuration module for pickleball vision project.
"""

from pathlib import Path
import yaml

class Config:
    """Base configuration class."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration."""
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
        
    def __getitem__(self, key: str):
        """Get configuration value using dict-like access."""
        return self.config[key] 