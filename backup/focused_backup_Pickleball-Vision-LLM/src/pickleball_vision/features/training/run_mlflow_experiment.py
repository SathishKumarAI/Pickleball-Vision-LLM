"""MLflow experiment management for pickleball vision."""
import os
import mlflow
import yaml
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime
import cv2
import numpy as np
import argparse
import sys

from ...core.config.config import Config
from ...core.logging.logger import setup_logging
from ...vision.preprocessor import FramePreprocessor
from ...vision.detector import ObjectDetector

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class MLFlowExperiment:
    """MLflow experiment manager for data filtering pipeline evaluation."""
    
    def __init__(self, config_path: str):
        """Initialize MLflow experiment manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.experiment_name = self.config.get("experiment_name", "data_filtering_pipeline")
        
        # Set up MLflow
        mlflow.set_tracking_uri(self.config.get("mlflow_tracking_uri", "mlruns"))
        self.experiment = mlflow.set_experiment(self.experiment_name)
        
        # Initialize components
        self.preprocessor = FramePreprocessor()
        self.detector = ObjectDetector()

    def _load_config(self, config_path: str) -> Config:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config object
        """
        try:
            config = Config()
            config.load_yaml_config(config_path)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    # ... rest of the existing code ... 