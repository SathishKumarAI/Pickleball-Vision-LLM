import os
import mlflow
import yaml
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime
from data_filtering import filter_frames
import cv2
import numpy as np
import argparse
import sys

# Add src to Python path
src_path = Path(__file__).parent.parent.parent
sys.path.append(str(src_path))

from pickleball_vision.config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLFlowExperiment:
    """
    MLflow experiment manager for data filtering pipeline evaluation.
    """
    def __init__(self, config_path: str):
        """
        Initialize MLflow experiment manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.experiment_name = self.config.get("experiment_name", "data_filtering_pipeline")
        
        # Set up MLflow
        mlflow.set_tracking_uri(self.config.get("mlflow_tracking_uri", "mlruns"))
        self.experiment = mlflow.set_experiment(self.experiment_name)

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
            print(f"Failed to load config from {config_path}: {e}")
            raise

    def run_experiment(self, input_path: str, output_path: str) -> None:
        """
        Run and track a data filtering experiment.

        Args:
            input_path (str): Path to input video or directory
            output_path (str): Path to save filtered frames
        """
        try:
            with mlflow.start_run() as run:
                # Log basic information
                mlflow.log_param("input_path", input_path)
                mlflow.log_param("output_path", output_path)
                mlflow.log_params(self.config.get("filter_params", {}))
                
                start_time = datetime.now()

                # Import and run the filtering pipeline
                
                # Read video or frames
                if os.path.isfile(input_path):
                    cap = cv2.VideoCapture(input_path)
                    frames = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                    cap.release()
                else:
                    # Handle directory of frames
                    frames = []
                    for img_path in Path(input_path).glob("*.jpg"):
                        frames.append(cv2.imread(str(img_path)))

                # Run filtering pipeline
                filtered_frames = filter_frames(frames, self.config.get("filter_params", {}))

                # Log metrics
                mlflow.log_metric("total_frames", len(frames))
                mlflow.log_metric("filtered_frames", len(filtered_frames))
                mlflow.log_metric("filtering_ratio", len(filtered_frames) / len(frames))
                mlflow.log_metric("processing_time", (datetime.now() - start_time).total_seconds())

                # Save filtered frames
                os.makedirs(output_path, exist_ok=True)
                for i, frame in enumerate(filtered_frames):
                    cv2.imwrite(os.path.join(output_path, f"frame_{i:06d}.jpg"), frame)

                # Log artifacts
                mlflow.log_artifact(config_path)
                
                logger.info(f"Experiment completed. Run ID: {run.info.run_id}")
                logger.info(f"Filtered {len(filtered_frames)} frames from {len(frames)} input frames")

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise

def main():
    """Main entry point for running MLflow experiments."""
    
    parser = argparse.ArgumentParser(description="Run data filtering MLflow experiment")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to configuration file")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to input video or directory of frames")
    parser.add_argument("--output", type=str, required=True,
                      help="Path to save filtered frames")
    
    args = parser.parse_args()
    
    experiment = MLFlowExperiment(args.config)
    experiment.run_experiment(args.input, args.output)

if __name__ == "__main__":
    main()