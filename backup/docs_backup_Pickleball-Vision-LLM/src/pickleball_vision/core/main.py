import os
import argparse
import mlflow
import mlflow.pytorch
from pathlib import Path
from rich.console import Console
from rich.progress import Progress

from .core.config.config import Config
from .processors.video_processor import VideoProcessor
from .utils.logger import setup_logger
from .utils.model_registry import ModelRegistry
from .utils.ab_testing import ABTesting
from .utils.distributed_training import DistributedTrainer
from .utils.visualizer import Visualizer
from .utils.analyzer import GameAnalyzer

console = Console()
logger = setup_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Pickleball Vision System")
    parser.add_argument("--config_path", type=str, default="configs/base_config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--video_path", type=str, default="data/raw_videos/",
                      help="Path to input video or directory")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                      help="Path to output directory")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "ab_test"],
                      default="inference", help="Operation mode")
    parser.add_argument("--distributed", action="store_true",
                      help="Use distributed training")
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Pickleball Vision Analysis")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, help="Path to output video")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    
    try:
        # Initialize MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("pickleball-vision")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("config_path", args.config_path)
            mlflow.log_param("video_path", args.video_path)
            mlflow.log_param("output_dir", args.output_dir)
            mlflow.log_param("mode", args.mode)
            mlflow.log_param("distributed", args.distributed)
            
            # Load configuration
            config = Config()
            if args.config:
                config.load_yaml_config(args.config)
            mlflow.log_dict(config.to_dict(), "config.yaml")
            
            # Initialize components
            processor = VideoProcessor(config)
            model_registry = ModelRegistry()
            visualizer = Visualizer(config)
            analyzer = GameAnalyzer(config)
            
            if args.mode == "train":
                if args.distributed:
                    # Initialize distributed trainer
                    trainer = DistributedTrainer()
                    DistributedTrainer.run_distributed_training(
                        trainer=trainer,
                        model=processor.detector.model,
                        train_dataset=processor.train_dataset,
                        val_dataset=processor.val_dataset,
                        batch_size=config.batch_size,
                        num_epochs=config.num_epochs,
                        learning_rate=config.learning_rate
                    )
                else:
                    # Regular training
                    results = processor.train()
                    mlflow.log_metrics(results)
                
                # Register model
                model_registry.register_model(
                    model=processor.detector.model,
                    run_id=mlflow.active_run().info.run_id,
                    model_name="pickleball-detector",
                    metrics=results,
                    params=config.to_dict(),
                    description="YOLOv8 model for pickleball detection"
                )
            
            elif args.mode == "ab_test":
                # Initialize A/B testing
                ab_tester = ABTesting()
                
                # Load models for comparison
                model_a = model_registry.load_model("pickleball-detector", version="1")
                model_b = model_registry.load_model("pickleball-detector", version="2")
                
                # Run A/B test
                results = ab_tester.run_ab_test(
                    model_a=model_a,
                    model_b=model_b,
                    test_data=processor.test_dataset,
                    metrics=["accuracy", "precision", "recall", "f1_score"]
                )
                
                # Log results
                mlflow.log_dict(results, "ab_test_results.json")
            
            else:  # inference
                # Process video
                if args.video:
                    results = processor.process_video(args.video, args.output)
                    
                    # Print results
                    print("\nProcessing Results:")
                    print(f"Frames processed: {results['num_frames']}")
                    print(f"Processing time: {results['processing_time']:.2f} seconds")
                    if args.output:
                        print(f"Output saved to: {args.output}")
                    
                    # Log metrics
                    mlflow.log_metric(f"{Path(args.video).stem}_total_frames", results["total_frames"])
                    mlflow.log_metric(f"{Path(args.video).stem}_processed_frames", results["processed_frames"])
                    mlflow.log_metric(f"{Path(args.video).stem}_detection_time", results["detection_time"])
                    
                    # Log artifacts
                    if results["output_video"]:
                        mlflow.log_artifact(results["output_video"], "videos")
                    if results["detections_csv"]:
                        mlflow.log_artifact(results["detections_csv"], "detections")
                    
                else:
                    print("Please provide an input video path")
                    sys.exit(1)
            
            console.print("\n[bold green]✓ Processing complete!")

    except Exception as e:
        console.print(f"[red]✗ Error processing: {str(e)}")
        logger.error(f"Error processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 