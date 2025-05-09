import os
import argparse
from pathlib import Path
from pickleball_vision.models.pickleball_trainer import PickleballTrainer
from pickleball_vision.utils.logger import setup_logger

logger = setup_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train pickleball vision model')
    parser.add_argument('--data', type=str, default='config/data.yaml',
                      help='Path to data configuration file')
    parser.add_argument('--model-type', type=str, default='yolov8n.pt',
                      help='YOLOv8 model type to use')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                      help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to train on (cuda or cpu)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--export-format', type=str, default='onnx',
                      help='Format to export the trained model')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = PickleballTrainer(
            model_type=args.model_type,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir
        )
        
        # Train model
        logger.info("Starting training...")
        metrics = trainer.train_yolo(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch_size=args.batch_size
        )
        
        # Export model
        logger.info(f"Exporting model to {args.export_format} format...")
        exported_path = trainer.export_model(
            format=args.export_format,
            output_path=str(checkpoint_dir / f"model.{args.export_format}")
        )
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Final metrics: {metrics}")
        logger.info(f"Model exported to: {exported_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 