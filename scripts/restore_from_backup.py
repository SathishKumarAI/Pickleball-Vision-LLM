import os
import shutil
from pathlib import Path

def restore_files():
    """Restore relevant files from backup to new locations."""
    file_restore_map = {
        # Vision components
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/ball_detection/src/detector.py": "src/vision/detection/detector.py",
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/ball_detection/src/tracker.py": "src/vision/tracking/tracker.py",
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/processors/preprocessor.py": "src/vision/preprocessing/preprocessor.py",
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/processors/video_processor.py": "src/vision/preprocessing/video_processor.py",
        
        # LLM components
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/models/embedding.py": "src/llm/embedding.py",
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/ml/training/train_model.py": "src/llm/training/train_model.py",
        
        # API components
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/api/serving/app.py": "src/api/app.py",
        
        # Configuration
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/shared/config/environments/development.env": "src/fusion/config/development.env",
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/shared/config/environments/production.env": "src/fusion/config/production.env",
        
        # Scripts
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/scripts/process_video.py": "scripts/data/process_video.py",
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/ball_detection/scripts/collect_data.py": "scripts/data/collect_data.py",
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/ball_detection/scripts/test_collection.py": "scripts/data/test_collection.py",
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/ball_detection/scripts/find_videos.py": "scripts/data/find_videos.py",
        "backup/cleanup_backup_Pickleball-Vision-LLM/src/pickleball_vision/ball_detection/scripts/download_sample.py": "scripts/data/download_sample.py",
        
        # Documentation
        "backup/cleanup_backup_Pickleball-Vision-LLM/docs/pickleball_vision/core/configuration.md": "docs/technical/configuration.md",
        "backup/cleanup_backup_Pickleball-Vision-LLM/docs/pickleball_vision/utils/preprocessing.md": "docs/technical/preprocessing.md",
        "backup/cleanup_backup_Pickleball-Vision-LLM/docs/pickleball_vision/utils/visualization.md": "docs/technical/visualization.md",
        "backup/cleanup_backup_Pickleball-Vision-LLM/docs/pickleball_vision/monitoring/metrics.md": "docs/technical/metrics.md"
    }
    
    for src_path, dst_path in file_restore_map.items():
        try:
            if os.path.exists(src_path):
                # Create destination directory if it doesn't exist
                Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(src_path, dst_path)
                print(f"Restored {src_path} to {dst_path}")
            else:
                print(f"Source file not found: {src_path}")
        except Exception as e:
            print(f"Error restoring {src_path}: {e}")

def create_module_files():
    """Create essential module files."""
    module_files = {
        # Vision module
        "src/vision/detection/__init__.py": """from .detector import Detector

__all__ = ['Detector']
""",

        "src/vision/tracking/__init__.py": """from .tracker import Tracker

__all__ = ['Tracker']
""",

        "src/vision/preprocessing/__init__.py": """from .preprocessor import Preprocessor
from .video_processor import VideoProcessor

__all__ = ['Preprocessor', 'VideoProcessor']
""",

        # LLM module
        "src/llm/__init__.py": """from .embedding import EmbeddingModel

__all__ = ['EmbeddingModel']
""",

        # Fusion module
        "src/fusion/__init__.py": """from .config import load_config

__all__ = ['load_config']
""",

        # API module
        "src/api/__init__.py": """from .app import create_app

__all__ = ['create_app']
"""
    }
    
    for path, content in module_files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created module file: {path}")

def main():
    """Main function to execute file restoration."""
    print("Starting file restoration from backup...")
    
    # Restore files
    restore_files()
    
    # Create module files
    create_module_files()
    
    print("File restoration completed!")

if __name__ == "__main__":
    main() 