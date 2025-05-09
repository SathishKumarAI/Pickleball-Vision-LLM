import os
import shutil
from pathlib import Path

def move_remaining_files():
    """Move all remaining relevant files from backup to new locations."""
    backup_root = "backup/cleanup_backup_Pickleball-Vision-LLM"
    
    # Define file patterns and their new locations
    file_patterns = {
        # Vision related
        "**/ball_detection/**/*.py": "src/vision/detection/",
        "**/tracking/**/*.py": "src/vision/tracking/",
        "**/preprocessing/**/*.py": "src/vision/preprocessing/",
        "**/detection/**/*.py": "src/vision/detection/",
        
        # LLM related
        "**/models/**/*.py": "src/llm/",
        "**/ml/**/*.py": "src/llm/",
        "**/analytics/**/*.py": "src/llm/analytics/",
        
        # API and Web
        "**/api/**/*.py": "src/api/",
        "**/frontend/**/*.py": "src/web/",
        "**/serving/**/*.py": "src/api/",
        
        # Utils and Config
        "**/utils/**/*.py": "src/fusion/utils/",
        "**/config/**/*.py": "src/fusion/config/",
        "**/core/**/*.py": "src/fusion/",
        
        # Scripts
        "**/scripts/**/*.py": "scripts/data/",
        "**/ball_detection/scripts/**/*.py": "scripts/data/",
        
        # Documentation
        "**/docs/**/*.md": "docs/technical/",
        "**/pickleball_vision/docs/**/*.md": "docs/technical/"
    }
    
    for pattern, dest_dir in file_patterns.items():
        try:
            # Find all matching files
            for file_path in Path(backup_root).glob(pattern):
                if file_path.is_file():
                    # Create destination path
                    rel_path = file_path.relative_to(backup_root)
                    dest_path = Path(dest_dir) / rel_path.name
                    
                    # Create destination directory
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move file
                    shutil.copy2(file_path, dest_path)
                    print(f"Moved {file_path} to {dest_path}")
        except Exception as e:
            print(f"Error processing pattern {pattern}: {e}")

def create_essential_files():
    """Create essential files in new structure."""
    essential_files = {
        # Configuration
        "src/fusion/config/__init__.py": """from pathlib import Path
import os
from dotenv import load_dotenv

def load_config():
    \"\"\"Load configuration from environment files.\"\"\"
    env_file = os.getenv('ENV_FILE', 'development.env')
    env_path = Path(__file__).parent / env_file
    load_dotenv(env_path)
    return {
        'model_path': os.getenv('MODEL_PATH'),
        'api_key': os.getenv('API_KEY'),
        'debug': os.getenv('DEBUG', 'False').lower() == 'true'
    }
""",

        # Main application
        "src/main.py": """from src.api import create_app
from src.fusion.config import load_config

def main():
    \"\"\"Main application entry point.\"\"\"
    config = load_config()
    app = create_app(config)
    app.run(debug=config['debug'])

if __name__ == '__main__':
    main()
""",

        # Requirements
        "requirements.txt": """# Core dependencies
numpy>=1.21.0
opencv-python>=4.5.0
torch>=1.9.0
transformers>=4.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
python-dotenv>=0.19.0

# Vision
ultralytics>=8.0.0  # YOLO
mediapipe>=0.8.0

# LLM
sentence-transformers>=2.0.0
langchain>=0.0.200

# Utils
tqdm>=4.62.0
pillow>=8.0.0
pandas>=1.3.0
"""
    }
    
    for path, content in essential_files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created essential file: {path}")

def cleanup_backup():
    """Remove backup directory after successful migration."""
    try:
        backup_dir = "backup/cleanup_backup_Pickleball-Vision-LLM"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
            print(f"Removed backup directory: {backup_dir}")
    except Exception as e:
        print(f"Error removing backup: {e}")

def main():
    """Main function to execute final cleanup."""
    print("Starting final cleanup and file migration...")
    
    # Move remaining files
    move_remaining_files()
    
    # Create essential files
    create_essential_files()
    
    # Clean up backup
    cleanup_backup()
    
    print("Final cleanup completed!")

if __name__ == "__main__":
    main() 