import os
import shutil
from pathlib import Path

def move_files():
    """Move files to their new locations based on functionality."""
    file_moves = {
        # Vision related files
        "src/pickleball_vision/vision/*": "src/vision/",
        "src/pickleball_vision/detection/*": "src/vision/detection/",
        "src/pickleball_vision/tracking/*": "src/vision/tracking/",
        "src/pickleball_vision/preprocessing/*": "src/vision/preprocessing/",
        
        # LLM related files
        "src/pickleball_vision/llm/*": "src/llm/",
        "src/pickleball_vision/analytics/*": "src/llm/analytics/",
        
        # API and Web related files
        "src/pickleball_vision/api/*": "src/api/",
        "src/pickleball_vision/frontend/*": "src/web/",
        
        # Core utilities
        "src/pickleball_vision/utils/*": "src/fusion/utils/",
        "src/pickleball_vision/core/config.py": "src/fusion/config.py",
        "src/pickleball_vision/core/utils/*": "src/fusion/utils/",
        
        # Data files
        "src/data/*": "data/raw/",
        "src/models/*": "data/models/"
    }
    
    for src_pattern, dst_dir in file_moves.items():
        try:
            # Handle wildcard patterns
            if "*" in src_pattern:
                src_dir = os.path.dirname(src_pattern)
                if os.path.exists(src_dir):
                    for item in os.listdir(src_dir):
                        src_path = os.path.join(src_dir, item)
                        if os.path.isfile(src_path):
                            Path(dst_dir).mkdir(parents=True, exist_ok=True)
                            shutil.move(src_path, os.path.join(dst_dir, item))
                            print(f"Moved {src_path} to {dst_dir}")
            else:
                # Handle single file moves
                if os.path.exists(src_pattern):
                    Path(dst_dir).parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(src_pattern, dst_dir)
                    print(f"Moved {src_pattern} to {dst_dir}")
        except Exception as e:
            print(f"Error moving {src_pattern}: {e}")

def cleanup_duplicate_dirs():
    """Remove duplicate and old directories."""
    dirs_to_remove = [
        # Old source directories
        "src/pickleball_vision",
        "src/pickleball_vision.egg-info",
        "src/logs",
        "src/config",
        "src/utils",
        
        # Old documentation directories
        "docs/pickleball_vision",
        "docs/architecture",
        "docs/development",
        "docs/prompts",
        
        # Empty or redundant directories
        "src/scripts/scripts",
        "src/data",
        "src/models",
        "nginx/conf.d",
        "grafana/provisioning"
    ]
    
    for directory in dirs_to_remove:
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"Removed directory: {directory}")
        except Exception as e:
            print(f"Error removing {directory}: {e}")

def create_init_files():
    """Create __init__.py files in new directories."""
    init_dirs = [
        "src/vision",
        "src/vision/detection",
        "src/vision/tracking",
        "src/vision/preprocessing",
        "src/llm",
        "src/llm/analytics",
        "src/fusion",
        "src/fusion/utils",
        "src/api",
        "src/web"
    ]
    
    for directory in init_dirs:
        init_file = os.path.join(directory, "__init__.py")
        Path(directory).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write('"""Module initialization."""\n')
            print(f"Created {init_file}")

def create_component_readmes():
    """Create README files for each component."""
    readmes = {
        "src/vision/README.md": """# Vision Module

## Components
- Detection: Ball and player detection using YOLO
- Tracking: Object tracking and motion analysis
- Preprocessing: Video frame preprocessing and optimization

## Usage
See technical documentation for detailed usage instructions.
""",

        "src/llm/README.md": """# LLM Module

## Components
- Game Analysis: State interpretation and strategy analysis
- Coaching: Tip generation and natural language output
- Analytics: Performance and quality metrics

## Usage
See technical documentation for detailed usage instructions.
""",

        "src/fusion/README.md": """# Fusion Module

## Components
- Integration: Vision-LLM data integration
- Synchronization: Real-time data flow management
- Utils: Common utilities and configuration

## Usage
See technical documentation for detailed usage instructions.
""",

        "src/api/README.md": """# API Module

## Components
- Endpoints: RESTful API endpoints
- Authentication: User authentication and authorization
- Documentation: API documentation and examples

## Usage
See API documentation for detailed usage instructions.
""",

        "src/web/README.md": """# Web Interface

## Components
- Dashboard: Real-time game analysis dashboard
- Visualization: Game state and analytics visualization
- User Management: User profiles and settings

## Usage
See user guides for detailed usage instructions.
"""
    }
    
    for path, content in readmes.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created {path}")

def main():
    """Main function to execute cleanup and file movement."""
    print("Starting cleanup and file movement...")
    
    # Create backup
    backup_dir = f"backup/cleanup_backup_{Path.cwd().name}"
    if not os.path.exists(backup_dir):
        shutil.copytree(".", backup_dir, ignore=shutil.ignore_patterns("backup", ".git"))
        print(f"Created backup at: {backup_dir}")
    
    # Execute cleanup steps
    move_files()
    cleanup_duplicate_dirs()
    create_init_files()
    create_component_readmes()
    
    print("Cleanup and file movement completed!")

if __name__ == "__main__":
    main() 