import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the new directory structure."""
    directories = [
        # Core directories
        "src/pickleball_vision/core/config",
        "src/pickleball_vision/core/utils",
        
        # Vision components
        "src/pickleball_vision/vision",
        "src/pickleball_vision/detection",
        "src/pickleball_vision/tracking",
        "src/pickleball_vision/preprocessing",
        
        # Application components
        "src/pickleball_vision/api",
        "src/pickleball_vision/frontend",
        "src/pickleball_vision/services",
        
        # ML components
        "src/pickleball_vision/ml/experiments",
        "src/pickleball_vision/ml/training",
        
        # Documentation
        "docs/api",
        "docs/guides",
        "docs/development",
        "docs/architecture",
        
        # Data organization
        "data/raw",
        "data/processed",
        "data/test",
        "data/models",
        
        # Scripts organization
        "scripts/setup",
        "scripts/testing",
        "scripts/deployment",
        "scripts/utils"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def move_files():
    """Move files to their new locations."""
    moves = [
        # Move scripts
        ("src/scripts/*", "scripts/utils/"),
        ("src/scripts/train_model.py", "src/pickleball_vision/ml/training/"),
        
        # Move vision components
        ("src/vision/*", "src/pickleball_vision/vision/"),
        ("src/preprocessing/*", "src/pickleball_vision/preprocessing/"),
        
        # Move application components
        ("src/app/*", "src/pickleball_vision/api/"),
        ("src/frontend/*", "src/pickleball_vision/frontend/"),
        
        # Move data
        ("src/data/*", "data/raw/"),
        ("src/models/*", "data/models/"),
        
        # Move documentation
        ("docs/pickleball_vision/api/*", "docs/api/"),
        ("docs/pickleball_vision/guides/*", "docs/guides/"),
        ("docs/pickleball_vision/development/*", "docs/development/")
    ]
    
    for src, dst in moves:
        try:
            if "*" in src:
                # Handle wildcard moves
                for file in Path(src).glob("*"):
                    if file.is_file():
                        shutil.move(str(file), dst)
            else:
                # Handle single file moves
                if Path(src).exists():
                    shutil.move(src, dst)
            print(f"Moved {src} to {dst}")
        except Exception as e:
            print(f"Error moving {src} to {dst}: {e}")

def cleanup():
    """Remove empty directories."""
    directories_to_remove = [
        "src/scripts",
        "src/vision",
        "src/preprocessing",
        "src/app",
        "src/frontend",
        "src/data",
        "src/models"
    ]
    
    for directory in directories_to_remove:
        try:
            if Path(directory).exists():
                shutil.rmtree(directory)
                print(f"Removed directory: {directory}")
        except Exception as e:
            print(f"Error removing {directory}: {e}")

def main():
    """Main function to execute the reorganization."""
    print("Starting project reorganization...")
    
    # Create backup
    backup_dir = f"backup/project_backup_{Path.cwd().name}"
    if not Path(backup_dir).exists():
        shutil.copytree(".", backup_dir, ignore=shutil.ignore_patterns("backup", ".git"))
        print(f"Created backup at: {backup_dir}")
    
    # Execute reorganization steps
    create_directory_structure()
    move_files()
    cleanup()
    
    print("Project reorganization completed!")

if __name__ == "__main__":
    main() 