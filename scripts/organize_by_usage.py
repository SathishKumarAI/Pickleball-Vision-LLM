import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create a focused directory structure based on usage patterns."""
    directories = {
        # Core Components
        "src/core": {
            "description": "Core application components",
            "subdirs": ["config", "utils", "logging"]
        },
        
        # Vision Pipeline
        "src/vision": {
            "description": "Computer vision components",
            "subdirs": [
                "detection",  # Ball and player detection
                "tracking",   # Object tracking
                "preprocessing",  # Video preprocessing
                "postprocessing"  # Results processing
            ]
        },
        
        # LLM Components
        "src/llm": {
            "description": "Language model components",
            "subdirs": [
                "models",     # Model definitions
                "prompts",    # Prompt templates
                "embeddings", # Embedding utilities
                "analytics"   # LLM analytics
            ]
        },
        
        # Data Management
        "src/data": {
            "description": "Data handling components",
            "subdirs": [
                "storage",    # Data storage
                "processing", # Data processing
                "validation"  # Data validation
            ]
        },
        
        # API and Web
        "src/api": {
            "description": "API and web components",
            "subdirs": [
                "endpoints",  # API endpoints
                "middleware", # API middleware
                "validation"  # Request validation
            ]
        },
        
        # Integration
        "src/integration": {
            "description": "Integration components",
            "subdirs": [
                "fusion",     # Multi-modal fusion
                "streaming",  # Stream processing
                "analytics"   # Integration analytics
            ]
        },
        
        # Scripts
        "scripts": {
            "description": "Utility scripts",
            "subdirs": [
                "data",       # Data processing scripts
                "training",   # Training scripts
                "deployment", # Deployment scripts
                "utils"       # Utility scripts
            ]
        },
        
        # Documentation
        "docs": {
            "description": "Project documentation",
            "subdirs": [
                "api",        # API documentation
                "guides",     # User guides
                "technical",  # Technical documentation
                "examples"    # Usage examples
            ]
        }
    }
    
    # Create directories and README files
    for dir_path, info in directories.items():
        # Create main directory
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create README
        readme_content = f"""# {Path(dir_path).name.title()}

{info['description']}

## Structure

{chr(10).join(f"- `{subdir}/`: {subdir.replace('_', ' ').title()} components" for subdir in info['subdirs'])}
"""
        with open(f"{dir_path}/README.md", "w") as f:
            f.write(readme_content)
        
        # Create subdirectories
        for subdir in info['subdirs']:
            subdir_path = Path(dir_path) / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py for Python packages
            if dir_path.startswith("src"):
                with open(subdir_path / "__init__.py", "w") as f:
                    f.write(f'"""{subdir.replace("_", " ").title()} components."""\n')

def move_files_to_new_structure():
    """Move files to their new locations based on usage patterns."""
    file_mappings = {
        # Vision components
        "src/vision/detection/*.py": "src/vision/detection/",
        "src/vision/tracking/*.py": "src/vision/tracking/",
        "src/vision/preprocessing/*.py": "src/vision/preprocessing/",
        
        # LLM components
        "src/llm/models/*.py": "src/llm/models/",
        "src/llm/embeddings/*.py": "src/llm/embeddings/",
        "src/llm/analytics/*.py": "src/llm/analytics/",
        
        # Data components
        "src/data/storage/*.py": "src/data/storage/",
        "src/data/processing/*.py": "src/data/processing/",
        
        # API components
        "src/api/endpoints/*.py": "src/api/endpoints/",
        "src/api/middleware/*.py": "src/api/middleware/",
        
        # Integration components
        "src/integration/fusion/*.py": "src/integration/fusion/",
        "src/integration/streaming/*.py": "src/integration/streaming/",
        
        # Core components
        "src/core/config/*.py": "src/core/config/",
        "src/core/utils/*.py": "src/core/utils/",
        "src/core/logging/*.py": "src/core/logging/"
    }
    
    for pattern, dest_dir in file_mappings.items():
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                dest_path = Path(dest_dir) / file_path.name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(dest_path))
                print(f"Moved {file_path} to {dest_path}")

def create_essential_files():
    """Create essential files in the new structure."""
    essential_files = {
        "src/core/config/settings.py": """from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "default"))
API_KEY = os.getenv("API_KEY", "")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "app.log"
""",
        
        "src/core/utils/logger.py": """import logging
from pathlib import Path
from ..config.settings import LOG_LEVEL, LOG_FILE

def setup_logger():
    \"\"\"Configure application logging.\"\"\"
    # Create logs directory
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
""",
        
        "src/main.py": """from src.api import create_app
from src.core.config.settings import API_HOST, API_PORT, DEBUG
from src.core.utils.logger import setup_logger

def main():
    \"\"\"Main application entry point.\"\"\"
    # Setup logging
    logger = setup_logger()
    logger.info("Starting Pickleball Vision LLM application")
    
    # Create and run application
    app = create_app()
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)

if __name__ == "__main__":
    main()
"""
    }
    
    for path, content in essential_files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created essential file: {path}")

def main():
    """Main function to execute directory organization."""
    print("Starting directory organization...")
    
    # Create new directory structure
    create_directory_structure()
    
    # Move files to new locations
    move_files_to_new_structure()
    
    # Create essential files
    create_essential_files()
    
    print("Directory organization completed!")

if __name__ == "__main__":
    main() 