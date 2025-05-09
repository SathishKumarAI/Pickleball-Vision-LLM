import os
import shutil
from pathlib import Path

def create_doc_structure():
    """Create new documentation structure."""
    directories = [
        # Main documentation categories
        "docs/architecture",
        "docs/api",
        "docs/guides",
        "docs/development",
        "docs/prompts",
        
        # Architecture subdirectories
        "docs/architecture/system",
        "docs/architecture/components",
        "docs/architecture/data_flow",
        
        # API documentation
        "docs/api/endpoints",
        "docs/api/models",
        "docs/api/examples",
        
        # Guides
        "docs/guides/installation",
        "docs/guides/usage",
        "docs/guides/troubleshooting",
        
        # Development
        "docs/development/setup",
        "docs/development/contributing",
        "docs/development/testing",
        
        # Prompts
        "docs/prompts/vision",
        "docs/prompts/analysis",
        "docs/prompts/common"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def organize_docs():
    """Organize documentation files."""
    doc_moves = {
        # Architecture documentation
        "docs/pickleball_vision/architecture/system_overview.md": "docs/architecture/system/overview.md",
        "docs/pickleball_vision/architecture/component_design.md": "docs/architecture/components/design.md",
        "docs/pickleball_vision/architecture/data_flow.md": "docs/architecture/data_flow/main.md",
        
        # API documentation
        "docs/pickleball_vision/api/endpoints.md": "docs/api/endpoints/main.md",
        "docs/pickleball_vision/api/models.md": "docs/api/models/main.md",
        "docs/pickleball_vision/api/examples.md": "docs/api/examples/main.md",
        
        # Guides
        "docs/pickleball_vision/guides/installation.md": "docs/guides/installation/main.md",
        "docs/pickleball_vision/guides/usage.md": "docs/guides/usage/main.md",
        "docs/pickleball_vision/guides/troubleshooting.md": "docs/guides/troubleshooting/main.md",
        
        # Development
        "docs/pickleball_vision/development/setup.md": "docs/development/setup/main.md",
        "docs/pickleball_vision/development/contributing.md": "docs/development/contributing/main.md",
        "docs/pickleball_vision/development/testing.md": "docs/development/testing/main.md",
        
        # Prompts
        "docs/prompts/vision_prompts.md": "docs/prompts/vision/main.md",
        "docs/prompts/analysis_prompts.md": "docs/prompts/analysis/main.md",
        "docs/prompts/common_prompts.md": "docs/prompts/common/main.md"
    }
    
    for src, dst in doc_moves.items():
        try:
            if Path(src).exists():
                # Create parent directory if it doesn't exist
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src, dst)
                print(f"Moved {src} to {dst}")
        except Exception as e:
            print(f"Error moving {src} to {dst}: {e}")

def cleanup_empty_dirs():
    """Remove empty and unwanted directories."""
    directories_to_remove = [
        # Empty documentation directories
        "docs/pickleball_vision",
        "docs/empty",
        "docs/temp",
        "docs/draft",
        
        # Empty source directories
        "src/pickleball_vision/empty",
        "src/pickleball_vision/temp",
        "src/pickleball_vision/draft",
        
        # Empty test directories
        "tests/empty",
        "tests/temp",
        
        # Empty script directories
        "scripts/empty",
        "scripts/temp"
    ]
    
    for directory in directories_to_remove:
        try:
            if Path(directory).exists():
                # Check if directory is empty
                if not any(Path(directory).iterdir()):
                    shutil.rmtree(directory)
                    print(f"Removed empty directory: {directory}")
        except Exception as e:
            print(f"Error removing {directory}: {e}")

def create_main_readme():
    """Create main README.md with updated structure."""
    readme_content = """# Pickleball Vision Project

## Project Structure

### Source Code (`src/pickleball_vision/`)
- **core/**: Core components and utilities
  - `config/`: Configuration management
  - `utils/`: Utility functions
  - `database/`: Database components
- **vision/**: Vision processing components
  - `detection/`: Object detection
  - `tracking/`: Object tracking
  - `preprocessing/`: Image preprocessing
- **ml/**: Machine learning components
  - `training/`: Model training scripts
  - `experiments/`: Experiment tracking
- **api/**: API endpoints and services
- **frontend/**: Frontend components
- **infrastructure/**: Infrastructure components
  - `monitoring/`: Monitoring setup
  - `nginx/`: Nginx configuration
  - `grafana/`: Grafana dashboards

### Documentation (`docs/`)
- **architecture/**: System architecture documentation
  - `system/`: System overview
  - `components/`: Component design
  - `data_flow/`: Data flow diagrams
- **api/**: API documentation
  - `endpoints/`: API endpoints
  - `models/`: Data models
  - `examples/`: Usage examples
- **guides/**: User guides
  - `installation/`: Installation guides
  - `usage/`: Usage guides
  - `troubleshooting/`: Troubleshooting guides
- **development/**: Development documentation
  - `setup/`: Development setup
  - `contributing/`: Contribution guidelines
  - `testing/`: Testing guidelines
- **prompts/**: LLM prompts
  - `vision/`: Vision-related prompts
  - `analysis/`: Analysis prompts
  - `common/`: Common prompts

### Scripts (`scripts/`)
- **setup/**: Setup scripts
- **monitoring/**: Monitoring scripts
- **deployment/**: Deployment scripts
- **utils/**: Utility scripts

### Data (`data/`)
- **raw/**: Raw data
- **processed/**: Processed data
- **test/**: Test data
- **models/**: Model files

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up the environment:
   ```bash
   scripts/setup/setup_env.sh
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("Created main README.md")

def main():
    """Main function to execute documentation cleanup."""
    print("Starting documentation cleanup and organization...")
    
    # Create backup
    backup_dir = f"backup/docs_backup_{Path.cwd().name}"
    if not Path(backup_dir).exists():
        shutil.copytree(".", backup_dir, ignore=shutil.ignore_patterns("backup", ".git"))
        print(f"Created backup at: {backup_dir}")
    
    # Execute cleanup steps
    create_doc_structure()
    organize_docs()
    cleanup_empty_dirs()
    create_main_readme()
    
    print("Documentation cleanup and organization completed!")

if __name__ == "__main__":
    main() 