#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

def create_directory_structure():
    """
    Create the complete MLOps project directory structure
    """
    base_dirs = [
        'src/data_management',
        'src/preprocessing',
        'src/model',
        'src/training',
        'src/inference',
        'src/utils',
        'tests/unit',
        'tests/integration',
        'tests/e2e',
        'notebooks/exploration',
        'notebooks/experiments',
        'data/raw',
        'data/processed',
        'data/features',
        'data/artifacts',
        'mlflow',
        'configs',
        'scripts',
        'requirements',
        '.github/workflows',
        'infrastructure/terraform',
        'infrastructure/kubernetes'
    ]

    # Create directories
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def create_initial_files():
    """
    Create initial project files
    """
    files_to_create = {
        # Source files
        'src/data_management/__init__.py': '',
        'src/preprocessing/__init__.py': '',
        'src/model/__init__.py': '',
        'src/training/__init__.py': '',
        'src/inference/__init__.py': '',
        'src/utils/__init__.py': '',

        # Configuration files
        'configs/data_config.yaml': '''
data:
  sources:
    - type: "youtube"
      categories: ["professional", "amateur"]
      sports: ["pickleball"]
  
  preprocessing:
    frame_rate: 5
    augmentations:
      - name: "random_crop"
      - name: "horizontal_flip"
      - name: "brightness_adjust"
''',
        'configs/model_config.yaml': '''
model:
  vision:
    backbone: "yolov8"
    input_size: [640, 640]
    pretrained: true
  
  llm:
    model_name: "mistral-7b"
    max_tokens: 512
    temperature: 0.7
''',
        'configs/inference_config.yaml': '''
inference:
  deployment:
    type: "kubernetes"
    replicas: 3
    resources:
      gpu: 1
      memory: "16Gi"
      cpu: "4"
  
  monitoring:
    enabled: true
    metrics:
      - prediction_latency
      - model_performance
      - data_drift
  
  scaling:
    min_replicas: 1
    max_replicas: 10
    target_cpu_utilization: 70
''',

        # Requirements files
        'requirements/base.txt': '''
torch==2.1.0
torchvision==0.16.0
ultralytics==8.0.43
opencv-python==4.7.0.72
mlflow==2.3.1
feast==0.32.1
prometheus-client==0.17.1
pyyaml==6.0.1
''',
        'requirements/dev.txt': '''
-r base.txt
pytest==7.3.1
ruff==0.0.270
mypy==1.3.0
black==23.3.0
''',
        'requirements/prod.txt': '''
-r base.txt
gunicorn==20.1.0
''',

        # GitHub Actions workflow
        '.github/workflows/ml_pipeline.yml': '''
name: Pickleball AI MLOps Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: 3.11
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements/dev.txt
      
      - name: Lint and test
        run: |
          ruff check src tests
          pytest tests/
''',

        # Main README
        'README.md': '''# Pickleball AI MLOps Project

## Project Setup

1. Create virtual environment
```bash
conda create -n pickleball-ai python=3.11
conda activate pickleball-ai
```

2. Install dependencies
```bash
pip install -r requirements/dev.txt
```

3. Run tests
```bash
pytest tests/
```

## Project Structure

- `src/`: Source code modules
- `tests/`: Test suites
- `configs/`: Configuration files
- `data/`: Data storage
- `notebooks/`: Experimental notebooks
''',

        # Dockerfile
        'Dockerfile': '''
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements/prod.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "src.inference.real_time_predictor"]
''',

        # Docker Compose
        'docker-compose.yml': '''
version: '3.8'

services:
  pickleball-ai:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - MLFLOW_TRACKING_URI=sqlite:///mlflow.db
''',

        # Sample training script
        'scripts/train.py': '''
#!/usr/bin/env python3

import mlflow
import torch
import yaml
from src.model.vision_model import VisionModel
from src.training.experiment_tracking import ExperimentTracker

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config('configs/experiment_config.yaml')
    
    # Initialize experiment tracking
    tracker = ExperimentTracker('pickleball_ai')
    
    # Initialize model
    model = VisionModel(config['model'])
    
    # Run training
    mlflow.start_run()
    try:
        # Training logic would go here
        pass
    finally:
        mlflow.end_run()

if __name__ == '__main__':
    main()
''',
    }

    # Create files
    for file_path, content in files_to_create.items():
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content.strip())
        print(f"Created file: {file_path}")

def initialize_git():
    """
    Initialize git repository and create initial commit
    """
    try:
        # Initialize git
        subprocess.run(['git', 'init'], check=True)
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial project setup'], check=True)
        print("Git repository initialized and first commit created")
    except subprocess.CalledProcessError as e:
        print(f"Error initializing git: {e}")

def main():
    """
    Main function to set up the project
    """
    print("🚀 Initializing Pickleball AI MLOps Project")
    
    # Create directory structure
    create_directory_structure()
    
    # Create initial files
    create_initial_files()
    
    # Initialize git repository
    initialize_git()
    
    print("✅ Project setup complete!")

if __name__ == '__main__':
    main()