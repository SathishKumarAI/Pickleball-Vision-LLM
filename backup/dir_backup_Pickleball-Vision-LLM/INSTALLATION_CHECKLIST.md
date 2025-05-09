# Pickleball Vision Installation Checklist

## System Requirements
- [ ] Python 3.8 or higher (3.8-3.11)
- [ ] CUDA-capable GPU (optional, for GPU acceleration)
- [ ] 8GB+ RAM recommended
- [ ] 10GB+ free disk space
- [ ] Git installed
- [ ] CMake installed (for some dependencies)
- [ ] Visual Studio Build Tools (Windows only)

## Python Environment Setup
- [ ] Create and activate virtual environment:
  ```bash
  # Using venv
  python -m venv venv
  source venv/bin/activate  # Linux/Mac
  .\venv\Scripts\activate   # Windows
  
  # Or using conda
  conda create -n pickle python=3.10
  conda activate pickle
  ```

- [ ] Upgrade pip and setuptools:
  ```bash
  python -m pip install --upgrade pip setuptools wheel
  ```

## Package Installation
- [ ] Install base package:
  ```bash
  pip install -e .
  ```

- [ ] Install development dependencies:
  ```bash
  pip install -e ".[dev]"
  ```

- [ ] Install GPU support (if needed):
  ```bash
  pip install -e ".[gpu]"
  ```

- [ ] Install test dependencies:
  ```bash
  pip install pytest-cov pytest-asyncio pytest-benchmark
  ```

## Configuration Check
- [ ] Verify CUDA installation (if using GPU):
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

- [ ] Check MLflow server (if enabled):
  ```bash
  # Should be running on http://localhost:5000
  curl http://localhost:5000
  ```

- [ ] Verify OpenCV installation:
  ```bash
  python -c "import cv2; print(cv2.__version__)"
  ```

- [ ] Check PyTorch installation:
  ```bash
  python -c "import torch; print(torch.__version__)"
  ```

## Directory Structure
- [ ] Create required directories:
  ```bash
  mkdir -p logs
  mkdir -p data
  mkdir -p output/frames
  mkdir -p output/vector_store
  mkdir -p models
  mkdir -p tests/data
  mkdir -p tests/output
  ```

- [ ] Set up test data:
  ```bash
  # Copy sample data to tests/data
  cp -r sample_data/* tests/data/
  ```

## Model Downloads
- [ ] Download YOLOv8 model:
  ```bash
  python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
  ```

- [ ] Download additional models:
  ```bash
  # Download CLIP model
  python -c "import clip; clip.load('ViT-B/32')"
  ```

## Testing Setup
- [ ] Run test suite:
  ```bash
  pytest
  ```

- [ ] Check test coverage:
  ```bash
  pytest --cov=pickleball_vision --cov-report=html
  ```

- [ ] Run performance benchmarks:
  ```bash
  pytest --benchmark-only
  ```

- [ ] Run async tests:
  ```bash
  pytest -m asyncio
  ```

## Development Tools
- [ ] Install pre-commit hooks:
  ```bash
  pre-commit install
  ```

- [ ] Verify code formatting:
  ```bash
  black .
  isort .
  ```

- [ ] Run type checking:
  ```bash
  mypy src/pickleball_vision
  ```

- [ ] Run linting:
  ```bash
  flake8 src/pickleball_vision tests
  ```

## Logging Configuration
- [ ] Verify log directory:
  ```bash
  ls logs/
  ```

- [ ] Check log rotation:
  ```bash
  # Should see log files with timestamps
  ls logs/pickleball_vision_*.log
  ```

- [ ] Test logging levels:
  ```bash
  python -c "from pickleball_vision.utils.logger import setup_logger; logger = setup_logger(); logger.debug('Test debug'); logger.info('Test info'); logger.warning('Test warning'); logger.error('Test error')"
  ```

## Performance Verification
- [ ] Run performance tests:
  ```bash
  pytest -m performance
  ```

- [ ] Check GPU utilization (if using GPU):
  ```bash
  nvidia-smi
  ```

- [ ] Run memory profiling:
  ```bash
  pytest --profile
  ```

## Common Issues and Solutions

### CUDA Issues
- [ ] If CUDA not found:
  ```bash
  # Check CUDA version
  nvcc --version
  
  # Verify PyTorch CUDA
  python -c "import torch; print(torch.version.cuda)"
  ```

### MLflow Issues
- [ ] If MLflow server not accessible:
  ```bash
  # Start MLflow server
  mlflow server --host 0.0.0.0 --port 5000
  ```

### Memory Issues
- [ ] If running out of memory:
  ```bash
  # Check available memory
  free -h  # Linux
  top      # Mac
  taskmgr  # Windows
  ```

### OpenCV Issues
- [ ] If OpenCV not working:
  ```bash
  # Reinstall OpenCV
  pip uninstall opencv-python
  pip install opencv-python-headless
  ```

## Final Verification
- [ ] Run example script:
  ```bash
  pickleball-vision --help
  ```

- [ ] Check all dependencies:
  ```bash
  pip list
  ```

- [ ] Verify environment:
  ```bash
  python -c "import pickleball_vision; print(pickleball_vision.__version__)"
  ```

- [ ] Run integration tests:
  ```bash
  pytest tests/integration/
  ```

## Optional Components
- [ ] Install additional visualization tools:
  ```bash
  pip install streamlit plotly
  ```

- [ ] Set up development IDE:
  - [ ] VS Code with Python extension
  - [ ] PyCharm Professional
  - [ ] Jupyter Notebook support

- [ ] Install additional development tools:
  ```bash
  pip install ipython jupyter notebook
  ```

## Security Checks
- [ ] Verify no sensitive data in logs
- [ ] Check file permissions
- [ ] Review environment variables
- [ ] Run security scan:
  ```bash
  pip install bandit
  bandit -r src/pickleball_vision
  ```

## Documentation
- [ ] Build documentation:
  ```bash
  # If using Sphinx
  cd docs
  make html
  ```

- [ ] Check API documentation:
  ```bash
  pydocstyle src/pickleball_vision
  ```

## Maintenance
- [ ] Set up log rotation
- [ ] Configure backup strategy
- [ ] Set up monitoring
- [ ] Set up CI/CD pipeline

## Support
- [ ] Join community channels
- [ ] Review issue templates
- [ ] Check contribution guidelines
- [ ] Set up development workflow

## Notes
- Keep this checklist updated as requirements change
- Document any custom configurations
- Note any workarounds or special cases
- Update test cases when adding new features 