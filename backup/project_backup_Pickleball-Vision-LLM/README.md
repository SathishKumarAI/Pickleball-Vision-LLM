# Pickleball Vision LLM

A modern computer vision and LLM-powered system for analyzing pickleball gameplay and providing coaching feedback.

## 🎯 Overview

This system combines state-of-the-art computer vision (YOLOv8, CLIP) with large language models (LLaVA) to analyze pickleball videos and provide actionable coaching feedback. The system processes video input, detects players and ball movements, generates embeddings, and uses vector search to provide contextual coaching insights.

## 🏗️ Architecture

```
Video Input → Frame Extraction → Object Detection → Embedding Generation → Vector Search → Coaching Feedback
```

### Core Components

1. **VideoProcessor**
   - Adaptive frame sampling
   - Resolution optimization
   - GPU acceleration support

2. **PickleballDetector (YOLOv8)**
   - Player detection and tracking
   - Ball trajectory analysis
   - Court boundary detection

3. **EmbeddingGenerator**
   - CLIP for visual embeddings
   - LLaVA for scene understanding
   - Whisper for audio analysis

4. **VectorStore**
   - FAISS/Weaviate integration
   - Efficient similarity search
   - Real-time indexing

5. **FeedbackEngine**
   - Coaching tip generation
   - Performance metrics
   - Actionable insights

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- Docker (for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pickleball-vision-llm.git
cd pickleball-vision-llm

# Create and activate conda environment
conda env create -f conda.yaml
conda activate pickle

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for development
```

### Quick Start

```bash
# Run the main pipeline
python -m pickleball_vision.main --input video.mp4 --output analysis.json

# Start the dashboard
streamlit run src/pickleball_vision/dashboard/app.py
```

## 🛠️ Development

### Project Structure

```
pickleball-vision-llm/
├── src/
│   ├── pickleball_vision/
│   │   ├── processors/      # Video and frame processing
│   │   ├── models/         # ML models and embeddings
│   │   ├── services/       # Business logic
│   │   ├── database/       # Vector store and caching
│   │   ├── visualization/  # Results visualization
│   │   └── web/           # API and dashboard
│   ├── tests/             # Unit and integration tests
│   └── notebooks/         # Jupyter notebooks
├── config/                # Configuration files
├── docker/               # Docker configuration
└── docs/                # Documentation
```

### Development Tools

- **Testing**: pytest with coverage reporting
- **Linting**: pre-commit hooks with black, isort, flake8
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Monitoring**: MLflow for experiment tracking
- **Containerization**: Docker for consistent environments

## 📊 Features

- Real-time player and ball detection
- Shot classification and analysis
- Performance metrics calculation
- Coaching feedback generation
- Interactive visualization dashboard
- API endpoints for integration

## 🔮 Future Work

- [ ] LLaVA integration for advanced scene understanding
- [ ] Whisper integration for game audio analysis
- [ ] LangChain for structured feedback generation
- [ ] Roboflow integration for dataset labeling
- [ ] Airflow for batch processing pipelines

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

For detailed documentation, visit our [docs](docs/) directory or check out the [API reference](docs/api.md).

