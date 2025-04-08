# Pickleball-Vision-LLM

**Pickleball LLM** is a multimodal AI-powered system that analyzes Pickleball gameplay from videos using computer vision and language models. The system provides real-time feedback, performance analytics, and personalized coaching using cutting-edge technology.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Environment Setup](#environment-setup)
- [Project Structure Explained](#project-structure-explained)
- [Usage](#usage)
- [Development Roadmap](#development-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 📖 Overview

The goal of Pickleball LLM is to revolutionize Pickleball training by offering an automated, AI-driven coaching system that uses video input to detect player movements, classify actions, and generate coaching advice using LLMs.

---

## ✨ Key Features

- 🎥 YouTube or local video ingestion
- 🧍‍♂️ Player detection & tracking
- 🕴️ Pose estimation and movement classification
- 📊 Action recognition (serve, volley, drops)
- 🧠 Natural language feedback using LLMs
- 💾 Save and compare historical sessions

---

## 🔧 Tech Stack

| Area              | Tools / Frameworks                                           |
|-------------------|--------------------------------------------------------------|
| Language          | Python, Rust (Tauri), JavaScript (Svelte)                   |
| AI/ML             | PyTorch, YOLOv8, Mediapipe, Transformers, OpenCV            |
| Backend           | FastAPI / Tauri                                              |
| Frontend          | Svelte + Tailwind CSS                                        |
| LLMs              | GPT-4 / Mistral / LLaVA                                      |
| Data & Storage    | SQLite, optionally PostgreSQL, Local Filesystem              |
| Orchestration     | Docker, GitHub Actions, MLflow                              |
| Package Manager   | `uv`, `conda`, `pyproject.toml`                             |

---

## 🧠 Environment Setup: Modern Python Stack with `uv` and `conda`

### ✅ Step 1: Conda Environment

```yaml
# environment.yml
name: pickleball_llm
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - uv
```

```bash
conda env create -f environment.yml
conda activate pickleball_llm
```

### ⚡ Step 2: Manage Dependencies with `uv`

```toml
# pyproject.toml
[project]
name = "pickleball_llm"
version = "0.1.0"
description = "Multimodal LLM model for analyzing pickleball gameplay"
authors = [{ name = "Sathish Kumar", email = "SathishKumar786.ML@gmail.com" }]
dependencies = [
    "torch", "transformers", "opencv-python", "scikit-learn", "matplotlib",
    "pandas", "numpy", "pydantic", "fastapi", "uvicorn", "pillow",
    "langchain", "openai", "pytube", "moviepy"
]
```

```bash
uv pip install .
# or
uv pip freeze > requirements.lock.txt
```

---

## 📁 Project Structure Explained

```yaml
Pickleball Vision Model
├── 📜 Root: Metadata, environment and versioning
│   ├── README.md, LICENSE, pyproject.toml, .gitignore
│   ├── environment.yml: Conda environment
│   ├── mlflow_setup.sh: MLflow config script
│   └── CONTRIBUTING.md, CODE_OF_CONDUCT.md, CHANGELOG.md
│
├── ⚙️ config/: Central config files
│   ├── config.yaml: General project configs
│   └── hyperparams.yaml: Model/training parameters
│
├── 📦 data/: Organized input/output data folders
│   ├── raw/: Unprocessed data
│   ├── processed/: Cleaned and labeled data
│   ├── interim/: Intermediate processing outputs
│   ├── external/: 3rd party sources
│   └── outputs/: Final exportable results
│
├── 📚 docs/: Technical documentation
│   ├── setup.md, architecture.md, inference_api.md, model_zoo.md
│   ├── vision_models.md, llm_prompts.md, evaluation_metrics.md
│   └── assets/: Images and diagrams (e.g., pipeline, YOLO output)
│
├── 🧠 src/: Core Python modules
│   ├── data_collection/: Download, sample, and track video frames
│   ├── preprocessing/: Frame/pose extraction, augmentation
│   ├── vision/: YOLO, DINO, DETR, and VideoMAE models
│   ├── vision/tracker/: ByteTrack implementation
│   ├── llm/: Feedback generation and prompt engineering
│   └── utils/: Reusable utilities (metrics, config, alerts)
│
├── 🔁 scripts/: One-off utility scripts
│   ├── train_model.py, run_mlflow_experiment.py
│   └── init_empty_files.py, download_assets.py
│
├── 🌐 app/: FastAPI backend and Streamlit UI
│   ├── endpoints/: REST endpoints (e.g., analyze, feedback)
│   ├── models/: Inference wrapper for deployed model
│   └── streamlit_ui/: Lightweight demo UI
│
├── 🎨 frontend/: Svelte-based frontend
│   ├── App.svelte, main.js
│   └── Tailwind CSS setup
│
├── 🐳 docker/: Docker environment
│   ├── Dockerfile, docker-compose.yml, start.sh
│
├── 🧪 evaluation/: Model benchmarking scripts
│   ├── benchmark_metrics.py, model_compare.py
│   └── confusion_matrix.py
│
├── 📓 notebooks/: R&D and experiment notebooks
│   ├── demo_pipeline.ipynb, model_ablation_study.ipynb
│
├── 🧪 tests/: Unit tests
│   ├── test_video_utils.py, test_pose_utils.py, test_api_endpoints.py
│
├── 💾 checkpoints/: Trained model checkpoints
├── 🪵 logs/: Log outputs
├── 📈 mlruns/: MLflow experiment runs
└── 🧬 .github/workflows/: CI/CD setup (linting, deployment)
```

---

## 🧪 Usage

```bash
conda activate pickleball_llm
uvicorn app.api_server:app --reload
# Frontend
cd frontend && npm install && npm run dev
```

---

## 🗺️ Development Roadmap

### P0 – Core
- ✅ Video ingestion
- ✅ Pose detection
- ✅ LLM feedback

### P1 – UI
- [ ] Interactive dashboard
- [ ] Annotation overlay

### P2 – Player Profiles
- [ ] Session history
- [ ] Skill level estimation

### P3 – Cloud Ready
- [ ] Postgres migration
- [ ] S3 video storage

---

## 🤝 Contributing

We welcome contributions! Please review `CONTRIBUTING.md` and open a PR.

---

## 📜 License

Apache License. See `LICENSE`.

---

## 📬 Contact

**Sathish Kumar**  
📧 SathishKumar786.ML@gmail.com
