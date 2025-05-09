# 🏓 Pickleball LLM Project: Modern Environment Setup

To align with modern development practices and make your **Pickleball LLM project** reproducible, isolated, and manageable, this guide will help you:

1. ✅ Create a **Conda environment** with Python 3.11.
2. ⚡ Use **`uv`**, an ultrafast Python package manager from Astral.
3. 🗃️ Manage your dependencies using **`pyproject.toml`**.
4. 🔐 Generate lock files for consistency.

---

## ✅ Step 1: Conda Environment Setup

We'll define a `conda` environment YAML file and include `uv` as the package manager:

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

### 📦 Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate pickleball_llm
```

---

## ⚡ Step 2: Use `uv` to Manage Dependencies

We use a `pyproject.toml` file for modern Python dependency management:

```toml
# pyproject.toml
[project]
name = "pickleball_llm"
version = "0.1.0"
description = "Multimodal LLM model for analyzing pickleball gameplay"
authors = [{ name = "Sathish Kumar", email = "SathishKumar786.ML@gmail.com" }]
dependencies = [
    "torch",
    "transformers",
    "opencv-python",
    "scikit-learn",
    "matplotlib",
    "pandas",
    "numpy",
    "pydantic",
    "fastapi",
    "uvicorn",
    "pillow",
    "langchain",
    "openai",
    "pytube",
    "moviepy"
]
```

### 📥 Install Dependencies

You can install everything using `uv`:

```bash
uv pip install -r requirements.txt
# or
uv pip install .
```

---

## 🔐 Step 3: Lock Your Environment

Lock the versions of your dependencies for reproducibility:

```bash
uv pip freeze > requirements.lock.txt
```

---

## 🔁 Optional: Convert Existing `requirements.txt`

If you already have a `requirements.txt` file, use `uv` to install it:

```bash
uv pip install -r requirements.txt
```

---

## 📁 Files You Should Have

- `environment.yml` → Conda environment spec.
- `pyproject.toml` → Project metadata and dependencies.
- `requirements.lock.txt` → Frozen package versions.

---

## 🧠 About `uv`

[`uv`](https://github.com/astral-sh/uv) is a blazing-fast Python package manager and a drop-in replacement for pip. It's written in Rust and handles dependency resolution quickly and accurately.

---

> 🚀 This setup ensures a fast, reproducible, and modern Python environment for developing your Pickleball LLM project.