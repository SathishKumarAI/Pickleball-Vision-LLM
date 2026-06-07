# 🛠️ Setup & Development

Reproducible environment for Pickleball-Vision-LLM. Python **3.10–3.12**
(`pyproject.toml` pins `>=3.10,<3.13`; torch / opencv wheels lag on 3.13+).

---

## Quick start (minimal — boots the API)

```bash
# 1. create an isolated env (conda or venv)
conda create -n pickleball python=3.11 -y
conda activate pickleball

# 2. install core runtime only
pip install -r requirements.txt

# 3. run
python -m src.main            # Flask app on http://0.0.0.0:8000
curl localhost:8000/health    # -> {"status":"healthy"}
```

## Full install (vision + LLM + MLOps)

The project is an installable package with optional extras:

```bash
pip install -e ".[vision,llm,mlops,dev]"
```

| Extra | Pulls | For |
|-------|-------|-----|
| `vision` | opencv, ultralytics (YOLO), mediapipe, torch, torchvision, pillow, imagehash | detection / pose / tracking |
| `llm` | transformers, langchain, openai | captioning + coaching feedback |
| `mlops` | mlflow, uvicorn, fastapi | experiment tracking / serving |
| `dev` | pytest, ruff, pre-commit | tests + lint |

## Faster installs with `uv` (optional)

```bash
pip install uv
uv pip install -e ".[vision,llm]"
```

---

## Configuration

Runtime config is read from environment variables (see
`src/core/config/settings.py`), loaded from a `.env` file if `python-dotenv`
is installed. Defaults work out of the box.

| Var | Default | Meaning |
|-----|---------|---------|
| `API_HOST` | `0.0.0.0` | bind host |
| `API_PORT` | `8000` | bind port |
| `DEBUG` | `False` | Flask debug |
| `MODEL_PATH` | `models/default` | weights dir |
| `LOG_LEVEL` | `INFO` | log verbosity |

Example `.env`:

```dotenv
API_PORT=8080
DEBUG=true
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

> Note: tests are being brought back online (see `docs/PLAN.md`, Phase 4) —
> some import paths are still being fixed after the de-duplication refactor.

## Deployment

Docker + Compose live under `deployment/` (`Dockerfile`,
`docker-compose.yml`, monitoring via Prometheus/Alertmanager). See
`docs/PLAN.md` Phase 5 for the deploy validation checklist.

---

## References / Further reading
- [`uv` (Astral) — fast Python package manager](https://docs.astral.sh/uv/)
- [Conda — managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [Python packaging — pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [pytest](https://docs.pytest.org/) · [ruff](https://docs.astral.sh/ruff/)
- Internal: `docs/ROADMAP.md`, `docs/DELIVERY_PLAN.md`
