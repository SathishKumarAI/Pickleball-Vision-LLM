from pathlib import Path
import os

# Load environment variables (optional dependency — app still boots without it)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

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
