from pathlib import Path
import os
from dotenv import load_dotenv

def load_config():
    """Load configuration from environment files."""
    env_file = os.getenv('ENV_FILE', 'development.env')
    env_path = Path(__file__).parent / env_file
    load_dotenv(env_path)
    return {
        'model_path': os.getenv('MODEL_PATH'),
        'api_key': os.getenv('API_KEY'),
        'debug': os.getenv('DEBUG', 'False').lower() == 'true'
    }
