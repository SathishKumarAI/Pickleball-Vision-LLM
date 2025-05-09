import subprocess
import sys
import os
import time
import threading
from ..database.init_db import main as init_db

def run_fastapi():
    """Run FastAPI server."""
    subprocess.run([
        "uvicorn",
        "pickleball_vision.web.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

def run_streamlit():
    """Run Streamlit dashboard."""
    subprocess.run([
        "streamlit",
        "run",
        os.path.join(os.path.dirname(__file__), "dashboard.py"),
        "--server.port", "8501"
    ])

def main():
    """Initialize database and start servers."""
    # Initialize database
    print("Initializing database...")
    init_db()
    print("Database initialized")
    
    # Start FastAPI server in a separate thread
    print("Starting FastAPI server...")
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()
    
    # Wait for FastAPI server to start
    time.sleep(2)
    
    # Start Streamlit dashboard
    print("Starting Streamlit dashboard...")
    run_streamlit()

if __name__ == "__main__":
    main() 