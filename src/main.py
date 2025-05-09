from src.api import create_app
from src.core.config.settings import API_HOST, API_PORT, DEBUG
from src.core.utils.logger import setup_logger

def main():
    """Main application entry point."""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting Pickleball Vision LLM application")
    
    # Create and run application
    app = create_app()
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)

if __name__ == "__main__":
    main()
