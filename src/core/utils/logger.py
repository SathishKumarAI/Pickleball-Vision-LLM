import logging
import os
from datetime import datetime

def setup_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with the specified name and log directory.

    Args:
        name (str): Name of the logger.
        log_dir (str): Directory where log files will be stored. Defaults to "logs".
        level (int): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create a timestamped log file
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    # Configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create file handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Define a common log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Example usage
if __name__ == "__main__":
    logger = setup_logger("pickleball_vision")
    logger.info("Logger is set up and ready to use.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")