import logging
import sys


def setup_logger(log_file_path):
    """
    Sets up a logger that writes to both the console and a specified file.

    Args:
        log_file_path (str): The full path to the log file.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')

    # Create a handler to write to the console (stdout)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Create a handler to write to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info("Logger setup complete.")
