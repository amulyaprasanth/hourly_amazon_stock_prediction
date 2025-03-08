import logging
import datetime
import os

def setup_logger(log_variable="my_app", log_dir="../logs"):
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{log_variable}_{timestamp}.log")

    # Create a logger
    logger = logging.getLogger(log_variable)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if not logger.hasHandlers():
        # File Handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(stream_handler)

    # Debugging: Log the file path (useful for pytest debugging)
    logger.debug(f"Logger initialized. Log file: {log_filename}")

    return logger