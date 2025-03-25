import logging
import sys

def setup_logger(log_file = "app.log"):
    """Configure and return a logger that writes to a file"""
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s -%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Return a logger instance
    return logging.getLogger(__name__)