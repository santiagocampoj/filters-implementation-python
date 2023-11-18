import logging

def setup_logger(log_file_path: str) -> logging.Logger:
    """Logger"""
    logger = logging.getLogger("low_pass_filter")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_header = logging.FileHandler(log_file_path)
    file_header.setLevel(logging.DEBUG)

    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_header.setFormatter(file_format)

    logger.addHandler(file_header)
    return logger