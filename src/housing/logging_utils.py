import logging
import os


def configure_logging(log_level="INFO", log_path=None, console_log=True):
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    logging.debug(
        f"Logging configured. "
        f"Level: {log_level}, "
        f"File: {log_path}, "
        f"Console: {console_log}"
    )
