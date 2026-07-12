import logging


LOGGER_NAME = "chess_ai.log"

logger = logging.getLogger(LOGGER_NAME)


def configure_logging(level=logging.INFO) -> None:
    """Configure the application logger to write only to logs.log file."""
    logger.handlers.clear()
    logger.setLevel(level)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    file_handler = logging.FileHandler(LOGGER_NAME, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
