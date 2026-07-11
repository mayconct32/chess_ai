import logging


LOGGER_NAME = "chess_ai"

logger = logging.getLogger(LOGGER_NAME)


def configure_logging(level=logging.INFO) -> None:
    """Configure the application logger once from the entrypoint."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(level)
