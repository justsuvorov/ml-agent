from loguru import logger


def pytest_configure() -> None:
    logger.remove()
