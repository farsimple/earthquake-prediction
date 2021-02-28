import logging
import sys

from autologging import TRACE


def configure_logger(name: str = None, level: str = None) -> None:
    """Configures logging.

    Sets log output format and logging level for specified logger.

    Args:
        name: Module name as a period-separated hierarchical value. Logger uses this name on creation.
            Defaults to root logger
        level: Logging level. Available values are: CRITICAL, ERROR, WARNING, INFO, DEBUG, TRACE. Defaults to INFO
    """
    logger = logging.getLogger() if name is None else logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setStream(sys.stdout)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if level is None:
        logger.setLevel(logging.INFO)
    else:
        if level.upper() == "TRACE":
            logger.setLevel(TRACE)
        else:
            logger.setLevel(level.upper())
