# -*- coding: utf-8 -*-
import logging
from pkg_resources import DistributionNotFound, get_distribution

from eqml.helpers import configure_logger

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

# Configure logging
logger = logging.getLogger(__name__)
logging.captureWarnings(True)
configure_logger(level="info")
