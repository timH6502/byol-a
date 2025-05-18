import logging
import random
import sys

import numpy as np
import torch


def init_logger(logger_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Initialize a logger with the specified name.

    Parameters
    ----------
    logger_name : str
        The name of the logger.
    level : int, default=20 (INFO)
        The logging level.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
