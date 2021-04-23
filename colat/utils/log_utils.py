import logging
import re
from typing import Union

from omegaconf import DictConfig, OmegaConf


def to_clean_str(s: str) -> str:
    """Keeps only alphanumeric characters and lowers them

    Args:
        s: a string

    Returns:
        cleaned string
    """
    return re.sub("[^a-zA-Z0-9]", "", s).lower()


def display_config(cfg: DictConfig) -> None:
    """Displays the configuration"""
    logger = logging.getLogger()
    logger.info("Configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 40 + "\n")


def flatten(d: Union[dict, list], parent_key: str = "", sep: str = ".") -> dict:
    """Flattens a dictionary or list into a flat dictionary

    Args:
        d: dictionary or list to flatten
        parent_key: key of parent dictionary
        sep: separator between key and child key

    Returns:
        flattened dictionary

    """
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.extend(flatten(v, new_key, sep=sep).items())
    elif isinstance(d, list):
        for i, elem in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten(elem, new_key, sep).items())
    else:
        items.append((parent_key, d))
    return dict(items)
