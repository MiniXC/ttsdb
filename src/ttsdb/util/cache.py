import os
from pathlib import Path

import numpy as np

CACHE_DIR = os.getenv("TTSDB_CACHE_DIR", os.path.expanduser("~/.cache/ttsdb"))
CACHE_DIR = Path(CACHE_DIR)
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache(obj: np.ndarray, name: str) -> np.ndarray:
    """
    Cache a numpy array to disk.

    Args:
        obj (np.ndarray): The numpy array to cache.
        name (str): The name of the cache file.

    Returns:
        np.ndarray: The cached numpy array.
    """
    cache_file = CACHE_DIR / f"{name}.npy"
    np.save(cache_file, obj)
    return obj


def load_cache(name: str) -> np.ndarray:
    """
    Load a cached numpy array from disk.

    Args:
        name (str): The name of the cache file.

    Returns:
        np.ndarray: The cached numpy array.
    """
    cache_file = CACHE_DIR / f"{name}.npy"
    return np.load(cache_file)


def check_cache(name: str) -> bool:
    """
    Check if a cache file exists.

    Args:
        name (str): The name of the cache file.

    Returns:
        bool: True if the cache file exists, False otherwise.
    """
    cache_file = CACHE_DIR / f"{name}.npy"
    return cache_file.exists()
