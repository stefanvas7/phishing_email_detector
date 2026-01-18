from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import tensorflow as tf
import pandas as pd

@dataclass
class SeedConfig:
    """
    Attributes:
    seed: Optional[int]
        A positive integer seed for random number generation. Default is 42
    """

    seed: int = 42

def _set_python_random_seed(seed: int) -> None:
    """
    Set the seed for the built-in random module.
    Attributes:
    seed: int
        A positive integer seed for random number generation.
    """
    random.seed(seed)

def _set_numpy_random_seed(seed: int) -> None:
    """
    Set the seed for numpy's random number generator.
    Attributes:
    seed: int
        A positive integer seed for random number generation.
    """
    np.random.seed(seed)

def _set_tensorflow_random_seed(seed: int) -> None:
    """
    Set the seed for TensorFlow's random number generator.
    Attributes:
    seed: int
        A positive integer seed for random number generation.
    """
    tf.random.set_seed(seed)

def set_global_seed(cfg_or_seed: int | SeedConfig) -> Dict[str, int]:
    """
    Set the global random seed for various libraries.
    Attributes:
    cfg_or_seed: int | SeedConfig
        Either a positive integer seed or a SeedConfig object containing the seed.
    
    Returns:
    Dict[str, int]
        A dictionary containing the seed used for each library to be logged
    """
    
    if isinstance(cfg_or_seed, int):
        cfg = SeedConfig(seed=cfg_or_seed)
    elif isinstance(cfg_or_seed, SeedConfig):
        cfg = cfg_or_seed
    else:
        raise TypeError(f"cfg_or_seed must be either an int or a SeedConfig instance. Got {type(cfg_or_seed)}")
    
    _set_python_random_seed(cfg.seed)
    _set_numpy_random_seed(cfg.seed)
    _set_tensorflow_random_seed(cfg.seed)

    return {
        "python_random_seed": cfg.seed,
        "numpy_random_seed": cfg.seed,
        "tensorflow_random_seed": cfg.seed,
    }