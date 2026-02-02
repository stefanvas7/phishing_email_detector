import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple
from src.phishing_email_detector.utils.logging import configure_logging, LoggingConfig, get_logger
from src.phishing_email_detector.utils.config import DataConfig
from src.phishing_email_detector.utils.seeding import SeedConfig


logger = get_logger(__name__)

def load_dataset(config: DataConfig, debug: bool = False, test_size: int = 150) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split email dataset."""
    logger.info(f"Loading dataset from {config.dataset_path}")
    df = pd.read_csv(config.dataset_path, usecols=['body', 'label'])
    
    # Clean
    df = df.dropna(subset=['body', 'label'])
    df['body'] = df['body'].astype(str)
    
    # Split
    if debug:
        df = df.sample(n=min(test_size, len(df)),random_state=SeedConfig.seed)
    df_shuffled = df.sample(frac=1, random_state=SeedConfig.seed)
    train_size = int(config.train_split * len(df))
    val_size = int(config.val_split * len(df))
    
    # Slicing into train, val and test datasets
    train = df_shuffled.iloc[:train_size]
    val = df_shuffled.iloc[train_size:train_size+val_size]
    test = df_shuffled.iloc[train_size+val_size:]
    
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

def df_to_dataset(dataframe: pd.DataFrame, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
    """
    Convert DataFrame to tf.data.Dataset.
    Args:
        dataframe (pd.DataFrame): Input DataFrame with 'body' and 'label' columns.
        batch_size (int): Batch size for the dataset.
        shuffle (bool): Whether to shuffle the dataset.
    Returns:
        tf.data.Dataset: Prepared dataset.
    """

    df = dataframe.copy()
    labels = df.pop('label')
    # texts = df['body']
    
    ds = tf.data.Dataset.from_tensor_slices((df['body'].values, labels.values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
