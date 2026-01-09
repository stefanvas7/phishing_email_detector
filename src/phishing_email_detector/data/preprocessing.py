import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple
from src.phishing_email_detector.utils.logging import configure_logging, LoggingConfig, get_logger
from src.phishing_email_detector.utils.config import DataConfig


logger = get_logger(__name__)

def load_dataset(config: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split email dataset."""
    logger.info(f"Loading dataset from {config.dataset_path}")
    df = pd.read_csv(config.dataset_path, usecols=['body', 'label'])
    
    # Clean
    df = df.dropna(subset=['body', 'label'])
    df['body'] = df['body'].astype(str)
    
    # Split
    df_shuffled = df.sample(frac=1, random_state=42)
    train_size = int(config.train_split * len(df))
    val_size = int(config.val_split * len(df))
    
    train = df_shuffled.iloc[:train_size]
    val = df_shuffled.iloc[train_size:train_size+val_size]
    test = df_shuffled.iloc[train_size+val_size:]
    
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

def df_to_dataset(dataframe: pd.DataFrame, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
    """Convert DataFrame to tf.data.Dataset."""
    df = dataframe.copy()
    labels = df.pop('label')
    # texts = df['body']
    
    ds = tf.data.Dataset.from_tensor_slices((df, labels.values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds