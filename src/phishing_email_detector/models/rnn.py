# necessary to make function on Macos
from json import encoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warning logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent OOM by allocating GPU memory gradually
os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL (Intel Math Kernel Library) threads
os.environ['OPENBLAS_NUM_THREADS'] = '4'  # Limit OpenBLAS threads
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'  # macOS Accelerate framework

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

import tensorflow_hub as hub
from typing import Optional, Literal
from src.phishing_email_detector.utils.config import RNNConfig, ModelConfig, TrainConfig
from .base import BaseModel

def build_text_vectorizer(max_tokens: int, adapt_dataset: Optional[tf.data.Dataset] = None) -> tf.keras.layers.TextVectorization:
    """
    Attributes:
    max_tokens: Maximum number of tokens (vocabulary size).
    adapt_dataset: Optional tf.data.Dataset to adapt the vectorizer on

    Returns:
    tf.keras.layers.TextVectorization
        Textvectorization layer to convert text to integer sequences.
    """

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize='lower_and_strip_punctuation',
        split='whitespace',
        output_mode='int',
        output_sequence_length=128, 
    )

    if adapt_dataset is not None:
        test_only_ds = adapt_dataset.map(lambda text, label: text)
        vectorizer.adapt(test_only_ds)
    
    return vectorizer


class RNNModel(BaseModel):
    """
    Recurrent Neural Network with Universal Sentence Encoder embeddings.
    returns sequential object, it itself is not the sequential object
    """

    def __init__(self, config: RNNConfig):
        super().__init__(config)
        self.config: RNNConfig = config
        self.vectorizer = build_text_vectorizer(
            max_tokens=self.config.max_tokens
        )

    def set_vectorizer(self, vectorizer: tf.keras.layers.TextVectorization):
        """
        Set the text vectorizer layer.
        NEEDS TO BE SET BEFORE BUILD OR COMPILE
        """
        self.vectorizer = vectorizer

    def build(self):
        """Build RNN model"""

        
        self.model = tf.keras.Sequential([
            self.vectorizer,
            tf.keras.layers.Embedding(
                input_dim=self.config.max_tokens,
                output_dim=self.config.embedding_dim,
                mask_zero=True
            ),
            *[layer for _ in range(self.config.num_layers) 
              for layer in [
                  tf.keras.layers.Bidirectional(
                      tf.keras.layers.LSTM(self.config.hidden_dim, return_sequences=False)
                  ),
                  tf.keras.layers.Dropout(self.config.dropout_rate)
              ]],
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        return self.model