# necessary to make function on Macos
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
from typing import Dict
from src.phishing_email_detector.utils.config import FNNConfig, ModelConfig, TrainConfig
from .base import BaseModel

class FeedforwardModel(BaseModel):
    """
    Feedforward Neural Network class with NNLM embeddings
    RETURNS SEQUENTIAL OBJECT, IT ITSELF IS NOT THE SEQUENTIAL OBJECT
    """
    
    # TensorFlow Hub URL for NNLM embeddings (more compatible than kagglehub)
    NNLM_URL = "https://tfhub.dev/google/nnlm-en-dim128/2"
    
    def __init__(self, config: FNNConfig):
        super().__init__(config)
        self.config: FNNConfig = config
        print(f"Loading NNLM embeddings from TensorFlow Hub")
    
    def build(self) -> tf.keras.Model:
        """Build FNN model with NNLM embeddings."""
        # Load pre-trained NNLM embeddings from TensorFlow Hub
        hub_layer = hub.KerasLayer(
            self.NNLM_URL,
            dtype=tf.string,
            trainable=False
        )
        
        self.model = tf.keras.Sequential([
            hub_layer,
            *[layer for _ in range(self.config.num_layers) 
              for layer in [
                  tf.keras.layers.Dense(self.config.hidden_dim, activation='relu'),
                  tf.keras.layers.Dropout(self.config.dropout_rate)
              ]],
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return self.model