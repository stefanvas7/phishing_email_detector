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
from src.phishing_email_detector.utils.config import FFNConfig, ModelConfig
from .base import BaseModel

class FeedforwardModel(BaseModel):
    """Feedforward Neural Network with NNLM embeddings."""
    
    NNLM_URL = "https://www.kaggle.com/models/google/nnlm/TensorFlow2/en-dim128/1"
    
    def __init__(self, config: FFNConfig):
        super().__init__(config)
        self.config: FFNConfig = config
    
    def build(self) -> tf.keras.Model:
        """Build FNN model with NNLM embeddings."""
        # Load pre-trained NNLM embeddings
        hub_layer = hub.KerasLayer(
            self.NNLM_URL,
            dtype=tf.string,
            trainable=False
        )
        
        model = tf.keras.Sequential()
        model.add(hub_layer)
        
        # Hidden layers based on num_layers
        for _ in range(self.config.num_layers):
            model.add(tf.keras.layers.Dense(self.config.hidden_dim, activation='relu'))
            model.add(tf.keras.layers.Dropout(self.config.dropout_rate))
        
        # Output layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model