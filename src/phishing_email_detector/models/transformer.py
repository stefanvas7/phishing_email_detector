# The transformer model implementation isn't completed or functional 
# due to tensorflow-text incompatibility with M-series processors
# 

# # necessary to make function on Macos
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warning logs
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent OOM by allocating GPU memory gradually
# os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
# os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL (Intel Math Kernel Library) threads
# os.environ['OPENBLAS_NUM_THREADS'] = '4'  # Limit OpenBLAS threads
# os.environ['VECLIB_MAXIMUM_THREADS'] = '4'  # macOS Accelerate framework

import tensorflow as tf


tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)


import tensorflow_hub as hub
from typing import Dict
from src.phishing_email_detector.utils.config import TransformerConfig, ModelConfig, TrainConfig
from .base import BaseModel

BERT_MODELS_URLS = {
    "h-128": {
        "encoder": "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-12-h-128-a-2/2"
    },
    "h-768": {
        "encoder": "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-12-h-768-a-12/2"
    }
}

PREPROCESSOR_URL = "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3"



class TransformerModel(BaseModel):
    """
    Transformer Neural Network class with BERT embeddings
    RETURNS SEQUENTIAL OBJECT, IT ITSELF IS NOT THE SEQUENTIAL OBJECT
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.config: TransformerConfig = config
        
        model_variant = self.config.model_variant
        if model_variant not in BERT_MODELS_URLS:
            raise ValueError(f"Unsupported model_variant: {model_variant}. Expected one of {list(BERT_MODELS_URLS.keys())}.")

        self.bert_encoder_url = BERT_MODELS_URLS[model_variant]
        print(f"URL: {self.bert_encoder_url}")
        self.preprocessor_url = PREPROCESSOR_URL
    
    def build(self) -> tf.keras.Model:
        """Build Transformer model with BERT embeddings."""
        # Load pre-trained BERT embeddings from TensorFlow Hub
        preprocessor = hub.KerasLayer(self.preprocessor_url, name='preprocessing')
        encoder = hub.KerasLayer(self.bert_encoder_url, trainable=True, name='BERT_encoder')

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = preprocessor(text_input)
        outputs = encoder(preprocessed_text)

        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(self.config.dropout_rate)(net)
        
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)

        self.model = tf.keras.Model(inputs=text_input, outputs=net)
        DUMMY_INPUT = tf.constant(["dummy"] * 32)  # Match batch_size      
        return self.model