from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Tuple, Optional
from src.phishing_email_detector.utils.config import ModelConfig

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[tf.keras.Model] = None
    
    @abstractmethod
    def build(self) -> tf.keras.Model:
        """Build and return Keras model."""
        pass
    
    def compile(self, optimizer: str, learning_rate: float):
        """Compile model with standard settings."""
        self.model = self.build()
        
        if optimizer.lower() == "adamw":
            opt = tf.keras.optimizers.AdamW(learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate)
        
        self.model.compile(
            optimizer=opt,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model:
            print(self.model.summary())
            
