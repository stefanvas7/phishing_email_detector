"""
Dataclasses directly configure code components.

YAML config exist for more user friendly configuration parsed using from_yaml() function.

"""

from dataclasses import dataclass, asdict
import os
from typing import Optional, Dict, Any, Literal
import yaml
from pathlib import Path

@dataclass
class DataConfig:
    """Data loading configuration."""
    dataset_path: str = "data/raw/CEAS_08.csv"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 32
    max_tokens: int = 2000  # For RNN vectorization
    
@dataclass
class ModelConfig:
    """Base model configuration."""
    model_type: Literal["fnn", "rnn", "lstm", "transformer"] = "transformer"
    dropout_rate: float = 0.2
    num_layers: int = 1

@dataclass
class FFNConfig(ModelConfig):
    """Feedforward network configuration."""
    model_type: str = "fnn"
    embedding_dim: int = 128  # NNLM embedding dimension
    hidden_dim: int = 16
    num_layers: int = 1

@dataclass
class TransformerConfig(ModelConfig):
    """Transformer configuration."""
    model_type: str = "transformer"
    bert_variant: Literal["h-128", "h-768"] = "h-768"
    dropout_rate: float = 0.4

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 5
    learning_rate: float = 3e-5
    optimizer: str = "adamw"  # For transformers
    loss_fn: str = "binary_crossentropy"
    warmup_fraction: float = 0.1
    seed: int = 42

@dataclass
class ExperimentConfig:
    """Root experiment configuration."""
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    output_dir: str = "results/"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        # Parse each section
        data = DataConfig(**cfg_dict.get("data", {}))
        model_dict = cfg_dict.get("model", {})
        model_type = model_dict.get("model_type", "transformer")
        
        if model_type == "fnn":
            model = FFNConfig(**model_dict)
        # elif model_type == "transformer":
        #     model = TransformerConfig(**model_dict)
        # elif model_type == "rnn":
        #     model = RNNConfig(**model_dict)
        # elif model_type == "lstm":
        #     model = LSTMConfig(**model_dict)

        
        train = TrainConfig(**cfg_dict.get("train", {}))
        return cls(data=data, model=model, train=train)
