from __future__ import annotations

# Use to quieten output when running
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)


from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Dict, Type, Any, Tuple, Optional, Literal

import tensorflow as tf

from src.phishing_email_detector.models.feedforward import FeedforwardModel
from src.phishing_email_detector.models.rnn import RNNModel
# from src.phishing_email_detector.models.transformer import TransformerModel

from src.phishing_email_detector.utils.config import ModelConfig, FNNConfig, RNNConfig#, TransformerConfig

class ModelRegistryError(Exception):
    pass
class ModelEntry:
    """
    Attributes:
        key: short string identifying network type, eg. fnn, lstm, transformer, etc.
        model_cls: class that implements model family
        config_type: The dataclass type that holds the models configuration
        id_fields: Names of config fields that define the models ID. Used to build unique model_id
    """

    def __init__(
            self, key: str, 
            model_cls: Type[Any], 
            config_type: Type[ModelConfig], 
            id_fields: Tuple[str, ...]
            ) -> None:
        self.key = key
        self.model_cls = model_cls
        self.config_type = config_type
        self.id_fields = id_fields


# Dictionary maps model_type string to ModelEntry
# expected to live on config objects (config.model_type == ... etc.)
_MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "fnn": ModelEntry(
        key="fnn",
        model_cls=FeedforwardModel,
        config_type=FNNConfig,
        id_fields=("model_type", "num_layers","dropout_rate")
    ),
    "rnn": ModelEntry(
        key="rnn",
        model_cls=RNNModel,
        config_type=RNNConfig,
        id_fields=("model_type", "num_layers","hidden_dim","dropout_rate")
    ),
    # To add new architecture add entry here
}

def get_model(config: ModelConfig) -> Any:
    """
    Args:
        config:
            dataclass insance for the model configuration. Necessary to have model_type attribute matching key in _MODEL_REGISTRY
    Returns:
        Instance of appropriate model class, initialized with the config

    ModelRegistryError raised if model_type is unknown or config doesnt match registry 
    TODO Add option to train model with config if it doesnt exist
    """
    # check if config is dataclass
    if not is_dataclass(config):
        raise ModelRegistryError(f"Config object must be a dataclass, got: {type(config)}")
    # check if config has correct model_type attribute
    if not hasattr(config, "model_type"):
        raise ModelRegistryError("Config must have a 'model_type' attribute")

    model_type: str = getattr(config, "model_type") # "fnn", "rnn", etc.

    # check if model_type attribute has known model type from _MDOEL_REGISTRY
    entry = _MODEL_REGISTRY.get(model_type)
    if entry is None:
        raise ModelRegistryError(
            f"Unknown model_type '{model_type}'. "
            f"Known types: {list(_MODEL_REGISTRY.keys())}"
        )
    
    # check if config is instance of expected config_type
    if not isinstance(config, entry.config_type):
        raise ModelRegistryError(
            f"Config for model_type {model_type} is not instance of "
            f"{entry.config_type.__name__}, got: {type(config).__name__}"
        )
    
    
    model_instance = entry.model_cls(config)
    return model_instance

    
def get_registered_model_types() -> Dict[str, ModelEntry]:
    """
    returns the registry mapping model_type to ModelEntry
    """
    return dict(_MODEL_REGISTRY)

def get_model_id(config: ModelConfig) -> str:
    """
    Used to differentiate between variations of same architecture, eg. same FNN but different dropout rate

    Args:
        config:
            Model config dataclass (eg. 'model_type': 'fnn', 'dropout_rate': 0.2, 'num_layers': 1, 'embedding_dim': 128, 'hidden_dim': 16)
    Returns:
        String id 
    """
    if not is_dataclass(config):
        raise ModelRegistryError("Config must be dataclass")

    if not hasattr(config, "model_type"):
        raise ModelRegistryError("Config must have a 'model_type attribute")

    model_type: str = getattr(config, "model_type")
    entry = _MODEL_REGISTRY.get(model_type)
    if entry is None:
        raise ModelRegistryError(f"Unknown model_type '{model_type}' for model_id generation")
    
    # to flat dictionary
    config_dict = asdict(config)
# (REMINDER example FNNConfig) == {'model_type': 'fnn', 'dropout_rate': 0.2, 'num_layers': 1, 'embedding_dim': 128, 'hidden_dim': 16}


    parts = []
    # id_fields=("model_type", "num_layers","dropout_rate")
    _shorten_field_name_mapping = {
        "num_layers": "layers",
        "hidden_dim": "hiddenDim",
        "hidden_size": "hiddenSize",
        "dropout_rate": "dr",
        "bert_variant": "bert",
        "embedding_dim": "embeddingDim",
        "max_tokens": "maxTokens",
    }
    for field in entry.id_fields:
        if field not in config_dict:
            raise ModelRegistryError(f"Field '{field}' missing from config of type {type(config).__name__}, required for model_id generation")
        value = config_dict[field]
        if field == "model_type":
            parts.append(str(value))
        else:
            shortened = _shorten_field_name_mapping.get(field,field)
            parts.append(f"{shortened}{value}")

    model_id = "_".join(parts)
    if prefix:
        model_id = f"{prefix}_{model_id}"
    return model_id

def get_model_save_path(
        model_id: str,
        base_dir: str | Path = Path("results", "models"),
        filename: str = "model.keras",
) -> Path:
    
    """
    Compute path where model should be saved given 'model_id'
    Example consisten directory structure:
        base_dir/
            fnn_layers2_hidden16_dr0.2/
                model.keras
                metrics.json
            transformer_h-768_dr0.4/
                model.keras
                metrics.json
    
                
    Args:
        base_dir:
            default results/models/[model_id]/model.keras
            Base directory for model outputs, eg. "results/models"
        model_id:
            Identifier from 'get_model_id(config)' function
        filename:
            File name for the model, default "model.keras", keras v3 format
    
    Returns:
        Path
    """
    base = Path(base_dir)
    dir_path = base / model_id
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / filename

def load_saved_model(
        model_id: str,
        base_dir: str | Path = Path("results", "models"),
        filename: str = "model.keras",
        compile: bool = False,
) -> tf.keras.Model:
        """
        Load previously saved tf.keras model with model_id

        Args:
            base_dir:
                default results/models
                Base directory where models are saved
            model_id:
                Model identifier
            filename:
                Model file name
            compile:
                Whether to compile the model after loading. If false, can be compiled manually with specific optimizer/loss/metrics

        Returns:
            A tf.keras.Model instance
        """

        path = get_model_save_path(model_id=model_id,base_dir=base_dir,filename=filename)
        if not path.exists():
            raise FileNotFoundError(f"No model file found at {path}")
        return tf.keras.models.load_model(path,compile=compile)

def save_model(
        model: tf.keras.Model,
        model_id: str,
        base_dir: str | Path = Path("results", "models"),
        filename: str = "model.keras",
):
    # TODO test with trained model
    """
    Save tf.keras model to disk with model_id

    Args:
        model:
            tf.keras.Model instance to save
        base_dir:
            default results/models
            Base directory where models are saved
        model_id:
            Model identifier
        filename:
            Model file name
    """
    path = get_model_save_path(model_id=model_id,base_dir=base_dir,filename=filename)
    model.save_model(path )