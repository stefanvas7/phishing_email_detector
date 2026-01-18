import pytest
import tensorflow as tf
from src.phishing_email_detector.models.registry import get_model
from src.phishing_email_detector.utils.config import FNNConfig, TransformerConfig

def test_fnn_build():
    """Smoke test FNN model instantiation."""
    config = FNNConfig()
    model = get_model(config)
    # assertion error if model is not compiled
    model.compile(optimizer="adamw", learning_rate=0.001)
    assert model.model is not None
    assert len(model.model.layers) > 0

def test_forward_pass():
    """Test model forward pass on dummy data."""
    config = FNNConfig()
    model = get_model(config)
    
    # Dummy input
    dummy_text = tf.constant(["This is a test email"])
    model.compile(optimizer="adamw", learning_rate=0.001)
    output = model.model(dummy_text)
    assert output.shape == (1, 1)