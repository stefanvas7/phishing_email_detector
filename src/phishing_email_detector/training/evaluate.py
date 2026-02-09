from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
from src.phishing_email_detector.models.registry import get_model_save_path, get_model_id
from pathlib import Path

if TYPE_CHECKING:
    from src.phishing_email_detector.utils.config import FnnConfig, RnnConfig, TransformerConfig
    import tensorflow as tf

def plot_model_metric(history: tf.keras.callbacks.History,
                        metric: str,
                        config: FnnConfig | RnnConfig | TransformerConfig,
                        base_dir: str | Path = Path("results", "models"),
                        debug: bool = False,
                        save: bool = True) -> None:
  metric_save_path = metric_save_path = Path(get_model_save_path(
        model_id=get_model_id(config), 
        base_dir=Path(base_dir), 
        filename="model.keras",
        debug=debug).parent, "metrics")
  plt.plot(history.history[metric], label=metric)
  plt.plot(history.history[f"val_{metric}"], label=f"validation {metric}")
  model_name = get_model_id(config)
  plt.title(f"{metric.capitalize()} : {model_name}")
  plt.ylabel(metric)
  plt.xlabel("epoch")
  plt.legend()
  if save:
    plt.savefig(metric_save_path / f"{metric}.png")
  plt.show()