import warnings
# Silence the pkg_resources deprecation warning from tensorflow_hub
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="At this time, the v2.11\\+ optimizer `tf.keras.optimizers.AdamW` runs slowly on M1/M2 Macs.*",
)

from src.phishing_email_detector.utils.logging import configure_logging, LoggingConfig, get_logger
import click
from pathlib import Path
from src.phishing_email_detector.utils.config import ExperimentConfig, FnnConfig, RnnConfig, TransformerConfig, DataConfig, TrainConfig 
#Testing configs
from src.phishing_email_detector.utils.config import Testing_TrainConfig, Testing_DataConfig

from src.phishing_email_detector.training.experiment import Experiment
from src.phishing_email_detector.utils.logging import get_logger

logger = get_logger(__name__)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              help='Path to YAML config file', 
              default='src/phishing_email_detector/configs/feedforward.yaml')
@click.option('--model-type', type=click.Choice(['fnn', 'rnn', 'lstm', 'transformer']),
              help='Type of model to train', 
              default='fnn',
              flag_value='fnn',
              is_flag=False)
@click.option('--output-dir', type=click.Path(file_okay=False, dir_okay=True, ), 
              help='Output directory for results',
              default='results/models')
@click.option('--predict', is_flag=True, help='Run model in prediction mode', default=False)
@click.option('--debug', is_flag=True, help='Enable debug logging', default=False)
def main(
    config, 
    output_dir, 
    predict, 
    debug,
    model_type,
    ):
    """
    Train a phishing email detection model.
    Attributes:
        config (str): Path to YAML config file.
        output_dir (str): Directory to save output results.
        predict (bool): If True, run in prediction mode. Else, train the model.

    """
    print("Configure logging")
    from src.phishing_email_detector.utils.logging import configure_logging, LoggingConfig, get_logger

    """
    Setup logging for a run with default configuration.
    Copy import and code below into every script run
    """
    config = LoggingConfig(
        level="INFO",
        console_level="INFO",
        file_level="INFO",
        log_dir="logs",
        log_file_name="phishing_detector.log",
        max_bytes=10 * 1024 * 1024,  # 10 MB
        backup_count=5
    )
    configure_logging(config)


    model_type_mapping = {
            'fnn' : FnnConfig(),
            'rnn' : RnnConfig(),
            'transformer' : TransformerConfig()
            }
    model_type_cls = model_type_mapping.get(model_type)
    print(f"Mapped model_config: {model_type_cls}")
    if predict:
        """Run prediction on email input (not implemented)."""
        logger.info("Prediction mode not implemented yet") 
        return
    elif debug:
        """Use dataclasses with much smaller data and training configurations"""
        logger.info("DEBUG MODE ENABLED: Using testing configurations with smaller dataset and fewer epochs")
        config = ExperimentConfig(
                data=Testing_DataConfig(),
                model=model_type_cls,
                train=Testing_TrainConfig(),
                output_dir=output_dir
                )
    else:
        """By default use YAML configuration"""
        config = ExperimentConfig.from_yaml(config)
    # if output_dir is specified change the config output path
    print(f"Output_dir =: {output_dir}")
    config.model = model_type_cls
    config.output_dir = output_dir
    output_dir: Path = Path(output_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    exp = Experiment(config=config, base_output_dir=config.output_dir, debug=debug, debug_size=config.data.test_subset_size)
    results = exp.run()
    
    logger.info(f"Results saved to {output_dir}")
    print(f"\n✓ Test Accuracy: {results['test_accuracy']:.4f}")

if __name__ == '__main__':
    main()
