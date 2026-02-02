import click
from pathlib import Path
from src.phishing_email_detector.utils.config import ExperimentConfig, FNNConfig, DataConfig, TrainConfig 
#Testing configs
from src.phishing_email_detector.utils.config import Test_TrainConfig, Testing_DataConfig

from src.phishing_email_detector.training.experiment import Experiment
from src.phishing_email_detector.utils.logging import get_logger

logger = get_logger(__name__)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              help='Path to YAML config file', 
              default='src/phishing_email_detector/configs/feedforward.yaml')
@click.option('--output-dir', type=click.Path(file_okay=False, dir_okay=True, ), 
              help='Output directory for results',
              default='results/models')
@click.option('--predict', is_flag=True, help='Run model in prediction mode', default=False)
@click.option('--debug', is_flag=True, help='Enable debug logging', default=False)
def main(
    config: str = 'src/phishing_email_detector/configs/feedforward.yaml', 
    output_dir: str = 'results/models', 
    predict: bool = False, 
    debug: bool = False,
    model_type: str = 'fnn'
    ):
    """
    Train a phishing email detection model.
    Attributes:
        config (str): Path to YAML config file.
        output_dir (str): Directory to save output results.
        predict (bool): If True, run in prediction mode. Else, train the model.
    """
    if predict:
        """Run prediction on email input (not implemented)."""
        print("Prediction mode is not yet implemented.") 
        return
    elif debug:
        """Use dataclasses with much smaller data and training configurations"""
        print("Debug mode is used")
        config = ExperimentConfig(
                data=Testing_DataConfig(),
                model=FNNConfig(),
                train=Test_TrainConfig,
                output_dir=output_dir
                )
    else:
        """By default use YAML configuration"""
        config = ExperimentConfig.from_yaml(config)
    config.output_dir = output_dir
    output_dir: Path = Path(output_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    exp = Experiment(config=config, base_output_dir=output_dir, debug=debug)
    results = exp.run()
    
    logger.info(f"Results saved to {output_dir}")
    print(f"\n✓ Test Accuracy: {results['test_accuracy']:.4f}")

if __name__ == '__main__':
    main()
