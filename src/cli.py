import click
from pathlib import Path
from src.phishing_email_detector.utils.config import ExperimentConfig, FNNConfig 
from src.phishing_email_detector.training.experiment import Experiment
from src.phishing_email_detector.utils.logging import get_logger

logger = get_logger(__name__)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              help='Path to YAML config file', 
              default='src/phishing_email_detector/configs/feedforward.yaml')
@click.option('--output-dir', type=click.Path(), 
              help='Output directory for results',
              default='results/models')
@click.option('--predict', is_flag=True, help='Run model in prediction mode')
def main(config: str = 'configs/feedforward.yaml', output_dir: str = 'results/models', predict: bool = False):
    if predict:
        """Run prediction on email input (not implemented)."""
        print("Prediction mode is not yet implemented.")
        pass
    """Train a phishing detection model."""
    config = ExperimentConfig.from_yaml(config)
    config.output_dir = output_dir
    output_dir = Path(output_dir)
    
    print(f"Starting training with config: {config}, output_dir: {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    exp = Experiment(config)
    results = exp.run()
    
    logger.info(f"Results saved to {output_dir}")
    print(f"\n✓ Test Accuracy: {results['test_accuracy']:.4f}")

if __name__ == '__main__':
    main()
