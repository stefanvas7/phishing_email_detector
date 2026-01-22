import click
from pathlib import Path
from src.phishing_email_detector.utils.config import ExperimentConfig
from src.phishing_email_detector.training.experiment import Experiment
from src.phishing_email_detector.utils.logging import get_logger

logger = get_logger(__name__)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              help='Path to YAML config file', 
              default='configs/feedforward.yaml')
@click.option('--output-dir', type=click.Path(), 
              help='Output directory for results',
              default='')
def main(config: str, output_dir: str):
    """Train a phishing detection model."""
    config = ExperimentConfig.from_yaml(config)
    config.output_dir = output_dir
    
    print(f"Starting training with config: {config}, output_dir: {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    exp = Experiment(config)
    results = exp.run()
    
    logger.info(f"Results saved to {output_dir}")
    print(f"\n✓ Test Accuracy: {results['test_accuracy']:.4f}")

if __name__ == '__main__':
    main()
