import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple
from src.phishing_email_detector.models.registry import get_model
from src.phishing_email_detector.data.preprocessing import load_dataset, df_to_dataset
from src.phishing_email_detector.utils.logging import get_logger
from src.phishing_email_detector.utils.seeding import set_seed
from src.phishing_email_detector.utils.config import ExperimentConfig

logger = get_logger(__name__)

class Experiment:
    """Unified experiment runner."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        set_seed(config.train.seed)
        self.results: Dict = {}
    
    def run(self):
        """Run complete experiment: load data, train, evaluate."""
        logger.info(f"Starting experiment with config: {self.config}")
        
        # Load data
        train_df, val_df, test_df = load_dataset(self.config.data)
        train_ds = df_to_dataset(train_df, self.config.data.batch_size)
        val_ds = df_to_dataset(val_df, self.config.data.batch_size, shuffle=False)
        test_ds = df_to_dataset(test_df, self.config.data.batch_size, shuffle=False)
        
        # Build & train model
        logger.info(f"Building {self.config.model.model_type} model...")
        model = get_model(self.config.model) # -> FNNConfig inherits base model which returns compiled optimizer and learning rate
        model.compile( # get_model -> model entry {model_cls:FeedforwardModel} FeedforwardModel -> tf.keras.Sequential 
            optimizer=self.config.train.optimizer,
            learning_rate=self.config.train.learning_rate
        )
        model.optimizer.learning_rate = self.config.train.learning_rate
        model.summary()
        
        logger.info("Training...")
        history = model.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.train.epochs
        )
        
        # Evaluate
        logger.info("Evaluating on test set...")
        test_loss, test_acc = model.model.evaluate(test_ds)
        self.results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'history': history.history
        }
        
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        return self.results

if __name__ == "__main__":
    exp = Experiment(ExperimentConfig)
    exp.run()