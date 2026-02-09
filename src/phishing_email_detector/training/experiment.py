import tensorflow as tf
from pathlib import Path
from typing import Dict
from src.phishing_email_detector.models.registry import get_model, save_model, get_model_save_path, get_model_id
from src.phishing_email_detector.models.rnn import build_text_vectorizer, RNNModel
from src.phishing_email_detector.data.preprocessing import load_dataset, df_to_dataset
from src.phishing_email_detector.utils.logging import get_logger
from src.phishing_email_detector.utils.seeding import set_global_seed
from src.phishing_email_detector.utils.config import ExperimentConfig, DataConfig, FnnConfig, TrainConfig
from src.phishing_email_detector.training.evaluate import plot_model_metric


logger = get_logger(__name__)

class Experiment:
    """Unified experiment runner."""
    
    def __init__(
            self, 
            config: ExperimentConfig, 
            base_output_dir: Path = Path("results", "models"),
            debug: any = False,
            debug_size: int = 150
            ):
        self.config = config
        set_global_seed(config.train.seed)
        self.results: Dict = {}
        self.debug = debug
        self.debug_size = debug_size 
    def run(self):
        """Run complete experiment: load data, train, evaluate."""
        logger.info(f"Starting experiment with config: {self.config}")
        
        # Load data
        train_df, val_df, test_df = load_dataset(self.config.data,debug=self.debug,test_size=self.debug_size)
        train_ds = df_to_dataset(train_df, self.config.data.batch_size)
        val_ds = df_to_dataset(val_df, self.config.data.batch_size, shuffle=False)
        test_ds = df_to_dataset(test_df, self.config.data.batch_size, shuffle=False)
        
        # Build & train model
        logger.info(f"Building {self.config.model.model_type} model...")
        model_wrapper = get_model(self.config.model)
        keras_model = model_wrapper.build()
        
        # For Rnn models
        if self.config.model.model_type == "rnn":
            train_text = train_df["body"].values
            text_vectorizer = None
            for layer in keras_model.layers:
                if isinstance(layer, tf.keras.layers.TextVectorization):
                    text_vectorizer = layer
                    break
                if text_vectorizer is None:
                    raise RuntimeError("No TextVectorization layer found in Rnn model")

            text_vectorizer.adapt(train_text)
        

        # string to optimizer object
        if self.config.train.optimizer == "adamw":
            opt = tf.keras.optimizers.AdamW(learning_rate=self.config.train.learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=self.config.train.learning_rate)
         

        keras_model.compile(
            optimizer=opt,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"]
        )

       # Create model directory in results/models for model checkpoints
        model_checkpoint_dir = Path(get_model_save_path(
            model_id=get_model_id(self.config.model),
            base_dir=Path(self.config.output_dir),
            filename="model.keras",
            debug=self.debug
        ).parent,"checkpoints")

        print(f"Model checkpoints will be saved to: {model_checkpoint_dir}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_checkpoint_dir) + "/checkpoint_epoch_{epoch:02d}.keras",
            save_weights_only=False,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
        logger.info("Training...")
        history = keras_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.train.epochs,
            callbacks=[checkpoint_callback]
        )
        print(keras_model.summary())
        # Evaluate
        logger.info("Evaluating on test set...")
        test_loss, test_acc = keras_model.evaluate(test_ds)
        self.results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'history': history.history
        }
        logger.info("Plotting training history...")
        plot_model_metric(history=history, metric="accuracy", config=self.config.model, debug=self.debug)
        plot_model_metric(history=history, metric="loss", config=self.config.model, debug=self.debug)
        logger.info(f"Saving model...{get_model_id(self.config.model)}")
        save_path = save_model(
            model=model_wrapper.model,
            model_id=get_model_id(self.config.model),
            base_dir=Path(self.config.output_dir),
            filename="model.keras",
            debug=self.debug
        )
        print(f"Save path: {save_path}")
        logger.info(f"Model saved to {save_path}")

        logger.info(f"Test Accuracy: {test_acc:.4f}")
        return self.results

if __name__ == "__main__":
    exp = Experiment(ExperimentConfig(data=DataConfig(), model=FnnConfig(), train=TrainConfig()))
    exp.run()
