import os
import sys
import pandas as pd
import numpy as np

#import logging and exception modules
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    validation_data_path: str = os.path.join("artifacts", "validation.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(os.path.join("data", "CEAS_08.csv"))
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train validation test split initiated")
            train_set, temp_set = train_test_split(df, test_size=0.2, random_state=42,shuffle=True)
            validation_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42,shuffle=True)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            validation_set.to_csv(self.ingestion_config.validation_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) 
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.validation_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)