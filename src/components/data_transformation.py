import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
# from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")
            
            numerical_columns = ["urls"]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            # no categorical features
            logging.info("Numerical columns standard scaling completed")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    # ("cat_pipeline", cat_pipeline, categorical_columns)  --- IGNORE  no categorical features --
                ]
            )



            return preprocessor
            

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, validation_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            validation_df = pd.read_csv(validation_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train, validation and test data completed")

            logging.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "label"
            numerical_columns = ["urls"]


            logging.info("Initiating separation of input and target features from train, validation and test data")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            logging.info("Separated input and target features from training data")  

            input_feature_validation_df = validation_df.drop(columns=[target_column_name], axis=1)
            target_feature_validation_df = validation_df[target_column_name]
            logging.info("Separated input and target features from validation data")

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Separated input and target features from test data")


            logging.info("Applying preprocessing object on training data and validation data")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_validation_arr = preprocessor_obj.transform(input_feature_validation_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            
        except Exception as e:
            raise CustomException(e, sys)