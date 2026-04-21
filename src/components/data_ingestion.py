import sys
import os
from src.logger import logger
from src.exception import CustomException

from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_source_path: str = os.path.join('data', 'Students_Performance.csv') # Update this path to point to the correct location of your dataset
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion method or component")
        try:
            if not os.path.exists(self.ingestion_config.raw_source_path):
                raise FileNotFoundError(f"Source data not found: {self.ingestion_config.raw_source_path}")

            df = pd.read_csv(self.ingestion_config.raw_source_path)
            logger.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # Create the directories for train and test data if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False) # Save the raw data to the specified path

            logger.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) # Save the training data to the specified path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) # Save the test data to the specified path

            logger.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()