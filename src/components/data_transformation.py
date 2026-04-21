import sys
import os
from src.logger import logger
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    transformed_train_data_path: str = os.path.join('artifacts', 'transformed_train.csv')# Update this path to point to the correct location where you want to save the transformed training data
    transformed_test_data_path: str = os.path.join('artifacts', 'transformed_test.csv')
    preprocessor_object_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['math score', 'reading score', 'writing score', 'Total Score']
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            num_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        logger.info("Entered the data transformation method or component")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor = self.get_data_transformer_object()

            target_column_name = 'Average Score'
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            save_object(file_path=self.data_transformation_config.preprocessor_object_path, obj=preprocessor)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # Concatenate the transformed input features with the target variable for the training set
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_data_path), exist_ok=True) # Create the directories for transformed train and test data if they don't exist
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_data_path), exist_ok=True)

            pd.DataFrame(train_arr).to_csv(self.data_transformation_config.transformed_train_data_path, index=False) # Save the transformed training data to the specified path
            pd.DataFrame(test_arr).to_csv(self.data_transformation_config.transformed_test_data_path, index=False) # Save the transformed test data to the specified path
            logger.info("Data transformation is completed")

            return (
                self.data_transformation_config.transformed_train_data_path,
                self.data_transformation_config.transformed_test_data_path,
                preprocessor
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataTransformation()
    train_path = os.path.join('artifacts', 'train.csv')
    test_path = os.path.join('artifacts', 'test.csv')
    obj.initiate_data_transformation(train_path, test_path)

    