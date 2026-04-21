import os
import sys
import pickle
import pandas as pd
from src.exception import CustomException


def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


class PredictPipeline:
    def predict(self, features: pd.DataFrame):
        try:
            preprocessor = load_object(os.path.join('artifacts', 'preprocessor.pkl'))
            model = load_object(os.path.join('artifacts', 'model.pkl'))
            transformed = preprocessor.transform(features)
            return model.predict(transformed)
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, math_score, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.math_score = float(math_score)
        self.reading_score = float(reading_score)
        self.writing_score = float(writing_score)

    def as_dataframe(self):
        total_score = self.math_score + self.reading_score + self.writing_score
        return pd.DataFrame([{
            "math score": self.math_score,
            "reading score": self.reading_score,
            "writing score": self.writing_score,
            "Total Score": total_score,
            "gender": self.gender,
            "race/ethnicity": self.race_ethnicity,
            "parental level of education": self.parental_level_of_education,
            "lunch": self.lunch,
            "test preparation course": self.test_preparation_course,
        }])
