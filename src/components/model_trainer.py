import sys
import os
from src.logger import logger
from src.exception import CustomException
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'model.pkl')


MODELS = {
    "Linear Regression": (LinearRegression(), {}),
    "Ridge": (Ridge(), {"alpha": [0.1, 1.0, 10.0]}),
    "Lasso": (Lasso(), {"alpha": [0.1, 1.0, 10.0]}),
}


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test):
        results = {}
        for name, (model, params) in MODELS.items():
            if params:
                gs = GridSearchCV(model, params, cv=3, scoring='r2', n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                logger.info(f"{name} best params: {gs.best_params_}")
            else:
                best_model = model.fit(X_train, y_train)

            r2 = r2_score(y_test, best_model.predict(X_test))
            results[name] = (best_model, r2)
            logger.info(f"{name} R2: {r2:.4f}")

        return results

    def initiate_model_trainer(self, train_path, test_path):
        logger.info("Entered the model trainer method or component")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            results = self.evaluate_models(X_train, y_train, X_test, y_test)

            best_name = max(results, key=lambda k: results[k][1])
            best_model, best_r2 = results[best_name]

            logger.info(f"Best model: {best_name} with R2 score: {best_r2:.4f}")
            print(f"Best model: {best_name} | R2: {best_r2:.4f}")

            save_object(file_path=self.model_trainer_config.trained_model_path, obj=best_model)

            return best_r2

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        train_path = os.path.join('artifacts', 'transformed_train.csv')
        test_path = os.path.join('artifacts', 'transformed_test.csv')

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_path, test_path)
    except Exception as e:
        raise CustomException(e, sys)
