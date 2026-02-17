import os
import sys

#importing custom exception and logger
from src.exception import CustomException
from src.logger import logging

#importing save_object function
from src.utils import save_object, evaluate_models

#importing dataclass
from dataclasses import dataclass

#importing models
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn. ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Model training started")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            logging.info("Data split successfully")
            
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNeighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "RandomForest": RandomForestRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0)
            }
                        
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
            
            logging.info("Model training completed")

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            best_model_score = model_report[best_model_name]

            if best_model_score < 0.7:
                raise CustomException("Best model score is less than 0.7", sys)

            predict = best_model.predict(X_test)
            r2 = r2_score(y_test, predict)
            logging.info(f"R2 score of best model is {r2}")
            logging.info(f"Best model is {best_model_name}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved successfully")
            
        except Exception as e:
            raise CustomException(e, sys)

