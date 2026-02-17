import os
import sys
from src.logger import logging
import numpy as np 
from sklearn.metrics import r2_score
import pandas as pd
import dill

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        logging.info("Model evaluation started")
        model_report = {}
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_report[model_name] = r2
        
        logging.info("Model evaluation completed")
        return model_report
        
    except Exception as e:
        raise CustomException(e, sys)