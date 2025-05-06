import logging

import joblib
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def evaluate_model(config_path, X_test, y_test):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_path = config["model_path"]

    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    logging.info(f"Model RMSE: {rmse}")
    return rmse
