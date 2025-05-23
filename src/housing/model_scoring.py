import logging

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def evaluate_model(model_path, X_test, y_test):

    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    return predictions, rmse, mae
