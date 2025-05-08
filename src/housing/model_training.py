import logging

import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


def train_model(X, y, model_type):
    if model_type == "linear_regression":
        model = LinearRegression()
        model.fit(X, y)
    if model_type == "decision_tree":
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X, y)
    if model_type == "random_forest_random_search":
        param_distribs = {
            "n_estimators": randint(1, 200),
            "max_features": randint(1, 8),
        }
        rnd_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42),
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(X, y)
        # Get best model
        model = rnd_search.best_estimator_
    if model_type == "random_forest_grid_search":
        param_grid = [
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
        ]
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        grid_search.fit(X, y)
        # Get best model
        model = grid_search.best_estimator_

    # Get predictions
    predictions = model.predict(X)
    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y, predictions))
    # Compute MAE
    mae = mean_absolute_error(y, predictions)

    return model, rmse, mae
