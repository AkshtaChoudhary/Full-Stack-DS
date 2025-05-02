import logging
import os

import joblib
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


def train_model(X, y, config_path="config/config.yaml"):
    param_grid = [
        {
            "n_estimators": [3, 10, 30],
            "max_features": [2, 4, 6, 8]
        },
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4]
            },
    ]
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X, y)

    logging.info("Training completed.")
    best_model = grid_search.best_estimator_

    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_path = config["model_path"]
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, config["model_path"])
    logging.info(f"Model saved to {config['model_path']}")

    return best_model
