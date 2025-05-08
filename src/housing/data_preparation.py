import logging
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


def load_data(config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_file = os.path.join(config["raw_data_path"], config["raw_data_file"])
    return pd.read_csv(data_file)


def stratified_split(data, testsize=0.2, splits=1):
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=splits, test_size=testsize, random_state=42)

    for train_idx, test_idx in split.split(data, data["income_cat"]):
        strat_train = data.loc[train_idx].drop("income_cat", axis=1)
        strat_test = data.loc[test_idx].drop("income_cat", axis=1)

    return strat_train, strat_test


def prepare_data(data):
    data = data.copy()
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]

    num = data.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    num_prepared = pd.DataFrame(
        imputer.fit_transform(num), columns=num.columns, index=num.index
    )

    cat = pd.get_dummies(data["ocean_proximity"], drop_first=True)
    return num_prepared.join(cat), imputer
