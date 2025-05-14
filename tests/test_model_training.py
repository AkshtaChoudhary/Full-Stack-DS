from housing import data_preparation, model_training


def test_model_training():
    df = data_preparation.load_data()
    train_set, _ = data_preparation.stratified_split(df)
    X, _ = data_preparation.prepare_data(train_set.drop("median_house_value", axis=1))
    y = train_set["median_house_value"]
    for model_type in [
        "linear_regression",
        #"decision_tree",
        #"random_forest_random_search",
        #"random_forest_grid_search",
    ]:
        model, rmse, mae = model_training.train_model(X, y, model_type)
        assert model is not None
        assert rmse is not None
        assert mae is not None
