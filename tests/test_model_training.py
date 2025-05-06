from housing import data_preparation, model_training


def test_model_training():
    df = data_preparation.load_data()
    train_set, _ = data_preparation.stratified_split(df)
    X, _ = data_preparation.prepare_data(
        train_set.drop("median_house_value", axis=1)
        )
    y = train_set["median_house_value"]
    model = model_training.train_model(X, y)
    assert model is not None
