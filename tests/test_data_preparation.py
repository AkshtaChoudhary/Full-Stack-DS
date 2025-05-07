from housing import data_preparation


def test_stratified_split():
    df = data_preparation.load_data()
    train, test = data_preparation.stratified_split(df)
    assert not train.empty and not test.empty