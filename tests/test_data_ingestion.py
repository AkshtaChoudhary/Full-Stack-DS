import os

from housing import data_ingestion


def test_data_download():
    data_ingestion.fetch_data("config/config.yaml")
    assert os.path.exists("data/housing.csv")