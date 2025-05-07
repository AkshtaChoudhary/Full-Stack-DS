import os


def test_package_import():
    try:
        import housing

        housing.data_ingestion.fetch_data("config/config.yaml")
        assert os.path.exists("data/housing.csv")

    except ImportError:
        assert False, "housing package not installed properly"