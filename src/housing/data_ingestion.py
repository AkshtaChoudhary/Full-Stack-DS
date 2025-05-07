import logging
import os
import tarfile
import urllib.request

import yaml

logger = logging.getLogger(__name__)


def fetch_data(config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    url = config["download_url"]
    data_dir = config["raw_data_path"]
    os.makedirs(data_dir, exist_ok=True)
    tgz_path = os.path.join(data_dir, "housing.tgz")

    logging.info(f"Downloading dataset from {url}")
    urllib.request.urlretrieve(url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=data_dir)
    logging.info(f"Dataset downloaded and extracted to {data_dir}")