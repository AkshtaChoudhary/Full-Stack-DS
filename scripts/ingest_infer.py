import argparse
import io
import logging
import tarfile
import urllib.request

import pandas as pd

from housing.data_preparation import prepare_data
from housing.logging_utils import configure_logging
from housing.model_scoring import evaluate_model


def preprocess(input_df):
    # Creating y variables
    y_test = input_df["median_house_value"]
    # Preparing data
    X_test, _ = prepare_data(input_df.drop("median_house_value", axis=1))
    logging.info("Processing complete.")
    return X_test, y_test


def main(args):
    # Configure logging
    configure_logging(
        log_level=args.log_level,
        log_path=args.log_path,
        console_log=not args.no_console_log,
    )

    # Load input data from JSON
    logging.info("Loading data from JSON input...")
    try:
        # Step 1: Download the .tgz file into memory
        response = urllib.request.urlopen(args.input)
        tgz_bytes = io.BytesIO(response.read())

        # Step 2: Extract the CSV file from the .tgz in memory
        with tarfile.open(fileobj=tgz_bytes) as tar:
            housing_file = tar.extractfile("housing.csv")
            input_df = pd.read_csv(housing_file)

    except Exception as e:
        logging.error(f"Failed to parse input JSON: {e}")
        raise

    # Preprocessing Data
    logging.info("Preprocessing data...")
    X, y = preprocess(input_df)

    # Calling evalutaion
    logging.info("Running inference...")
    preds, rmse, mae = evaluate_model(args.model, X, y)
    logging.info(f"Model scoring completed with Test RMSE:{rmse} & MAE:{mae}")

    logging.info("Saving predictions...")
    output_df = X.copy()
    output_df["prediction"] = preds
    output_df["rmse"] = rmse
    output_df["mae"] = mae
    output_df.to_csv(args.output, index=False)

    logging.info("Inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--input", required=True, help="Input Data URL")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (e.g. DEBUG, INFO)"
    )
    parser.add_argument("--log-path", help="Optional log file path")
    parser.add_argument(
        "--no-console-log", action="store_true", help="Suppress console logging"
    )
    args = parser.parse_args()

    main(args)
