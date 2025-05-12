import argparse
import logging
import os

import mlflow
import mlflow.sklearn
import yaml

from housing.data_preparation import load_data, prepare_data, stratified_split
from housing.logging_utils import configure_logging
from housing.model_scoring import evaluate_model

def preprocess(input_df):
    _, test_set = stratified_split(input_df)
    y_test = test_set["median_house_value"]
    X_test, _ = prepare_data(test_set.drop("median_house_value", axis=1))
    logging.info("Processing complete.")
    return X_test,y_test

def main(args):
    # Configure logging
    configure_logging(
        log_level=args.log_level,
        log_path=args.log_path,
        console_log=not args.no_console_log,
    )
    
    # Loading Data
    logging.info("Loading data...")
    input_df = pd.read_csv(args.input)

    # Preprocessing Data
    logging.info("Preprocessing data...")
    X,y = preprocess(input_df)

    # Calling evalutaion
    logging.info("Running inference...")
    preds,rmse, mae = evaluate_model(args.model, X, y)
    logging.info(f"Model scoring completed with Test RMSE:{rmse} & MAE:{mae}")

    logging.info("Saving predictions...")
    output_df = input_df.copy()
    output_df["prediction"] = preds
    output_df.to_csv(args.output, index=False)

    logging.info("Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(decription = "Inference")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g. DEBUG, INFO)")
    parser.add_argument("--log-path", help="Optional log file path")
    parser.add_argument("--no-console-log", action="store_true", help="Suppress console logging")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    args = parser.parse_args()

    main(args)