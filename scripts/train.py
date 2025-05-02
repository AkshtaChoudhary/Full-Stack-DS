import argparse
import logging

from housing.data_preparation import load_data, prepare_data, stratified_split
from housing.logging_utils import configure_logging
from housing.model_training import train_model


def main():
    parser = argparse.ArgumentParser(description="Train Housing Model")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (e.g. DEBUG, INFO)"
    )
    parser.add_argument("--log-path", help="Optional log file path")
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Suppress console logging"
    )
    args = parser.parse_args()

    configure_logging(
        log_level=args.log_level,
        log_path=args.log_path,
        console_log=not args.no_console_log,
    )

    logging.info("Starting data preparation...")
    df = load_data(args.config)
    train_set, _ = stratified_split(df)
    X_train, imputer = prepare_data(
        train_set.drop("median_house_value", axis=1)
        )
    y_train = train_set["median_house_value"]

    logging.info("Starting model training...")
    train_model(X_train, y_train, args.config)
    logging.info("Model training complete.")


if __name__ == "__main__":
    main()
