import argparse
import logging

from housing.data_preparation import load_data, prepare_data, stratified_split
from housing.logging_utils import configure_logging
from housing.model_scoring import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Score Housing Model")
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

    logging.info("Getting & Processing the test data...")
    df = load_data(args.config)
    _, test_set = stratified_split(df)
    y_test = test_set["median_house_value"]
    X_test, _ = prepare_data(test_set.drop("median_house_value", axis=1))
    logging.info("Processing complete.")

    logging.info("Starting model scoring...")
    evaluate_model(args.config, X_test, y_test)
    logging.info("Model scoring complete.")


if __name__ == "__main__":
    main()
