import argparse
import logging

import mlflow
import mlflow.sklearn
import yaml

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
        "--no-console-log", action="store_true", help="Suppress console logging"
    )
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    args = parser.parse_args()

    # Configure logging
    configure_logging(
        log_level=args.log_level,
        log_path=args.log_path,
        console_log=not args.no_console_log,
    )

    # Start MLflow run if enabled
    if args.mlflow:
        mlflow.set_experiment("Housing Experiment")
        mlflow.start_run(run_name="Model Scoring", nested=True)
        logging.info("MLflow tracking started for scoring.")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logging.info("Getting & Processing the test data...")
    df = load_data(args.config)
    _, test_set = stratified_split(
        df, splits=config["splits"], testsize=config["test_size"]
    )
    y_test = test_set["median_house_value"]
    X_test, _ = prepare_data(test_set.drop("median_house_value", axis=1))
    logging.info("Processing complete.")

    # Log parameters to MLflow if enabled
    if args.mlflow:
        mlflow.log_param("num_test_samples", len(y_test))

    logging.info("Starting model scoring...")

    for model_type in [
        "linear_regression",
        "decision_tree",
        "random_forest_random_search",
        "random_forest_grid_search",
    ]:
        # Model path
        m_path = config[model_type]
        # Scoring function
        rmse, mae = evaluate_model(m_path, X_test, y_test)
        logging.info(
            f"{model_type} Model scoring completed with Test RMSE:{rmse} & MAE:{mae}"
        )
        # Log metrics to MLflow if enabled
        if args.mlflow:
            with mlflow.start_run(run_name=model_type, nested=True):
                mlflow.log_metric("Test RMSE", rmse)
                mlflow.log_metric("Test MAE", mae)

    logging.info("Model scoring complete.")

    # End MLflow run if it was started
    if args.mlflow:
        mlflow.end_run()
        logging.info("MLflow run ended for scoring.")


if __name__ == "__main__":
    main()
