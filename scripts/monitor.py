import argparse
import logging
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import yaml

from housing.data_preparation import load_data, prepare_data, stratified_split
from housing.logging_utils import configure_logging
from housing.model_monitoring import (
    check_data_drift,
    check_model_performance,
    generate_evidently_reports,
)


def main():
    parser = argparse.ArgumentParser(description="Model Monitoring")
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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--drift_threshold", type=float, default=0.2)
    args = parser.parse_args()

    # Configure logging
    configure_logging(
        log_level=args.log_level,
        log_path=args.log_path,
        console_log=not args.no_console_log,
    )

    # Start MLflow run if enabled
    if args.mlflow:
        run_id = os.environ.get("MLFLOW_RUN_ID")
        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.set_experiment("Housing Experiment")
            mlflow.start_run(run_name="Final Model Monitoring", nested=True)
        logging.info("MLflow tracking started for scoring.")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Get the data
    fetch_data(args.config)
    # Load the data
    df = load_data(args.config)
    # Train test split
    train_set, test_set = stratified_split(
        df, splits=config["splits"], testsize=config["test_size"]
    )
    # Get data to predict
    X_train, _ = prepare_data(train_set.drop(columns=[config["target"]], axis=1))
    X_test, _ = prepare_data(test_set.drop(columns=[config["target"]], axis=1))

    logging.info("Starting model monitoring...")

    # Getting the final model name from config
    model_type = config["final_model"]
    # Loading the model
    model = joblib.load(config[model_type])

    # Make predictions
    train_set["prediction"] = model.predict(X_train)
    test_set["prediction"] = model.predict(X_test)

    # Monitoring function
    report_paths = generate_evidently_reports(
        train=train_set,
        test=test_set,
        output_dir=config["model_monitoring_path"],
        target_col=config["target"],
        model_type=model_type,
    )

    # Check Drift
    drift_ratio, drift_ok = check_data_drift(
        report_paths["data_drift"], drift_ratio_threshold=args.drift_threshold
    )

    # Check Performance
    r2, perf_ok = check_model_performance(
        report_paths["performance"], threshold=args.threshold
    )

    if not (drift_ok and perf_ok):
        logging.info(f"One or more checks failed for {model_type}.")
        if args.mlflow:
            with mlflow.start_run(run_name=model_type, nested=True):
                mlflow.log_metric("Drift Ratio", drift_ratio)
                mlflow.log_metric("Drift Threshold", args.drift_threshold)
                mlflow.log_metric("Drift Pass", drift_ok)
                mlflow.log_metric("R2", r2)
                mlflow.log_metric("Performance Threshold", args.threshold)
                mlflow.log_metric("Performance Pass", perf_ok)
                mlflow.log_metric("All Checked Passed", False)
        sys.exit(1)
    else:
        logging.info(f"All checks passed {model_type}")
        if args.mlflow:
            with mlflow.start_run(run_name=model_type, nested=True):
                mlflow.log_metric("Drift Ratio", drift_ratio)
                mlflow.log_metric("Drift Threshold", args.drift_threshold)
                mlflow.log_metric("Drift Pass", drift_ok)
                mlflow.log_metric("R2", r2)
                mlflow.log_metric("Performance Threshold", args.threshold)
                mlflow.log_metric("Performance Pass", perf_ok)
                mlflow.log_metric("All Checked Passed", True)
        sys.exit(0)


if __name__ == "__main__":
    main()
