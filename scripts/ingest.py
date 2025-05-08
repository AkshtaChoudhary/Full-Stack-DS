import argparse
import logging
import os

import mlflow

from housing.data_ingestion import fetch_data
from housing.logging_utils import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Ingest Housing Data")
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

    configure_logging(
        log_level=args.log_level,
        log_path=args.log_path,
        console_log=not args.no_console_log,
    )

    # Start MLflow run for data preparation if --mlflow is passed
    if args.mlflow:
        run_id = os.environ.get("MLFLOW_RUN_ID")
        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.set_experiment("Housing Experiment")
            mlflow.start_run(run_name="Data Preparation", nested=True)
        logging.info("MLflow tracking started.")

    logging.info("Starting data ingestion...")

    # Log parameters and metrics to MLflow if enabled
    if args.mlflow:
        mlflow.log_param("config_path", args.config)

    # Perform data ingestion
    fetch_data(args.config)

    logging.info("Data ingestion complete.")

    # Log success metric to MLflow if enabled
    if args.mlflow:
        mlflow.log_metric("ingestion_complete", 1)
        logging.info("Data ingestion completed and logged with MLflow.")

    # End MLflow run if it was started
    if args.mlflow:
        mlflow.end_run()
        logging.info("Data Ingestion MLflow run ended.")


if __name__ == "__main__":
    main()
