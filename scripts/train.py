import argparse
import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import yaml

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
        run_id = os.environ.get("MLFLOW_RUN_ID")
        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.set_experiment("Housing Experiment")
            mlflow.start_run(run_name="Data Preparation", nested=True)
        logging.info("MLflow tracking started.")

    logging.info("Starting data preparation...")
    df = load_data(args.config)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_set, _ = stratified_split(
        df, splits=config["splits"], testsize=config["test_size"]
    )
    X_train, imputer = prepare_data(train_set.drop("median_house_value", axis=1))
    y_train = train_set["median_house_value"]

    # Log parameters to MLflow if enabled
    if args.mlflow:
        mlflow.log_param("config", args.config)
        mlflow.log_param("num_features", X_train.shape[1])
        mlflow.log_param("Stratified Split", config["splits"])
        mlflow.log_param("Test Size", config["test_size"])
        mlflow.sklearn.log_model(imputer, artifact_path="imputer")

    # End MLflow run if it was started
    if args.mlflow:
        mlflow.end_run()
        logging.info("Data Preparation MLflow run ended.")

    # Start MLflow run if enabled
    if args.mlflow:
        mlflow.set_experiment("Housing Experiment")
        mlflow.start_run(run_name="Modeling")
        logging.info("MLflow tracking started.")

    logging.info("Starting model training...")
    for model_type in [
        "linear_regression",
        "decision_tree",
        "random_forest_random_search",
        "random_forest_grid_search",
    ]:
        logging.info(f"Starting {model_type}...")
        # Calling the function
        model, rmse, mae = train_model(X_train, y_train, model_type)
        logging.info(f"{model_type} Metrics - RMSE: {rmse} & MAE: {mae}")
        # Dumping model
        model_path = config[model_type]
        # Model dump directory
        model_dir = os.path.dirname(model_path)
        # Create directory if does not exists
        os.makedirs(model_dir, exist_ok=True)
        # Dump model
        joblib.dump(model, model_path)
        logging.info(f"{model_type} Model Pickle saved at: {model_path}")
        # Log model metrics to MLflow if enabled
        if args.mlflow:
            with mlflow.start_run(run_name=model_type, nested=True):
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=model_type,
                    input_example=X_train.iloc[:5],
                    signature=mlflow.models.infer_signature(
                        X_train, model.predict(X_train)
                    ),
                )
                params = model.get_params()
                for key, value in params.items():
                    mlflow.log_param(key, value)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_param("Model Pickle Path", model_path)
    logging.info("Model training completed.")
    # End MLflow run if it was started
    if args.mlflow:
        mlflow.end_run()
        logging.info("MLflow run ended.")


if __name__ == "__main__":
    main()
