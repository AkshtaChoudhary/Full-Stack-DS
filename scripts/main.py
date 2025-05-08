import argparse
import logging
import subprocess

import mlflow


def run_data_preparation(config_path, mlflow_enabled):
    if mlflow_enabled:
        # Start MLflow run for data preparation
        with mlflow.start_run(run_name="Data Preparation", nested=True) as run:
            logging.info(f"Data Preparation Run ID:{run.info.run_id}")
            mlflow.log_param("config_path", config_path)
            subprocess.run(
                ["python", "scripts/ingest.py", "--config", config_path, "--mlflow"]
            )
            mlflow.log_metric("ingestion_complete", 1)
    else:
        subprocess.run(["python", "scripts/ingest.py", "--config", config_path])


def run_model_training(config_path, mlflow_enabled):
    if mlflow_enabled:
        # Start MLflow run for model training
        with mlflow.start_run(run_name="Model Training", nested=True) as run:
            logging.info(f"Model Training Run ID:{run.info.run_id}")
            mlflow.log_param("config_path", config_path)
            subprocess.run(
                ["python", "scripts/train.py", "--config", config_path, "--mlflow"]
            )
            mlflow.log_metric("model_training_complete", 1)
    else:
        subprocess.run(["python", "scripts/train.py", "--config", config_path])


def run_model_scoring(config_path, mlflow_enabled):
    if mlflow_enabled:
        # Start MLflow run for model scoring
        with mlflow.start_run(run_name="Model Scoring", nested=True) as run:
            logging.info(f"Model Scoring Run ID:{run.info.run_id}")
            mlflow.log_param("config_path", config_path)
            subprocess.run(
                ["python", "scripts/score.py", "--config", config_path, "--mlflow"]
            )
            mlflow.log_metric("model_scoring_complete", 1)
    else:
        subprocess.run(["python", "scripts/score.py", "--config", config_path])


def main():
    parser = argparse.ArgumentParser(description="End-to-End ML Pipeline")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking for the entire pipeline",
    )
    args = parser.parse_args()

    config_path = args.config
    mlflow_enabled = args.mlflow

    # Start the parent MLflow run if enabled
    if mlflow_enabled:
        mlflow.set_experiment("Housing Experiment Pipeline")
        with mlflow.start_run(run_name="End-to-End ML Pipeline") as parent_run:
            mlflow.log_param("config_path", config_path)
            logging.info(
                f"Running ML pipeline with parent run ID: {parent_run.info.run_id}"
            )

            # Run the child tasks: data preparation, model training, and model scoring
            run_data_preparation(config_path, mlflow_enabled)
            run_model_training(config_path, mlflow_enabled)
            run_model_scoring(config_path, mlflow_enabled)

            logging.info(
                f"End-to-end ML pipeline completed under parent run ID: {parent_run.info.run_id}"
            )
    else:
        logging.info("Running ML pipeline without MLflow tracking.")
        run_data_preparation(config_path, mlflow_enabled)
        run_model_training(config_path, mlflow_enabled)
        run_model_scoring(config_path, mlflow_enabled)


if __name__ == "__main__":
    main()