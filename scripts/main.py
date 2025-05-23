import argparse
import logging
import os
import subprocess

import mlflow


def run_data_preparation(config_path, mlflow_enabled):
    if mlflow_enabled:
        # Start MLflow run for data preparation
        with mlflow.start_run(run_name="Data Preparation", nested=True) as run:
            logging.info(f"Data Preparation Run ID:{run.info.run_id}")
            mlflow.log_param("config_path", config_path)

            env = os.environ.copy()
            env["MLFLOW_RUN_ID"] = run.info.run_id
            env["MLFLOW_TRACKING_URI"] = mlflow.get_tracking_uri()

            subprocess.run(
                ["python", "scripts/ingest.py", "--config", config_path, "--mlflow"],
                env=env,
                check=True,
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

            env = os.environ.copy()
            env["MLFLOW_RUN_ID"] = run.info.run_id
            env["MLFLOW_TRACKING_URI"] = mlflow.get_tracking_uri()

            subprocess.run(
                ["python", "scripts/train.py", "--config", config_path, "--mlflow"],
                env=env,
                check=True,
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

            env = os.environ.copy()
            env["MLFLOW_RUN_ID"] = run.info.run_id
            env["MLFLOW_TRACKING_URI"] = mlflow.get_tracking_uri()

            subprocess.run(
                ["python", "scripts/score.py", "--config", config_path, "--mlflow"],
                env=env,
                check=True,
            )
            mlflow.log_metric("model_scoring_complete", 1)
    else:
        subprocess.run(["python", "scripts/score.py", "--config", config_path])


def run_model_monitoring(config_path, mlflow_enabled, thresh, drift_thresh):
    logging.info("Starting model monitoring...")

    env = os.environ.copy()
    if mlflow_enabled:
        with mlflow.start_run(run_name="Model Monitoring", nested=True) as run:
            logging.info(f"Model Monitoring Run ID: {run.info.run_id}")
            mlflow.log_param("config_path", config_path)
            env["MLFLOW_RUN_ID"] = run.info.run_id
            env["MLFLOW_TRACKING_URI"] = mlflow.get_tracking_uri()

            result = subprocess.run(
                [
                    "python",
                    "scripts/monitor.py",
                    "--config",
                    config_path,
                    "--mlflow",
                    "--threshold",
                    str(thresh),
                    "--drift_threshold",
                    str(drift_thresh),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

            logging.info(f"Model monitoring stdout:\n{result.stdout}")
            logging.warning(f"Model monitoring stderr:\n{result.stderr}")

            if result.returncode == 0:
                mlflow.log_metric("model_monitoring_complete", 1)
                logging.info("Model monitoring completed successfully!")
            else:
                mlflow.log_metric("model_monitoring_complete", 0)
                logging.warning("Model monitoring failed!")
    else:
        result = subprocess.run(
            [
                "python",
                "scripts/monitor.py",
                "--config",
                config_path,
                "--threshold",
                str(thresh),
                "--drift_threshold",
                str(drift_thresh),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        logging.info(f"Model monitoring stdout:\n{result.stdout}")
        logging.warning(f"Model monitoring stderr:\n{result.stderr}")

        # No MLflow when disabled
        if result.returncode == 0:
            logging.info("Model monitoring completed successfully!")
        else:
            logging.warning("Model monitoring failed!")


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
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--drift_threshold", type=float, default=0.2)
    args = parser.parse_args()

    config_path = args.config
    mlflow_enabled = args.mlflow
    thresh = args.threshold
    drift_thresh = args.drift_threshold

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
            run_model_monitoring(
                config_path, mlflow_enabled, thresh=thresh, drift_thresh=drift_thresh
            )

            logging.info(
                f"End-to-end ML pipeline completed-parent run ID: {parent_run.info.run_id}"
            )
    else:
        logging.info("Running ML pipeline without MLflow tracking.")
        run_data_preparation(config_path, mlflow_enabled)
        run_model_training(config_path, mlflow_enabled)
        run_model_scoring(config_path, mlflow_enabled)
        run_model_monitoring(
            config_path, mlflow_enabled, thresh=thresh, drift_thresh=drift_thresh
        )


if __name__ == "__main__":
    main()
