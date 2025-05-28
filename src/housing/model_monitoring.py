import json
import logging
import os

from evidently import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    RegressionPreset,
)
from evidently.report import Report

logger = logging.getLogger(__name__)


def generate_evidently_reports(train, test, output_dir, target_col, model_type):
    # Create path
    os.makedirs(output_dir, exist_ok=True)

    # Column Mapping
    column_mapping = ColumnMapping(target=target_col, prediction="prediction")

    logging.info("Generating Drift, Data Quality & Performance Reports...")

    # Create reports
    reports = {
        "data_drift": Report(metrics=[DataDriftPreset()]),
        "data_quality": Report(metrics=[DataQualityPreset()]),
        "performance": Report(metrics=[RegressionPreset()]),
    }

    reports["data_drift"].run(reference_data=train, current_data=test)
    reports["data_quality"].run(reference_data=train, current_data=test)
    reports["performance"].run(
        reference_data=train, current_data=test, column_mapping=column_mapping
    )

    for name, report in reports.items():
        html_file_name = os.path.join(
            output_dir, model_type + "_" + name + "_report.html"
        )
        report.save_html(html_file_name)
        logging.info(f"{name} HTML report saved at {html_file_name}")
        json_file_name = os.path.join(
            output_dir, model_type + "_" + name + "_report.json"
        )
        report.save_json(json_file_name)
        logging.info(f"{name} JSON report saved at {json_file_name}")

    return {
        "data_drift": os.path.join(output_dir, model_type + "_data_drift_report.json"),
        "performance": os.path.join(
            output_dir, model_type + "_performance_report.json"
        ),
    }


def check_data_drift(report_path, drift_ratio_threshold=0.2):

    logging.info("Checking for Data Drift...")

    # Loading the report
    with open(report_path) as f:
        report = json.load(f)
    # Initializing
    drifted_columns = {}
    drifted_count = 0

    for metric in report.get("metrics", []):
        if metric["metric"] == "DataDriftTable":
            result = metric["result"]
            columns_info = result.get("drift_by_columns", {})
            total_columns = len(columns_info)

            for col, info in columns_info.items():
                if info.get("drift_detected"):
                    drifted_columns[col] = {
                        "score": info.get("drift_score"),
                        "p_value": info.get("p_value"),
                        "stat_test": info.get("stat_test_name"),
                    }
                    drifted_count += 1

            drift_ratio = drifted_count / total_columns if total_columns else 0

            if drift_ratio > drift_ratio_threshold:
                logging.info("Data Drift Detected!")
                print(
                    f"Drift detected in {drifted_count}/{total_columns} features"
                    f"({drift_ratio:.2%} > {drift_ratio_threshold:.2%})"
                )
                for col, info in drifted_columns.items():
                    print(
                        f" - {col}: score={info['score']:.4f},"
                        f"p={info['p_value']:.4f}, test={info['stat_test']}"
                    )
                return drift_ratio, False
            else:
                logging.info("Data Drift in acceptable range.")
                print(
                    f"Drift within acceptable range"
                    f"({drifted_count}/{total_columns} = {drift_ratio:.2%})"
                )
                return drift_ratio, True

    # If "DataDriftTable" metric not found
    logging.warning("DataDriftTable metric not found in the report.")
    return 0.0, True


def check_model_performance(report_path, threshold=0.75):

    logging.info("Checking Model Performance...")
    # Reading teh reports
    with open(report_path) as f:
        report = json.load(f)

    for metric in report.get("metrics", []):
        if metric["metric"] == "RegressionQualityMetric":
            r2 = metric["result"]["current"].get("r2_score")
            if r2 is not None:
                print(f"R² Score: {r2:.4f}")
                logging.info(f"R² Score: {r2:.4f}")
                if r2 < threshold:
                    print(f"R² below threshold {threshold}")
                    logging.info(f"R² below threshold {threshold}")
                    return r2, False
                print("Model quality is acceptable.")
                logging.info("Model quality is acceptable.")
                return r2, True
    logging.info("R² score missing in report.")
    print("R² score missing in report.")
    return False
