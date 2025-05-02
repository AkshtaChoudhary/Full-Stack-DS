import argparse
import logging

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

    logging.info("Starting data ingestion...")
    fetch_data(args.config)
    logging.info("Data ingestion complete.")


if __name__ == "__main__":
    main()
