# Dockerfile for my-ml-project
# Author: Akshta
# Email: akshta.choudhary@tigeranalytics.com

# Download base image of Python 3.12 based on Debian
FROM python:3.12-slim

# Create a non-root user and switch to it
RUN useradd --create-home appuser
USER appuser

# Copy your application code into the container
COPY --chown=appuser:appuser ./dist/fsds-0.1.0-py3-none-any.whl /home/appuser/app/fsds-0.1.0-py3-none-any.whl
COPY --chown=appuser:appuser ./scripts/infer.py /home/appuser/app/infer.py
COPY --chown=appuser:appuser ./artifacts/model/rf_rs_model.pkl /home/appuser/app/rf_rs_model.pkl

# Set working directory
WORKDIR /home/appuser/app
# Install the Python package
RUN pip install /home/appuser/app/fsds-0.1.0-py3-none-any.whl

# Set default entrypoint
ENTRYPOINT ["python", "infer.py"]