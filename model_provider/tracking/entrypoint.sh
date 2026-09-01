#!/usr/bin/env bash
set -eo pipefail

# Default configuration from environment variables
MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:////var/lib/mlflow/db/mlflow.db}"
MLFLOW_ARTIFACTS_DESTINATION="${MLFLOW_ARTIFACTS_DESTINATION:-/var/lib/mlflow/artifacts}"

# Ensure directories for SQLite database and artifacts exist
mkdir -p /var/lib/mlflow/db
mkdir -p /var/lib/mlflow/artifacts

echo "================================================================="
echo " Starting MLflow Tracking Server & Model Registry"
echo " Host:               $MLFLOW_HOST"
echo " Port:               $MLFLOW_PORT"
echo " Backend Store URI:  $MLFLOW_BACKEND_STORE_URI"
echo " Artifacts Destination: $MLFLOW_ARTIFACTS_DESTINATION"
echo "================================================================="

exec mlflow server \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT" \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --serve-artifacts \
    --artifacts-destination "$MLFLOW_ARTIFACTS_DESTINATION"
