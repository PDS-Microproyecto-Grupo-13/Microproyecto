import argparse
import os
import sys
import time
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def log_event(level: str, event: str, **kwargs: Any) -> None:
    """Outputs a structured operational log line to stdout/stderr."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    context_items = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
    log_line = f"{timestamp} [{level.upper()}] event={event} {context_items}".strip()
    if level.upper() in ("ERROR", "CRITICAL"):
        print(log_line, file=sys.stderr, flush=True)
    else:
        print(log_line, file=sys.stdout, flush=True)


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for model promotion."""
    parser = argparse.ArgumentParser(
        description="Promotes a registered MLflow model version to an alias (e.g. champion)."
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Name of the registered model in MLflow Model Registry.",
    )
    parser.add_argument(
        "--version",
        "-v",
        required=True,
        help="Version number of the model to promote.",
    )
    parser.add_argument(
        "--alias",
        "-a",
        default="champion",
        help="Target alias to assign (default: 'champion').",
    )
    parser.add_argument(
        "--tracking-uri",
        "-u",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow Tracking Server URI (default: http://localhost:5000 or $MLFLOW_TRACKING_URI).",
    )
    return parser.parse_args()


def promote_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    alias: str,
) -> tuple[str | None, str]:
    """Promotes a registered model version to the target alias and returns (prev_version, new_version)."""
    # 1. Verify model and specific version exist
    try:
        model_version = client.get_model_version(name=model_name, version=str(version))
    except MlflowException as exc:
        log_event(
            "ERROR",
            "model_version_not_found",
            model=model_name,
            version=version,
            error=str(exc),
        )
        raise ValueError(
            f"Model version '{version}' for registered model '{model_name}' does not exist: {exc}"
        ) from exc

    # 2. Query previous version holding this alias if any
    prev_version: str | None = None
    try:
        prev_model = client.get_model_version_by_alias(name=model_name, alias=alias)
        prev_version = str(prev_model.version)
        log_event(
            "INFO",
            "previous_alias",
            model=model_name,
            alias=alias,
            version=prev_version,
        )
    except MlflowException:
        log_event(
            "INFO",
            "previous_alias",
            model=model_name,
            alias=alias,
            version="none",
        )

    # 3. Set the new alias
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=str(model_version.version),
        )
    except MlflowException as exc:
        log_event(
            "ERROR",
            "alias_assignment_failed",
            model=model_name,
            alias=alias,
            version=version,
            error=str(exc),
        )
        raise RuntimeError(
            f"Failed to assign alias '{alias}' to version '{version}' of '{model_name}': {exc}"
        ) from exc

    log_event(
        "INFO",
        "model_promoted",
        model=model_name,
        alias=alias,
        from_version=prev_version if prev_version else "none",
        to_version=str(model_version.version),
    )

    return prev_version, str(model_version.version)


def main() -> None:
    args = parse_args()

    log_event(
        "INFO",
        "model_promotion_started",
        model=args.model,
        version=args.version,
        alias=args.alias,
        tracking_uri=args.tracking_uri,
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    try:
        promote_model_version(
            client=client,
            model_name=args.model,
            version=args.version,
            alias=args.alias,
        )
    except Exception as exc:
        log_event("ERROR", "model_promotion_failed", error=str(exc))
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
