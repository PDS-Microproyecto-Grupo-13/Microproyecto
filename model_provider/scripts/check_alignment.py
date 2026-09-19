import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for alignment verification."""
    parser = argparse.ArgumentParser(
        description="Verifies operational alignment between MLflow Registry and the live inference serving container."
    )
    parser.add_argument(
        "--model",
        "-m",
        default=os.getenv("MODEL_NAME", "salary_predict_model"),
        help="Name of the registered model (default: salary_predict_model).",
    )
    parser.add_argument(
        "--alias",
        "-a",
        default=os.getenv("MODEL_ALIAS", "champion"),
        help="Alias to verify against (default: champion).",
    )
    parser.add_argument(
        "--tracking-uri",
        "-u",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow Tracking Server URI (default: http://localhost:5000).",
    )
    parser.add_argument(
        "--status-url",
        "-s",
        default=os.getenv("INFERENCE_STATUS_URL", "http://localhost:5002/status"),
        help="Live serving status URL (default: http://localhost:5002/status).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON.",
    )
    return parser.parse_args()


def fetch_registry_version(client: MlflowClient, model_name: str, alias: str) -> str:
    """Queries MLflow Model Registry to resolve which version holds the target alias."""
    try:
        model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
        return str(model_version.version)
    except MlflowException as exc:
        raise RuntimeError(
            f"Failed to resolve alias '{alias}' for model '{model_name}' in Registry: {exc}"
        ) from exc


def fetch_runtime_status(status_url: str, timeout: float = 3.0) -> dict[str, Any]:
    """Queries the live serving status HTTP server."""
    req = urllib.request.Request(status_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Inference status server returned HTTP {resp.status} at {status_url}"
                )
            content = resp.read().decode("utf-8")
            return json.loads(content)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to connect to inference status server at {status_url}: {exc}"
        ) from exc


def check_alignment(
    model_name: str,
    alias: str,
    tracking_uri: str,
    status_url: str,
) -> dict[str, Any]:
    """Compares Registry champion version with live serving loaded_version."""
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Fetch registry version
    registry_version = fetch_registry_version(client, model_name, alias)

    # 2. Fetch runtime loaded version
    runtime_status = fetch_runtime_status(status_url)
    loaded_version = str(runtime_status.get("loaded_version") or "")
    server_running = bool(runtime_status.get("model_server_running", False))

    # 3. Determine status
    if not server_running:
        alignment_status = "runtime_unhealthy"
    elif registry_version == loaded_version:
        alignment_status = "synchronized"
    else:
        alignment_status = "redeploy_required"

    return {
        "model_name": model_name,
        "alias": alias,
        "registry_version": registry_version,
        "runtime_loaded_version": loaded_version,
        "runtime_server_running": server_running,
        "alignment_status": alignment_status,
        "runtime_metadata": runtime_status,
    }


def main() -> None:
    args = parse_args()
    try:
        result = check_alignment(
            model_name=args.model,
            alias=args.alias,
            tracking_uri=args.tracking_uri,
            status_url=args.status_url,
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        else:
            print(f"[ERROR] Alignment check failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 65)
        print(" MLflow Serving Operational Alignment Check")
        print("=" * 65)
        print(f" Model Name:             {result['model_name']}")
        print(f" Target Alias:           {result['alias']}")
        print(f" Registry Version:       {result['registry_version']}")
        print(f" Runtime Loaded Version: {result['runtime_loaded_version']}")
        print(f" Runtime Server Alive:   {result['runtime_server_running']}")
        print("-" * 65)

        status = result["alignment_status"]
        if status == "synchronized":
            print(" Alignment Status:       SYNCHRONIZED (serving latest champion)")
            print("=" * 65)
            sys.exit(0)
        elif status == "redeploy_required":
            print(" Alignment Status:       REDEPLOY REQUIRED (mismatch detected!)")
            print(
                f" Notice: Serving container runs version {result['runtime_loaded_version']}, "
                f"but Registry '{result['alias']}' is version {result['registry_version']}."
            )
            print(" Action: Redeploy or restart the inference serving container.")
            print("=" * 65)
            sys.exit(2)
        else:
            print(f" Alignment Status:       {status.upper()}")
            print("=" * 65)
            sys.exit(1)


if __name__ == "__main__":
    main()
