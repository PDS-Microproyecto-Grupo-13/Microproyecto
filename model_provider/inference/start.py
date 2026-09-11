import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


@dataclass
class ServingConfig:
    """Operational configuration for MLflow Model Serving."""

    tracking_uri: str
    model_name: str
    model_alias: str
    host: str
    port: int
    status_port: int = 5002
    env_manager: str = "local"


@dataclass
class ResolvedModelInfo:
    """Metadata of the resolved model version."""

    model_name: str
    model_alias: str
    version: str
    run_id: str | None
    source: str | None
    creation_timestamp: int | None


def log_event(level: str, event: str, **kwargs: Any) -> None:
    """Outputs a structured operational log line to stdout."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    context_items = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
    log_line = (
        f"{timestamp} [{level.upper()}] service=inference event={event} {context_items}".strip()
    )
    if level.upper() in ("ERROR", "CRITICAL"):
        print(log_line, file=sys.stderr, flush=True)
    else:
        print(log_line, file=sys.stdout, flush=True)


def load_config_from_env() -> ServingConfig:
    """Loads and validates configuration from environment variables."""
    model_name = os.getenv("MODEL_NAME")
    if not model_name or not model_name.strip():
        log_event("ERROR", "startup_failed", error="MODEL_NAME environment variable is required")
        sys.exit(1)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-tracking:5000").strip()
    model_alias = os.getenv("MODEL_ALIAS", "champion").strip()
    host = os.getenv("INFERENCE_HOST", os.getenv("HOST", "0.0.0.0")).strip()
    port_str = os.getenv("INFERENCE_PORT", os.getenv("PORT", "5001")).strip()

    try:
        port = int(port_str)
    except ValueError:
        log_event("ERROR", "startup_failed", error=f"Invalid port value: '{port_str}'")
        sys.exit(1)

    status_port_str = os.getenv("INFERENCE_STATUS_PORT", "5002").strip()
    try:
        status_port = int(status_port_str)
    except ValueError:
        log_event("WARN", "invalid_status_port", value=status_port_str)
        status_port = 5002

    return ServingConfig(
        tracking_uri=tracking_uri,
        model_name=model_name.strip(),
        model_alias=model_alias,
        host=host,
        port=port,
        status_port=status_port,
    )


def wait_for_tracking_server(
    client: MlflowClient, max_retries: int = 5, delay: float = 2.0
) -> None:
    """Verifies connectivity with MLflow Tracking server before proceeding."""
    for attempt in range(1, max_retries + 1):
        try:
            # Simple API probe to verify tracking server responsiveness
            client.search_experiments(max_results=1)
            log_event("INFO", "tracking_connected", tracking_uri=client.tracking_uri)
            return
        except Exception as exc:
            if attempt == max_retries:
                log_event(
                    "ERROR",
                    "tracking_connection_failed",
                    tracking_uri=client.tracking_uri,
                    error=str(exc),
                )
                raise ConnectionError(
                    f"Unable to connect to MLflow Tracking Server at {client.tracking_uri}: {exc}"
                ) from exc
            log_event(
                "WARN",
                "tracking_retry",
                attempt=attempt,
                max_retries=max_retries,
                retry_in_seconds=delay,
            )
            time.sleep(delay)


def resolve_model_info(client: MlflowClient, config: ServingConfig) -> ResolvedModelInfo:
    """Resolves model alias into a concrete model version and extracts operational metadata."""
    # 1. Check registered model existence
    try:
        client.get_registered_model(name=config.model_name)
    except MlflowException as exc:
        log_event(
            "ERROR",
            "model_lookup_failed",
            model=config.model_name,
            error=f"Registered model '{config.model_name}' does not exist",
        )
        raise ValueError(f"Registered model '{config.model_name}' does not exist: {exc}") from exc

    # 2. Resolve alias to a specific version
    try:
        model_version = client.get_model_version_by_alias(
            name=config.model_name,
            alias=config.model_alias,
        )
    except MlflowException as exc:
        log_event(
            "ERROR",
            "alias_lookup_failed",
            model=config.model_name,
            alias=config.model_alias,
            error=f"Alias '{config.model_alias}' is not configured for model '{config.model_name}'",
        )
        raise ValueError(
            f"Alias '{config.model_alias}' is not configured for model '{config.model_name}': {exc}"
        ) from exc

    resolved_info = ResolvedModelInfo(
        model_name=config.model_name,
        model_alias=config.model_alias,
        version=str(model_version.version),
        run_id=getattr(model_version, "run_id", None),
        source=getattr(model_version, "source", None),
        creation_timestamp=getattr(model_version, "creation_timestamp", None),
    )

    exact_model_uri = f"models:/{resolved_info.model_name}/{resolved_info.version}"
    log_event(
        "INFO",
        "model_resolved",
        model_name=resolved_info.model_name,
        requested_alias=resolved_info.model_alias,
        resolved_version=resolved_info.version,
        exact_model_uri=exact_model_uri,
        run_id=resolved_info.run_id,
    )

    return resolved_info


def build_serve_command(config: ServingConfig, model_info: ResolvedModelInfo) -> list[str]:
    """Constructs the CLI command for mlflow models serve pinning to the concrete version."""
    # Pin directly to models:/<name>/<version> to ensure immutable serving during runtime
    model_uri = f"models:/{model_info.model_name}/{model_info.version}"
    return [
        sys.executable,
        "-m",
        "mlflow",
        "models",
        "serve",
        "--model-uri",
        model_uri,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--env-manager",
        config.env_manager,
    ]


class StatusHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler to report serving runtime status and metadata."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress standard HTTP access log spam
        pass

    def do_GET(self) -> None:
        server: InferenceStatusServer = self.server  # type: ignore[assignment]
        if self.path in ("/status", "/status/"):
            data = server.status_callback()
            is_running = data.get("model_server_running", False)
            status_code = 200 if is_running else 503
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/health", "/healthz", "/ping"):
            data = server.status_callback()
            is_running = data.get("model_server_running", False)
            status_code = 200 if is_running else 503
            body = json.dumps({"status": "healthy" if is_running else "unhealthy"}).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


class InferenceStatusServer(ThreadingHTTPServer):
    """Threading HTTP server that carries a status provider callback."""

    def __init__(
        self,
        server_address: tuple[str, int],
        status_callback: Callable[[], dict[str, Any]],
    ) -> None:
        self.status_callback = status_callback
        super().__init__(server_address, StatusHandler)


def run_serving_process(
    cmd: list[str],
    model_info: ResolvedModelInfo,
    tracking_uri: str,
    status_host: str = "0.0.0.0",
    status_port: int = 5002,
) -> int:
    """Launches the MLflow serving subprocess and manages OS termination signals and status server."""
    exact_model_uri = f"models:/{model_info.model_name}/{model_info.version}"
    log_event(
        "INFO",
        "model_server_starting",
        model_name=model_info.model_name,
        requested_alias=model_info.model_alias,
        resolved_version=model_info.version,
        exact_model_uri=exact_model_uri,
        command=" ".join(cmd),
    )

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri

    process = subprocess.Popen(cmd, env=env)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def get_status() -> dict[str, Any]:
        is_running = process.poll() is None
        return {
            "status": "ok" if is_running else "error",
            "model_name": model_info.model_name,
            "requested_alias": model_info.model_alias,
            "resolved_version": model_info.version,
            "loaded_version": model_info.version,
            "exact_model_uri": exact_model_uri,
            "run_id": model_info.run_id,
            "source": model_info.source,
            "model_server_pid": process.pid,
            "model_server_running": is_running,
            "started_at": started_at,
        }

    status_server: InferenceStatusServer | None = None
    if status_port > 0:
        try:
            status_server = InferenceStatusServer((status_host, status_port), get_status)
            status_thread = threading.Thread(target=status_server.serve_forever, daemon=True)
            status_thread.start()
            log_event(
                "INFO",
                "status_server_started",
                host=status_host,
                port=status_port,
                loaded_version=model_info.version,
            )
        except Exception as exc:
            log_event(
                "WARN",
                "status_server_failed_to_start",
                host=status_host,
                port=status_port,
                error=str(exc),
            )

    def signal_handler(signum: int, _frame: Any) -> None:
        signame = signal.Signals(signum).name
        log_event(
            "INFO",
            "shutdown_signal_received",
            signal=signame,
            model=model_info.model_name,
            version=model_info.version,
        )
        if process.poll() is None:
            process.terminate()
        if status_server is not None:
            try:
                status_server.shutdown()
            except Exception:
                pass

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    return_code = process.wait()

    if status_server is not None:
        try:
            status_server.shutdown()
            status_server.server_close()
        except Exception:
            pass

    log_event(
        "INFO",
        "model_server_stopped",
        model=model_info.model_name,
        version=model_info.version,
        exit_code=return_code,
    )
    return return_code


def main() -> None:
    """Entrypoint for inference server starter wrapper."""
    config = load_config_from_env()
    mlflow.set_tracking_uri(config.tracking_uri)
    client = MlflowClient(tracking_uri=config.tracking_uri)

    try:
        wait_for_tracking_server(client)
        model_info = resolve_model_info(client, config)
    except Exception as exc:
        log_event("ERROR", "startup_failed", error=str(exc))
        sys.exit(1)

    cmd = build_serve_command(config, model_info)
    exit_code = run_serving_process(
        cmd,
        model_info,
        config.tracking_uri,
        status_host=config.host,
        status_port=config.status_port,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
