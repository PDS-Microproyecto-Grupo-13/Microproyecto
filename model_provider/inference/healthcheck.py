import os
import sys
import urllib.error
import urllib.request


def check_health(host: str = "127.0.0.1", port: int = 5001, timeout: float = 3.0) -> bool:
    """Probes the MLflow Model Serving health endpoint."""
    endpoints = [
        f"http://{host}:{port}/health",
        f"http://{host}:{port}/ping",
    ]

    for url in endpoints:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            continue

    return False


def main() -> None:
    host = os.getenv("INFERENCE_HOST", os.getenv("HOST", "127.0.0.1")).strip()
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"

    port_str = os.getenv("INFERENCE_PORT", os.getenv("PORT", "5001")).strip()
    try:
        port = int(port_str)
    except ValueError:
        port = 5001

    if check_health(host=host, port=port):
        sys.exit(0)
    else:
        print(
            f"Healthcheck failed: Inference server at {host}:{port} is not responding",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
