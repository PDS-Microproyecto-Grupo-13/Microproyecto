from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ml_pipeline.common.io import sha256_file
from ml_pipeline.settings import Settings


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def git_metadata(root: Path) -> tuple[str | None, bool | None]:
    commit = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--porcelain"], root)
    return commit, None if status is None else bool(status)


def collect_lineage(settings: Settings, include_lock: bool = False) -> dict[str, Any]:
    commit, dirty = git_metadata(settings.root)
    lineage = {
        "git_commit": commit,
        "git_dirty": dirty,
        # Hashing dvc.lock inside an output described by that same lock would
        # create a self-reference. dvc.yaml is the stable pipeline revision.
        "dvc_revision": sha256_file(settings.path("dvc.yaml")),
        "params_hash": sha256_file(settings.path("params.yaml")),
        "dataset_fingerprint": sha256_file(settings.path("data/raw/dataset.csv")),
    }
    if include_lock:
        lineage["dvc_lock_hash"] = sha256_file(settings.path("dvc.lock"))
    return lineage


def as_mlflow_tags(lineage: dict[str, Any]) -> dict[str, str]:
    return {f"lineage.{key}": "unknown" if value is None else str(value).lower() if isinstance(value, bool) else str(value) for key, value in lineage.items()}
