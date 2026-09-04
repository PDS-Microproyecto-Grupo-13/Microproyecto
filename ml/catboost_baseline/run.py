from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor

from .data import (
    discover_csv_files,
    load_raw_data,
    prepare_model_data,
    select_experiment,
    source_file_manifest,
    temporal_split,
)
from .evaluation import (
    reconstruct_targets,
    regression_metrics,
    segment_metrics,
    training_targets,
)
from .features import build_features
from .mlflow_model import SalaryRangePyFunc, serving_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CatBoost salary-range baseline")
    parser.add_argument(
        "--config",
        default="ml/catboost_baseline/params.yaml",
        help="YAML configuration path, relative to the repository root",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Smoke-test only: read at most this many rows from each CSV",
    )
    parser.add_argument("--iterations", type=int, default=None, help="Override model iterations")
    parser.add_argument(
        "--experiment", choices=["reported", "expanded"], default=None
    )
    parser.add_argument(
        "--target-strategy", choices=["direct", "minimum_width_log"], default=None
    )
    parser.add_argument(
        "--exclude-company",
        action="store_true",
        help="Exclude company identity to test generalization to unseen employers",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the packaged pyfunc model in MLflow Model Registry",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the registered model name",
    )
    parser.add_argument(
        "--model-alias",
        default=None,
        help="Assign this alias after registration (for example, champion)",
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping")
    return config


def train_model(
    model_params: dict[str, Any],
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    validation_features: pd.DataFrame,
    validation_target: np.ndarray,
    categorical_features: list[str],
) -> CatBoostRegressor:
    model = CatBoostRegressor(**model_params)
    model.fit(
        train_features,
        train_target,
        cat_features=categorical_features,
        eval_set=(validation_features, validation_target),
        use_best_model=True,
    )
    return model


def predict_range(
    first_model: CatBoostRegressor,
    second_model: CatBoostRegressor,
    features: pd.DataFrame,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return reconstruct_targets(first_model.predict(features), second_model.predict(features), strategy)


def main() -> int:
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    root = repository_root()
    config_path = (root / args.config).resolve()
    config = load_config(config_path)
    if args.iterations is not None:
        config["model"]["iterations"] = args.iterations
    if args.experiment is not None:
        config["data"]["experiment"] = args.experiment
    if args.target_strategy is not None:
        config["features"]["target_strategy"] = args.target_strategy
    if args.exclude_company:
        config["features"]["include_company"] = False
    if args.no_mlflow:
        config["tracking"]["enabled"] = False
    if args.register_model:
        config["tracking"]["register_model"] = True
    if args.model_name:
        config["tracking"]["registered_model_name"] = args.model_name
    if args.model_alias:
        config["tracking"]["model_alias"] = args.model_alias

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or (
        f"catboost-{config['data']['experiment']}-{config['features']['target_strategy']}-{timestamp}"
    )
    output_root = root / config["artifacts"]["output_dir"]
    output_dir = output_root / run_name
    if output_dir.exists():
        raise FileExistsError(f"Artifact output already exists: {output_dir}")

    input_dir = root / config["data"]["input_dir"]
    files = discover_csv_files(input_dir, config["data"]["file_pattern"])
    print(f"Loading {len(files)} source CSV files from {input_dir}")
    if args.max_rows_per_file:
        print(
            "WARNING: --max-rows-per-file is active. Metrics from this run are only a smoke test."
        )
    raw = load_raw_data(files, max_rows_per_file=args.max_rows_per_file)
    prepared = prepare_model_data(
        raw, outlier_iqr_multiplier=float(config["data"]["outlier_iqr_multiplier"])
    )
    experiment_frame = select_experiment(
        prepared.frame, str(config["data"]["experiment"])
    )
    split, cutoffs = temporal_split(
        experiment_frame,
        train_fraction=float(config["data"]["train_fraction"]),
        validation_fraction=float(config["data"]["validation_fraction"]),
        cutoffs_from=prepared.frame,
    )
    usable = split.ne("excluded_missing_date")
    experiment_frame = experiment_frame.loc[usable].copy()
    split = split.loc[usable]
    features, categorical_features, metadata = build_features(
        experiment_frame,
        include_company=bool(config["features"]["include_company"]),
    )
    split_counts = {str(key): int(value) for key, value in split.value_counts().items()}
    for name in ("train", "validation", "test"):
        if split_counts.get(name, 0) == 0:
            raise ValueError(f"Temporal split produced no rows for {name}")

    train_mask = split.eq("train")
    validation_mask = split.eq("validation")
    test_mask = split.eq("test")
    strategy = str(config["features"]["target_strategy"])

    train_first, train_second, target_names = training_targets(
        metadata.loc[train_mask, "y_min_usd"],
        metadata.loc[train_mask, "y_max_usd"],
        strategy,
    )
    validation_first, validation_second, _ = training_targets(
        metadata.loc[validation_mask, "y_min_usd"],
        metadata.loc[validation_mask, "y_max_usd"],
        strategy,
    )

    print(
        f"Training rows={train_mask.sum():,}; validation={validation_mask.sum():,}; "
        f"test={test_mask.sum():,}; features={features.shape[1]}"
    )
    print(f"Categorical features: {', '.join(categorical_features)}")
    model_params = dict(config["model"])
    first_model = train_model(
        model_params,
        features.loc[train_mask],
        train_first,
        features.loc[validation_mask],
        validation_first,
        categorical_features,
    )
    second_model = train_model(
        model_params,
        features.loc[train_mask],
        train_second,
        features.loc[validation_mask],
        validation_second,
        categorical_features,
    )

    validation_predictions = predict_range(
        first_model, second_model, features.loc[validation_mask], strategy
    )
    test_predictions = predict_range(
        first_model, second_model, features.loc[test_mask], strategy
    )
    validation_metrics = regression_metrics(
        metadata.loc[validation_mask, "y_min_usd"].to_numpy(),
        metadata.loc[validation_mask, "y_max_usd"].to_numpy(),
        *validation_predictions,
    )
    test_metrics = regression_metrics(
        metadata.loc[test_mask, "y_min_usd"].to_numpy(),
        metadata.loc[test_mask, "y_max_usd"].to_numpy(),
        *test_predictions,
    )

    train_median_minimum = float(metadata.loc[train_mask, "y_min_usd"].median())
    train_median_maximum = float(metadata.loc[train_mask, "y_max_usd"].median())
    dummy_minimum = np.full(int(test_mask.sum()), train_median_minimum)
    dummy_maximum = np.full(int(test_mask.sum()), train_median_maximum)
    dummy_metrics = regression_metrics(
        metadata.loc[test_mask, "y_min_usd"].to_numpy(),
        metadata.loc[test_mask, "y_max_usd"].to_numpy(),
        dummy_minimum,
        dummy_maximum,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    first_model.save_model(output_dir / f"model_{target_names[0]}.cbm")
    second_model.save_model(output_dir / f"model_{target_names[1]}.cbm")

    importance = pd.DataFrame(
        {
            "feature": features.columns,
            f"importance_{target_names[0]}": first_model.get_feature_importance(),
            f"importance_{target_names[1]}": second_model.get_feature_importance(),
        }
    )
    importance["importance_mean"] = importance.iloc[:, 1:3].mean(axis=1)
    importance.sort_values("importance_mean", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )

    test_output = metadata.loc[test_mask].copy()
    test_output["predicted_y_min_usd"] = test_predictions[0]
    test_output["predicted_y_max_usd"] = test_predictions[1]
    test_output.to_csv(output_dir / "test_predictions.csv", index=False)
    segments = segment_metrics(
        metadata.loc[test_mask],
        test_predictions[0],
        test_predictions[1],
        minimum_size=int(config["data"]["minimum_segment_size"]),
    )
    pd.DataFrame(segments).to_csv(output_dir / "segment_metrics.csv", index=False)

    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
        "dummy_test": dummy_metrics,
    }
    manifest = {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "git_dirty": git_dirty(root),
        "config": config,
        "config_path": config_path.relative_to(root).as_posix(),
        "source_files": source_file_manifest(files),
        "dvc_pointers": dvc_pointer_manifest(input_dir),
        "preparation_audit": prepared.audit,
        "experiment_rows": len(experiment_frame),
        "split_counts": split_counts,
        "temporal_cutoffs": cutoffs,
        "feature_count": int(features.shape[1]),
        "feature_names": features.columns.tolist(),
        "categorical_features": categorical_features,
        "target_names": list(target_names),
        "package_versions": package_versions(),
        "smoke_test_max_rows_per_file": args.max_rows_per_file,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "manifest.json", manifest)

    tracking_result = None
    if config["tracking"]["enabled"]:
        tracking_result = log_to_mlflow(
            root=root,
            output_dir=output_dir,
            run_name=run_name,
            config=config,
            metrics=metrics,
            manifest=manifest,
            first_model_path=output_dir / f"model_{target_names[0]}.cbm",
            second_model_path=output_dir / f"model_{target_names[1]}.cbm",
            input_example=serving_example(experiment_frame.loc[train_mask].head(5)),
        )
        write_json(output_dir / "mlflow.json", tracking_result)

    summary = {
        "run_name": run_name,
        "output_dir": output_dir.relative_to(root).as_posix(),
        "test_mae_mean_usd": test_metrics["mae_mean_usd"],
        "dummy_mae_mean_usd": dummy_metrics["mae_mean_usd"],
        "improvement_over_dummy_pct": 100
        * (dummy_metrics["mae_mean_usd"] - test_metrics["mae_mean_usd"])
        / dummy_metrics["mae_mean_usd"],
        "mlflow": tracking_result,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def log_to_mlflow(
    root: Path,
    output_dir: Path,
    run_name: str,
    config: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    manifest: dict[str, Any],
    first_model_path: Path,
    second_model_path: Path,
    input_example: pd.DataFrame,
) -> dict[str, str]:
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", str(config["tracking"]["tracking_uri"]))
    if tracking_uri.startswith("sqlite:///"):
        database_path = root / tracking_uri.removeprefix("sqlite:///")
        database_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(config["tracking"]["experiment_name"]))
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "model_family": "catboost",
                "data_experiment": str(config["data"]["experiment"]),
                "target_strategy": str(config["features"]["target_strategy"]),
                "git_commit": str(manifest["git_commit"]),
                "git_dirty": str(manifest["git_dirty"]).lower(),
            }
        )
        mlflow.log_params(flatten_mapping(config))
        for split_name, split_metrics in metrics.items():
            mlflow.log_metrics(
                {f"{split_name}.{name}": value for name, value in split_metrics.items()}
            )
        mlflow.log_artifacts(str(output_dir))
        package = SalaryRangePyFunc(
            target_strategy=str(config["features"]["target_strategy"]),
            include_company=bool(config["features"]["include_company"]),
        )
        package.minimum_model = CatBoostRegressor()
        package.minimum_model.load_model(first_model_path)
        package.second_model = CatBoostRegressor()
        package.second_model.load_model(second_model_path)
        output_example = package.predict(None, input_example)
        register_model = bool(config["tracking"].get("register_model", False))
        model_name = str(config["tracking"].get("registered_model_name", "salary_predict_model"))
        with TemporaryDirectory(prefix="mlflow-code-", dir=output_dir) as temp_dir:
            code_path = build_mlflow_code_path(root, Path(temp_dir))
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=SalaryRangePyFunc(
                    target_strategy=str(config["features"]["target_strategy"]),
                    include_company=bool(config["features"]["include_company"]),
                ),
                artifacts={
                    "minimum_model": str(first_model_path),
                    "second_model": str(second_model_path),
                },
                code_paths=[str(code_path)],
                signature=infer_signature(input_example, output_example),
                input_example=input_example,
                pip_requirements=[
                    f"catboost=={importlib.metadata.version('catboost')}",
                    f"mlflow=={importlib.metadata.version('mlflow')}",
                    f"numpy=={importlib.metadata.version('numpy')}",
                    f"pandas=={importlib.metadata.version('pandas')}",
                    f"scikit-learn=={importlib.metadata.version('scikit-learn')}",
                ],
                registered_model_name=model_name if register_model else None,
                metadata={
                    "target_strategy": str(config["features"]["target_strategy"]),
                    "include_company": bool(config["features"]["include_company"]),
                    "validation_mae_mean_usd": metrics["validation"]["mae_mean_usd"],
                },
            )
        result = {
            "tracking_uri": tracking_uri,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
        }
        registered_version = getattr(model_info, "registered_model_version", None)
        if register_model and registered_version is None:
            client = MlflowClient(tracking_uri=tracking_uri)
            versions = client.search_model_versions(f"name='{model_name}'")
            matching = [version for version in versions if version.run_id == run.info.run_id]
            if matching:
                registered_version = max(matching, key=lambda item: int(item.version)).version
        if registered_version is not None:
            result["registered_model_name"] = model_name
            result["registered_model_version"] = str(registered_version)
            alias = str(config["tracking"].get("model_alias", "")).strip()
            if alias:
                client = MlflowClient(tracking_uri=tracking_uri)
                client.set_registered_model_alias(model_name, alias, str(registered_version))
                result["model_alias"] = alias
        return result


def build_mlflow_code_path(root: Path, destination: Path) -> Path:
    """Create the minimal importable package required by the pyfunc model."""

    source_package = root / "ml" / "catboost_baseline"
    target_ml = destination / "ml"
    target_package = target_ml / "catboost_baseline"
    target_package.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "ml" / "__init__.py", target_ml / "__init__.py")
    for name in ("__init__.py", "data.py", "evaluation.py", "features.py", "mlflow_model.py"):
        shutil.copy2(source_package / name, target_package / name)
    return target_ml


def flatten_mapping(value: dict[str, Any], prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten_mapping(item, name))
        else:
            result[name] = str(item)
    return result


def dvc_pointer_manifest(input_dir: Path) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for path in sorted(input_dir.glob("*.csv.dvc")):
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        output = content["outs"][0]
        manifest.append(
            {
                "path": path.name,
                "md5": str(output["md5"]),
                "size": str(output["size"]),
            }
        )
    return manifest


def package_versions() -> dict[str, str]:
    packages = ["catboost", "scikit-learn", "pandas", "numpy", "mlflow", "PyYAML"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def git_dirty(root: Path) -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
