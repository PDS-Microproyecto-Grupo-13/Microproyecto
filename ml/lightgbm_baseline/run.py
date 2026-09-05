from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import shutil
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Reused as-is: these modules have no CatBoost-specific logic, import
# them directly from the catboost's package.
from ml.catboost_baseline.data import (
    discover_csv_files,
    load_raw_data,
    prepare_model_data,
    select_experiment,
    source_file_manifest,
    temporal_split,
)
from ml.catboost_baseline.evaluation import (
    reconstruct_targets,
    regression_metrics,
    segment_metrics,
    training_targets,
)
from ml.catboost_baseline.features import build_features
from ml.catboost_baseline.mlflow_model import serving_example

from .evaluation_extras import (
    company_generalization_dummy_metrics,
    company_generalization_metrics,
    company_generalization_summary,
)
from .mlflow_model import SalaryRangeLightGBMPyFunc, dump_category_levels  # noqa: F401
from .model import feature_importance_frame, to_lightgbm_frame, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LightGBM salary-range baseline")
    parser.add_argument(
        "--config",
        default="ml/lightgbm_baseline/params.yaml",
        help="YAML configuration path, relative to the repository root",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Smoke-test only: read at most this many rows from each CSV",
    )
    parser.add_argument("--iterations", type=int, default=None, help="Override model n_estimators")
    parser.add_argument("--experiment", choices=["reported", "expanded"], default=None)
    parser.add_argument("--target-strategy", choices=["direct", "minimum_width_log"], default=None)
    parser.add_argument(
        "--exclude-company",
        action="store_true",
        help="Exclude company identity to test generalization to unseen employers",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--no-company-generalization",
        action="store_true",
        help="Skip the seen/unseen-company generalization check (overrides evaluation.company_generalization.enabled)",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the logged pyfunc model in MLflow's Model Registry (overrides tracking.register_model)",
    )
    parser.add_argument(
        "--model-alias",
        default=None,
        help="Alias to set on the registered model version, e.g. 'champion' (overrides tracking.model_alias)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Registered model name to use (overrides tracking.registered_model_name)",
    )
    return parser.parse_args()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def predict_range(
    first_model,
    second_model,
    features: pd.DataFrame,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return reconstruct_targets(first_model.predict(features), second_model.predict(features), strategy)


def main() -> int:
    args = parse_args()
    root = repository_root()
    config_path = (root / args.config).resolve()
    config = load_config(config_path)
    if args.iterations is not None:
        config["model"]["n_estimators"] = args.iterations
    if args.experiment is not None:
        config["data"]["experiment"] = args.experiment
    if args.target_strategy is not None:
        config["features"]["target_strategy"] = args.target_strategy
    if args.exclude_company:
        config["features"]["include_company"] = False
    if args.register_model:
        config["tracking"]["register_model"] = True
    if args.model_alias is not None:
        config["tracking"]["model_alias"] = args.model_alias
    if args.model_name is not None:
        config["tracking"]["registered_model_name"] = args.model_name
    if args.no_mlflow:
        config["tracking"]["enabled"] = False
    config.setdefault("evaluation", {}).setdefault(
        "company_generalization", {"enabled": True, "minimum_size": 30}
    )
    if args.no_company_generalization:
        config["evaluation"]["company_generalization"]["enabled"] = False

    input_dir = root / config["data"]["input_dir"]
    files = discover_csv_files(input_dir, config["data"]["file_pattern"])
    print(f"Loading {len(files)} source CSV files from {input_dir}")
    if args.max_rows_per_file:
        print("WARNING: --max-rows-per-file is active. Metrics from this run are only a smoke test.")
    raw = load_raw_data(files, max_rows_per_file=args.max_rows_per_file)
    prepared = prepare_model_data(raw, outlier_iqr_multiplier=float(config["data"]["outlier_iqr_multiplier"]))
    experiment_frame = select_experiment(prepared.frame, str(config["data"]["experiment"]))
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
    # LightGBM-specific step: fix category levels on the full frame before
    # splitting, then slice with the same masks CatBoost uses.
    features = to_lightgbm_frame(features, categorical_features)

    split_counts = {str(key): int(value) for key, value in split.value_counts().items()}
    for name in ("train", "validation", "test"):
        if split_counts.get(name, 0) == 0:
            raise ValueError(f"Temporal split produced no rows for {name}")

    train_mask = split.eq("train")
    validation_mask = split.eq("validation")
    test_mask = split.eq("test")
    strategy = str(config["features"]["target_strategy"])

    train_first, train_second, target_names = training_targets(
        metadata.loc[train_mask, "y_min_usd"], metadata.loc[train_mask, "y_max_usd"], strategy
    )
    validation_first, validation_second, _ = training_targets(
        metadata.loc[validation_mask, "y_min_usd"], metadata.loc[validation_mask, "y_max_usd"], strategy
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

    validation_predictions = predict_range(first_model, second_model, features.loc[validation_mask], strategy)
    test_predictions = predict_range(first_model, second_model, features.loc[test_mask], strategy)
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"lightgbm-{config['data']['experiment']}-{strategy}-{timestamp}"
    output_root = root / config["artifacts"]["output_dir"]
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    first_model.booster_.save_model(str(output_dir / f"model_{target_names[0]}.txt"))
    second_model.booster_.save_model(str(output_dir / f"model_{target_names[1]}.txt"))

    # Category levels as fixed by to_lightgbm_frame on the full pre-split
    # frame -- must be re-applied at inference time (see mlflow_model.py's
    # SalaryRangeLightGBMPyFunc._apply_training_categories), or a served
    # request gets silently wrong predictions rather than an error.
    levels_path = dump_category_levels(
        features, categorical_features, output_dir / "category_levels.json"
    )

    importance = feature_importance_frame(first_model, second_model, list(features.columns), target_names[1])
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

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

    # Seen-vs-unseen company generalization check (additive; see
    # ml/lightgbm_baseline/evaluation.py -- not part of the shared
    # catboost_baseline pipeline). Uses the same test_predictions as the
    # headline test_metrics above, and a train-median dummy computed per
    # segment so "% improvement over dummy" is meaningful for the
    # unseen-company population specifically, not just in aggregate.
    #
    # Toggleable via evaluation.company_generalization.enabled in params.yaml
    # (or --no-company-generalization).
    company_generalization_config = config["evaluation"]["company_generalization"]
    company_summary: dict[str, object] | None = None
    company_model_segments: list[dict[str, object]] = []
    if company_generalization_config.get("enabled", True):
        company_summary = company_generalization_summary(experiment_frame["company"], train_mask, test_mask)
        company_model_segments = company_generalization_metrics(
            experiment_frame["company"],
            metadata.loc[test_mask],
            train_mask,
            test_mask,
            test_predictions[0],
            test_predictions[1],
            minimum_size=int(company_generalization_config.get("minimum_size", 30)),
        )
        company_dummy_segments = company_generalization_dummy_metrics(
            experiment_frame["company"],
            metadata,
            train_mask,
            test_mask,
            minimum_size=int(company_generalization_config.get("minimum_size", 30)),
        )
        company_dummy_mae_by_value = {row["value"]: row["mae_mean_usd"] for row in company_dummy_segments}
        for row in company_model_segments:
            dummy_mae = company_dummy_mae_by_value.get(row["value"])
            row["dummy_mae_mean_usd"] = dummy_mae
            row["improvement_over_dummy_pct"] = (
                100 * (dummy_mae - row["mae_mean_usd"]) / dummy_mae if dummy_mae else None
            )
        pd.DataFrame(company_model_segments).to_csv(output_dir / "company_generalization_metrics.csv", index=False)
        write_json(output_dir / "company_generalization_summary.json", company_summary)
    else:
        print("Skipping company generalization metrics (evaluation.company_generalization.enabled=false)")

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
        "experiment_rows": int(len(experiment_frame)),
        "split_counts": split_counts,
        "temporal_cutoffs": cutoffs,
        "feature_count": int(features.shape[1]),
        "feature_names": features.columns.tolist(),
        "categorical_features": categorical_features,
        "target_names": list(target_names),
        "package_versions": package_versions(),
        "smoke_test_max_rows_per_file": args.max_rows_per_file,
        "company_generalization_enabled": bool(company_generalization_config.get("enabled", True)),
        "company_generalization_summary": company_summary,
        "best_iteration": {
            target_names[0]: getattr(first_model, "best_iteration_", None),
            target_names[1]: getattr(second_model, "best_iteration_", None),
        },
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
            first_model_path=output_dir / f"model_{target_names[0]}.txt",
            second_model_path=output_dir / f"model_{target_names[1]}.txt",
            levels_path=levels_path,
            input_example=serving_example(experiment_frame.loc[test_mask].head(2)),
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
    if company_summary is not None:
        summary["company_generalization"] = {
            "unseen_company_pct": company_summary["unseen_company_pct"],
            "segments": [
                {"value": row["value"], "mae_mean_usd": row["mae_mean_usd"], "rows": row["rows"]}
                for row in company_model_segments
            ],
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
    levels_path: Path,
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
                "model_family": "lightgbm",
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

        # Se instancia el paquete y se ejecuta una predicción de ejemplo, para
        # que MLflow pueda inferir la firma del contrato (entrada -> salida).
        package = SalaryRangeLightGBMPyFunc(
            target_strategy=str(config["features"]["target_strategy"]),
            include_company=bool(config["features"]["include_company"]),
        )
        import lightgbm as lgb

        package.minimum_model = lgb.Booster(model_file=str(first_model_path))
        package.second_model = lgb.Booster(model_file=str(second_model_path))
        package.category_levels = json.loads(levels_path.read_text(encoding="utf-8"))
        output_example = package.predict(None, input_example)

        register_model = bool(config["tracking"].get("register_model", False))
        model_name = str(config["tracking"].get("registered_model_name", "salary_predict_model"))

        with TemporaryDirectory(prefix="mlflow-code-", dir=output_dir) as temp_dir:
            code_path = build_mlflow_code_path(root, Path(temp_dir))
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=SalaryRangeLightGBMPyFunc(
                    target_strategy=str(config["features"]["target_strategy"]),
                    include_company=bool(config["features"]["include_company"]),
                ),
                artifacts={
                    "minimum_model": str(first_model_path),
                    "second_model": str(second_model_path),
                    "category_levels": str(levels_path),
                },
                code_paths=[str(code_path)],
                signature=infer_signature(input_example, output_example),
                input_example=input_example,
                pip_requirements=[
                    f"lightgbm=={importlib.metadata.version('lightgbm')}",
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
            matching = [v for v in versions if v.run_id == run.info.run_id]
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
    """Empaqueta el código mínimo que el modelo necesita para cargarse.

    OJO: copia LOS DOS paquetes. El envoltorio de LightGBM importa
    data.py, features.py y evaluation.py desde catboost_baseline; si sólo se
    copia lightgbm_baseline, el modelo no carga en el contenedor de inferencia.
    """
    target_ml = destination / "ml"
    target_ml.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "ml" / "__init__.py", target_ml / "__init__.py")

    source_catboost = root / "ml" / "catboost_baseline"
    target_catboost = target_ml / "catboost_baseline"
    target_catboost.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "data.py", "evaluation.py", "features.py", "mlflow_model.py"):
        shutil.copy2(source_catboost / name, target_catboost / name)

    source_lightgbm = root / "ml" / "lightgbm_baseline"
    target_lightgbm = target_ml / "lightgbm_baseline"
    target_lightgbm.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "model.py", "mlflow_model.py"):
        shutil.copy2(source_lightgbm / name, target_lightgbm / name)

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
        manifest.append({"path": path.name, "md5": str(output["md5"]), "size": str(output["size"])})
    return manifest


def package_versions() -> dict[str, str]:
    packages = ["lightgbm", "scikit-learn", "pandas", "numpy", "mlflow", "PyYAML"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def git_dirty(root: Path) -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
