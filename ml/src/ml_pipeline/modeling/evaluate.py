from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml_pipeline.common.io import ensure_parent, read_json, write_json
from ml_pipeline.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from ml_pipeline.modeling.qualify import compute_metrics, postprocess
from ml_pipeline.modeling.train import predict_range, predict_with_uncertainty
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.lineage import collect_lineage

# Suppress sklearn 1.9 TargetEncoder deprecation warning for shuffle & random_state
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.preprocessing._target_encoder")

LOGGER = logging.getLogger(__name__)

__all__ = ["evaluate"]


def run_segment_audit(audit_df: pd.DataFrame, min_n: int = 50) -> dict[str, list[dict[str, Any]]]:
    """Audit test prediction performance across operational segments."""
    segments: dict[str, list[dict[str, Any]]] = {}

    for col in ["country", "experience_level", "years_group", "work_mode"]:
        if col not in audit_df.columns:
            continue
        grouped = (
            audit_df.groupby(col, dropna=False, observed=True)
            .agg(
                n=("id", "size"),
                mediana_observada=("observado", "median"),
                mediana_predicha=("predicho", "median"),
                mae=("error_absoluto", "mean"),
                sesgo=("sesgo", "mean"),
            )
            .query("n >= @min_n")
            .sort_values("n", ascending=False)
            .reset_index()
        )
        records: list[dict[str, Any]] = []
        for r in grouped.to_dict(orient="records"):
            records.append({
                col: str(r[col]),
                "n": int(r["n"]),
                "mediana_observada": round(float(r["mediana_observada"]), 2),
                "mediana_predicha": round(float(r["mediana_predicha"]), 2),
                "mae": round(float(r["mae"]), 2),
                "sesgo": round(float(r["sesgo"]), 2),
            })
        segments[col] = records

    return segments


def run_novelty_audit(
    test_df: pd.DataFrame,
    trainval_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Audit test error between categories seen in training vs newly observed in test."""
    novel_results: dict[str, list[dict[str, Any]]] = {}

    for col in ["title", "company", "country"]:
        if col not in test_df.columns or col not in trainval_df.columns:
            continue
        known = set(trainval_df[col].dropna())
        flag = np.where(test_df[col].isin(known), "conocida", "nueva")
        temp = (
            audit_df.assign(estado_categoria=flag)
            .groupby("estado_categoria", observed=True)
            .agg(
                n=("id", "size"),
                mae=("error_absoluto", "mean"),
                sesgo=("sesgo", "mean"),
            )
            .reset_index()
        )
        records: list[dict[str, Any]] = []
        for r in temp.to_dict(orient="records"):
            records.append({
                "estado_categoria": str(r["estado_categoria"]),
                "n": int(r["n"]),
                "mae": round(float(r["mae"]), 2),
                "sesgo": round(float(r["sesgo"]), 2),
            })
        novel_results[col] = records

    return novel_results


def run_sensitivity_audit(
    bundle: dict[str, Any],
    trainval_df: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate controlled sensitivity across experience, geography, and skills."""
    base: dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            mode_vals = trainval_df[col].mode()
            base[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else "desconocido"
        else:
            base[col] = float(trainval_df[col].median()) if len(trainval_df[col].dropna()) > 0 else 0.0

    # 1. Unobserved profile
    new_profile = pd.DataFrame([{
        **base,
        "title": "cargo completamente nuevo",
        "country": "pais no observado",
        "region": "region no observada",
        "company": "empresa no observada",
        "experience_level": "MI",
        "experience_years": 4.0,
    }])
    new_pred = predict_range(new_profile, bundle)[0]
    limits = bundle["train_limits"]
    unobserved_valid = bool(
        np.isfinite(new_pred).all()
        and (new_pred > 0).all()
        and (new_pred[0] <= new_pred[1])
        and (new_pred[0] >= limits["floor"])
        and (new_pred[1] <= limits["ceiling"])
    )

    # 2. Experience progression
    exp_scenarios: list[dict[str, Any]] = []
    exp_levels = [("junior", 1.0), ("mid", 4.0), ("senior", 8.0), ("lead", 12.0)]
    exp_rows = [{**base, "experience_level": lvl, "experience_years": yrs} for lvl, yrs in exp_levels]
    exp_df = pd.DataFrame(exp_rows)[FEATURE_COLUMNS]
    exp_preds = predict_range(exp_df, bundle)
    exp_midpoints = (exp_preds[:, 0] + exp_preds[:, 1]) / 2.0
    non_decreasing_exp = bool((np.diff(exp_midpoints) >= 0).all())

    for i, (lvl, yrs) in enumerate(exp_levels):
        prog = float(exp_midpoints[i] - exp_midpoints[i - 1]) if i > 0 else 0.0
        exp_scenarios.append({
            "nivel": lvl,
            "años": yrs,
            "pred_y1": round(float(exp_preds[i, 0]), 2),
            "pred_y2": round(float(exp_preds[i, 1]), 2),
            "punto_medio": round(float(exp_midpoints[i]), 2),
            "progresion_respecto_anterior": round(prog, 2),
        })

    # 3. Geography scenarios
    top_countries = list(trainval_df["country"].value_counts().head(8).index)
    geo_rows = [{**base, "country": c} for c in top_countries]
    geo_df = pd.DataFrame(geo_rows)[FEATURE_COLUMNS]
    geo_preds = predict_range(geo_df, bundle)
    geo_scenarios: list[dict[str, Any]] = []
    for i, country in enumerate(top_countries):
        subset = trainval_df[trainval_df["country"] == country]
        obs_mid = float(((subset["y_min_usd"] + subset["y_max_usd"]) / 2.0).median()) if len(subset) > 0 else 0.0
        geo_scenarios.append({
            "country": country,
            "pred_y1": round(float(geo_preds[i, 0]), 2),
            "pred_y2": round(float(geo_preds[i, 1]), 2),
            "punto_medio_predicho": round(float((geo_preds[i, 0] + geo_preds[i, 1]) / 2.0), 2),
            "n_entrenamiento": int(len(subset)),
            "mediana_observada": round(obs_mid, 2),
        })

    # 4. Skill scenarios
    skill_sets: dict[str, list[str]] = {
        "sin_habilidades": [],
        "python_sql": ["python", "sql"],
        "datos_nube": ["python", "sql", "aws", "spark"],
        "mlops": ["python", "machine_learning", "docker", "kubernetes", "aws"],
    }
    skill_rows: list[dict[str, Any]] = []
    for label, skills in skill_sets.items():
        row = {**base}
        for s in [
            "python", "sql", "aws", "azure", "gcp", "spark",
            "docker", "kubernetes", "machine_learning", "pytorch",
            "tensorflow", "tableau", "power_bi",
        ]:
            row[f"skill_{s}"] = int(s in skills)
        skill_rows.append(row)
    skill_df = pd.DataFrame(skill_rows)[FEATURE_COLUMNS]
    skill_preds = predict_range(skill_df, bundle)
    skill_scenarios: list[dict[str, Any]] = []
    for i, (label, _) in enumerate(skill_sets.items()):
        skill_scenarios.append({
            "escenario": label,
            "pred_y1": round(float(skill_preds[i, 0]), 2),
            "pred_y2": round(float(skill_preds[i, 1]), 2),
            "punto_medio": round(float((skill_preds[i, 0] + skill_preds[i, 1]) / 2.0), 2),
        })

    return {
        "unobserved_profile": {
            "pred_y1": round(float(new_pred[0]), 2),
            "pred_y2": round(float(new_pred[1]), 2),
            "valido": unobserved_valid,
        },
        "experience_progression": {
            "scenarios": exp_scenarios,
            "progresion_no_decreciente": non_decreasing_exp,
        },
        "geography_scenarios": geo_scenarios,
        "skill_scenarios": skill_scenarios,
    }


def run_feature_importance_audit(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute average feature importance across dual LightGBM estimators."""
    pipe_min = bundle["pipeline_min"]
    pipe_max = bundle["pipeline_max"]
    estimators = [pipe_min.named_steps["modelo"], pipe_max.named_steps["modelo"]]
    importance = np.mean([model.feature_importances_ for model in estimators], axis=0)
    feature_names = list(pipe_min.named_steps["preprocesamiento"].get_feature_names_out())

    ranking: list[dict[str, Any]] = []
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        ranking.append({
            "variable": str(name),
            "importancia": round(float(imp), 2),
        })

    return ranking


def evaluate(settings: Settings) -> dict[str, Any]:
    """Execute final model evaluation and diagnostic audits on blind test.parquet."""
    model_path = settings.path("artifacts/work/model/model.joblib")
    test_path = settings.path("data/processed/test.parquet")
    train_path = settings.path("data/processed/train.parquet")
    val_path = settings.path("data/processed/validation.parquet")
    qual_path = settings.path("artifacts/reports/qualification.json")
    train_rep_path = settings.path("artifacts/reports/training.json")

    LOGGER.info("evaluate | loading model and test partition | model=%s test=%s", model_path, test_path)
    bundle: dict[str, Any] = joblib.load(model_path)
    test_df = pd.read_parquet(test_path)

    pipe_min = bundle["pipeline_min"]
    pipe_max = bundle["pipeline_max"]
    limits: dict[str, float] = bundle["train_limits"]
    uncertainty_margin = float(bundle["uncertainty_margin"])
    nominal_coverage = float(bundle.get("nominal_coverage", 0.80))
    feature_columns: list[str] = bundle.get("feature_columns", FEATURE_COLUMNS)
    algorithm: str = str(bundle.get("algorithm", "lightgbm"))
    configuration: dict[str, Any] = bundle.get("configuration", {})
    dataset_fingerprint: str = str(bundle.get("dataset_fingerprint", ""))

    # Inference on Test partition
    X_test = test_df[feature_columns]
    y_test = test_df[["y_min_usd", "y_max_usd"]].to_numpy()

    raw_min = np.expm1(pipe_min.predict(X_test))
    raw_max = np.expm1(pipe_max.predict(X_test))
    raw_test = np.column_stack([raw_min, raw_max])

    test_scores, test_pred = compute_metrics(y_test, raw_test, limits)

    # Uncertainty coverage on Test
    max_error = np.max(np.abs(y_test - test_pred), axis=1)
    uncertainty_test_coverage = float(np.mean(max_error <= uncertainty_margin))

    # Invariants and quality checks
    valores_finitos = bool(np.isfinite(test_pred).all())
    valores_positivos = bool((test_pred > 0).all())
    rangos_ordenados = bool((test_pred[:, 0] <= test_pred[:, 1]).all())
    dentro_limites_operativos = bool(
        (test_pred >= limits["floor"]).all() and (test_pred <= limits["ceiling"]).all()
    )

    quality_checks = {
        "valores_finitos": valores_finitos,
        "valores_positivos": valores_positivos,
        "rangos_ordenados": rangos_ordenados,
        "dentro_limites_operativos": dentro_limites_operativos,
    }

    # Consolidated audit DataFrame
    audit_df = test_df[["id", "country", "experience_level", "experience_years", "work_mode"]].copy()
    audit_df[["y1", "y2"]] = y_test
    audit_df[["pred_y1", "pred_y2"]] = test_pred
    audit_df["observado"] = (audit_df["y1"] + audit_df["y2"]) / 2.0
    audit_df["predicho"] = (audit_df["pred_y1"] + audit_df["pred_y2"]) / 2.0
    audit_df["error_absoluto"] = (audit_df["observado"] - audit_df["predicho"]).abs()
    audit_df["sesgo"] = audit_df["predicho"] - audit_df["observado"]
    audit_df["years_group"] = pd.cut(
        pd.to_numeric(audit_df["experience_years"], errors="coerce").where(lambda x: x.between(0, 50)),
        [-0.1, 2, 5, 10, 50],
        labels=["0-2", "3-5", "6-10", "11+"],
    )

    # Load trainval data for novelty and sensitivity audits
    trainval_df = pd.DataFrame()
    if train_path.is_file() and val_path.is_file():
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    # 1. Segment Audit
    segment_audit = run_segment_audit(audit_df, min_n=50)

    # 2. Novelty Audit
    novelty_audit = run_novelty_audit(test_df, trainval_df, audit_df) if len(trainval_df) > 0 else {}

    # 3. Sensitivity Audit
    sensitivity_audit = run_sensitivity_audit(bundle, trainval_df) if len(trainval_df) > 0 else {}
    diagnostic_checks = {
        "nuevo_perfil_valido": bool(
            sensitivity_audit.get("unobserved_profile", {}).get("valido", True)
        ),
        "progresion_experiencia_no_decreciente": bool(
            sensitivity_audit.get("experience_progression", {}).get("progresion_no_decreciente", True)
        ),
    }

    # 4. Feature Importance Audit
    feature_importance = run_feature_importance_audit(bundle)

    # Assemble metrics.json
    metrics_report: dict[str, Any] = {
        **{k: round(v, 6) for k, v in test_scores.items()},
        "uncertainty_margin": round(uncertainty_margin, 4),
        "uncertainty_nominal_coverage": nominal_coverage,
        "uncertainty_test_coverage": round(uncertainty_test_coverage, 6),
        "quality_checks": quality_checks,
        "diagnostic_checks": diagnostic_checks,
    }

    # Load qualification and training reports to connect candidate and manifest
    qual_data = read_json(qual_path) if qual_path.is_file() else {}
    training_data = read_json(train_rep_path) if train_rep_path.is_file() else {}

    # candidate.json: preserves qualification approval and records final test results
    eligible = bool(qual_data.get("eligible", True) and all(quality_checks.values()))
    reasons: list[str] = []
    if not qual_data.get("eligible", True):
        reasons.extend(qual_data.get("reasons", []))
    for check_name, passed in quality_checks.items():
        if not passed:
            reasons.append(f"Quality check '{check_name}' failed")

    candidate_report: dict[str, Any] = {
        "algorithm": algorithm,
        "eligible": eligible,
        "reasons": reasons,
        "dataset_fingerprint": dataset_fingerprint,
        "qualification": qual_data,
        "training": {
            "train_rows": training_data.get("train_rows", len(trainval_df)),
            "validation_rows": training_data.get("validation_rows", 0),
            "final_training_rows": training_data.get("final_training_rows", len(trainval_df)),
            "feature_count": len(feature_columns),
            "train_limits": limits,
        },
        "test_evaluation": {
            "test_rows": len(test_df),
            "primary_metric": "mae_promedio",
            "primary_metric_value": round(test_scores["mae_promedio"], 4),
            "metrics": metrics_report,
            "uncertainty": {
                "margin_usd": round(uncertainty_margin, 4),
                "nominal_coverage": nominal_coverage,
                "test_coverage": round(uncertainty_test_coverage, 6),
            },
            "quality_checks": quality_checks,
            "diagnostic_checks": diagnostic_checks,
        },
    }

    # Lineage and experiment_manifest.json
    lineage = collect_lineage(settings)
    experiment_manifest: dict[str, Any] = {
        "dataset_fingerprint": dataset_fingerprint,
        "algorithm": algorithm,
        "configuration": configuration,
        "candidate": eligible,
        "primary_metric": "mae_promedio",
        "primary_metric_value": round(test_scores["mae_promedio"], 4),
        "train_limits": limits,
        "uncertainty": {
            "margin_usd": round(uncertainty_margin, 4),
            "nominal_coverage": nominal_coverage,
            "test_coverage": round(uncertainty_test_coverage, 6),
        },
        "qualification": qual_data,
        "training": training_data,
        "final_test_metrics": metrics_report,
        "lineage": lineage,
    }

    # Persist all reports
    metrics_path = settings.path("artifacts/reports/metrics.json")
    candidate_path = settings.path("artifacts/reports/candidate.json")
    manifest_path = settings.path("artifacts/reports/experiment_manifest.json")
    audit_segments_path = settings.path("artifacts/reports/audit_segments.json")
    audit_novelty_path = settings.path("artifacts/reports/audit_novelty.json")
    audit_sensitivity_path = settings.path("artifacts/reports/audit_sensitivity.json")
    audit_feat_path = settings.path("artifacts/reports/feature_importance.json")

    ensure_parent(metrics_path)
    write_json(metrics_path, metrics_report)
    write_json(candidate_path, candidate_report)
    write_json(manifest_path, experiment_manifest)
    write_json(audit_segments_path, segment_audit)
    write_json(audit_novelty_path, novelty_audit)
    write_json(audit_sensitivity_path, sensitivity_audit)
    write_json(audit_feat_path, {"ranking": feature_importance})

    LOGGER.info(
        "evaluate | complete | mae_promedio=%.2f coverage=%.2f%% eligible=%s",
        test_scores["mae_promedio"],
        uncertainty_test_coverage * 100,
        eligible,
    )

    return candidate_report
