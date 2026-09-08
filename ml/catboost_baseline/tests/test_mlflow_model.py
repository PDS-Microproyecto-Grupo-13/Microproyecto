from pathlib import Path
from types import SimpleNamespace

import mlflow.pyfunc
import pandas as pd
from catboost import CatBoostRegressor
from mlflow.models import infer_signature

from ml.catboost_baseline.features import build_inference_features
from ml.catboost_baseline.mlflow_model import SalaryRangePyFunc, serving_example
from ml.catboost_baseline.run import build_mlflow_code_path


def test_packaged_model_loads_both_heads_and_predicts_ordered_range(tmp_path) -> None:
    raw = serving_example(
        pd.DataFrame(
            [
                {"title": "Data Analyst", "experience_level": "EN", "experience_years": 1, "countries": "Colombia", "has_remote": False, "company": "A"},
                {"title": "Data Scientist", "experience_level": "MI", "experience_years": 3, "countries": "Mexico", "has_remote": True, "company": "B"},
                {"title": "ML Engineer", "experience_level": "SE", "experience_years": 7, "countries": "United States", "has_remote": True, "company": "C"},
            ]
        )
    )
    features, categorical = build_inference_features(raw)
    minimum = CatBoostRegressor(iterations=3, verbose=False, random_seed=42, allow_writing_files=False)
    width = CatBoostRegressor(iterations=3, verbose=False, random_seed=42, allow_writing_files=False)
    minimum.fit(features, [10.0, 11.0, 12.0], cat_features=categorical)
    width.fit(features, [8.0, 9.0, 10.0], cat_features=categorical)
    minimum_path = tmp_path / "minimum.cbm"
    width_path = tmp_path / "width.cbm"
    minimum.save_model(minimum_path)
    width.save_model(width_path)

    packaged = SalaryRangePyFunc("minimum_width_log", include_company=True)
    packaged.load_context(SimpleNamespace(artifacts={"minimum_model": str(minimum_path), "second_model": str(width_path)}))
    prediction = packaged.predict(None, raw.head(1))

    assert list(prediction.columns) == ["salary_min_usd", "salary_max_usd", "salary_midpoint_usd"]
    assert prediction.loc[0, "salary_max_usd"] >= prediction.loc[0, "salary_min_usd"]

    model_path = tmp_path / "mlflow_model"
    code_path = build_mlflow_code_path(Path(__file__).resolve().parents[3], tmp_path / "code")
    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=SalaryRangePyFunc("minimum_width_log", include_company=True),
        artifacts={"minimum_model": str(minimum_path), "second_model": str(width_path)},
        code_paths=[str(code_path)],
        signature=infer_signature(raw.head(1), prediction),
        input_example=raw.head(1),
        pip_requirements=[],
    )
    reloaded = mlflow.pyfunc.load_model(model_path)
    reloaded_prediction = reloaded.predict(raw.head(1))
    assert reloaded_prediction.loc[0, "salary_max_usd"] >= reloaded_prediction.loc[0, "salary_min_usd"]
