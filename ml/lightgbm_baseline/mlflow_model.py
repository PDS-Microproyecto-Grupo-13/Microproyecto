"""Envoltorio MLflow pyfunc para el baseline LightGBM de rangos salariales.

Adaptado de ``ml/catboost_baseline/mlflow_model.py``.

DIFERENCIA ESTRUCTURAL CON CATBOOST
-----------------------------------
CatBoost consume las variables categóricas como texto crudo; LightGBM las
consume como códigos enteros de una categoría de pandas. Esos códigos dependen
de los NIVELES con que se construyó la categoría.

En entrenamiento, ``to_lightgbm_frame`` hace ``astype("category")`` sobre el
frame completo, así que los niveles son todos los valores del dataset. En
inferencia llega UNA fila: si se repitiera ``astype("category")``, esa fila
tendría un solo nivel y su código sería 0 — y LightGBM lo interpretaría como la
primera categoría del entrenamiento. No lanza error: predice mal en silencio.

Por eso los niveles de entrenamiento viajan como artefacto dentro del modelo y
se vuelven a aplicar en ``predict``. Es la pieza que no se puede copiar de
CatBoost.

"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow.pyfunc
import numpy as np
import pandas as pd

# Se reutilizan sin cambios: no tienen nada específico de CatBoost.
from ml.catboost_baseline.evaluation import reconstruct_targets
from ml.catboost_baseline.features import build_inference_features
from ml.catboost_baseline.mlflow_model import SERVING_COLUMNS, serving_example

__all__ = ["SalaryRangeLightGBMPyFunc", "dump_category_levels", "SERVING_COLUMNS"]


def dump_category_levels(
    features: pd.DataFrame,
    categorical_features: list[str],
    destination: Path,
) -> Path:
    """Guarda los niveles de cada columna categórica tal como quedaron al entrenar.

    Llamar DESPUÉS de ``to_lightgbm_frame`` y sobre el frame COMPLETO
    (pre-split), que es donde se fijaron los niveles. Si se llama sobre el
    subconjunto de entrenamiento, los niveles que sólo aparecen en validación o
    prueba se pierden y la inferencia los mapeará mal.
    """
    levels: dict[str, list[str]] = {}
    for column in categorical_features:
        if column not in features:
            continue
        serie = features[column]
        if not isinstance(serie.dtype, pd.CategoricalDtype):
            raise TypeError(
                f"La columna '{column}' no es categórica. "
                "Ejecute to_lightgbm_frame antes de volcar los niveles."
            )
        levels[column] = [str(value) for value in serie.cat.categories]

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(levels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


class SalaryRangeLightGBMPyFunc(mlflow.pyfunc.PythonModel):
    """Modelo portable: preprocesamiento + los dos boosters LightGBM.

    Recibe los 11 campos crudos de la vacante (SERVING_COLUMNS) y devuelve el
    rango salarial. El backend no necesita conocer las 54 variables.
    """

    def __init__(self, target_strategy: str, include_company: bool) -> None:
        self.target_strategy = target_strategy
        self.include_company = include_company
        self.minimum_model: lgb.Booster | None = None
        self.second_model: lgb.Booster | None = None
        self.category_levels: dict[str, list[str]] = {}

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.minimum_model = lgb.Booster(model_file=context.artifacts["minimum_model"])
        self.second_model = lgb.Booster(model_file=context.artifacts["second_model"])
        with open(context.artifacts["category_levels"], encoding="utf-8") as handle:
            self.category_levels = json.load(handle)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        del context, params
        if self.minimum_model is None or self.second_model is None:
            raise RuntimeError("Los artefactos del modelo no fueron cargados")

        raw = serving_example(pd.DataFrame(model_input))
        features, _ = build_inference_features(raw, include_company=self.include_company)
        features = self._apply_training_categories(features)

        first = np.asarray(self.minimum_model.predict(features), dtype=float)
        second = np.asarray(self.second_model.predict(features), dtype=float)
        minimum, maximum, _, _ = reconstruct_targets(first, second, self.target_strategy)

        return pd.DataFrame(
            {
                "salary_min_usd": minimum,
                "salary_max_usd": maximum,
                "salary_midpoint_usd": (minimum + maximum) / 2,
            },
            index=model_input.index,
        )

    def _apply_training_categories(self, features: pd.DataFrame) -> pd.DataFrame:
        """Reaplica los niveles exactos vistos en entrenamiento.

        Un valor que no estaba en entrenamiento (una empresa nueva, un país
        nuevo) queda como NaN, que es lo correcto: LightGBM lo trata como
        faltante en vez de confundirlo con otra categoría.
        """
        frame = features.copy()
        for column, levels in self.category_levels.items():
            if column not in frame:
                continue
            dtype = pd.CategoricalDtype(categories=levels)
            frame[column] = frame[column].astype(str).astype(dtype)
        return frame
