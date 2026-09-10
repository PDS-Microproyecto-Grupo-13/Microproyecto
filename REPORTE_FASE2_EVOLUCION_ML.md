# Reporte Final: Fase 2 — Trazabilidad Experimental y Filtrado Multi-Algoritmo en `ml/`

Se completó con éxito la Fase 2 de evolución del módulo `ml/`, enfocada exclusivamente en la **trazabilidad experimental** local y remota, garantizando que el algoritmo utilizado quede claramente identificado en artifacts, logs y MLflow, y que sólo se registren sus hiperparámetros efectivos.

---

### 1. Archivos Modificados y Creados

* [`ml/src/ml_pipeline/modeling/factory.py`](ml/src/ml_pipeline/modeling/factory.py):
  * Se implementó la función `effective_model_params(config: dict[str, Any]) -> dict[str, Any]` como única fuente de verdad para extraer los hiperparámetros efectivos del algoritmo seleccionado.
  * Se refactorizaron `_build_logistic_regression` y `_build_random_forest` para consumir directamente dicha función.
* [`ml/src/ml_pipeline/modeling/evaluate.py`](ml/src/ml_pipeline/modeling/evaluate.py):
  * Se enriqueció `artifacts/reports/experiment_manifest.json` con el contexto del modelo (`algorithm` y diccionario `model.parameters`).
  * Se actualizó el log final de evaluación para incluir el algoritmo (`algorithm=...`).
* [`ml/src/ml_pipeline/tracking/mlflow_tracker.py`](ml/src/ml_pipeline/tracking/mlflow_tracker.py):
  * Se implementaron `filter_inactive_algorithm_params` y `effective_tracking_params` para aislar los hiperparámetros del algoritmo activo y omitir los del inactivo.
  * Se configuró el run de MLflow con `run_name=algorithm` y se añadió el tag explícito `"algorithm": algorithm`.
  * Se incorporó `"algorithm": algorithm` en `artifacts/reports/tracking.json`.
* [`ml/dvc.yaml`](ml/dvc.yaml):
  * Se añadieron `src/ml_pipeline/modeling/factory.py` a las dependencias de la etapa `evaluate` y `model` a sus parámetros (`params: [evaluation, model]`).
* [`ml/tests/unit/test_factory.py`](ml/tests/unit/test_factory.py):
  * Se agregaron tests unitarios para `effective_model_params` tanto para `logistic_regression` como para `random_forest`.
* [`ml/tests/integration/test_pipeline.py`](ml/tests/integration/test_pipeline.py):
  * Se añadieron aserciones para validar que `experiment_manifest.json` contenga la clave `algorithm` y el bloque `model.parameters` con los hiperparámetros respectivos.
* [`ml/tests/unit/test_tracking.py`](ml/tests/unit/test_tracking.py) *(Nuevo)*:
  * Suite unitaria y de integración local de tracking que valida el filtrado de parámetros inactivos, creación de run, asignación de tags, nombrado del run y persistencia del modelo para ambos algoritmos.
* [`ml/README.md`](ml/README.md):
  * Actualizada la sección de tracking y outputs para documentar la trazabilidad de parámetros efectivos, tags y el manifiesto.

---

### 2. Comportamiento Anterior de Tracking

Anteriormente, `flatten_params` aplanaba ciegamente la totalidad del diccionario de `params.yaml`. Esto causaba que un run de `logistic_regression` registrara parámetros como `model.random_forest.n_estimators`, `model.random_forest.max_depth` y `model.random_forest.min_samples_leaf`, a pesar de que Random Forest nunca participó en el entrenamiento. Asimismo, el run de MLflow no contaba con un tag dedicado `algorithm` ni un `run_name` representativo.

---

### 3. Nueva Lógica para Obtener Parámetros Efectivos

Se introdujo una separación limpia en dos niveles:

1. **Parámetros locales del modelo (`factory.py`):**
   ```python
   def effective_model_params(config: dict[str, Any]) -> dict[str, Any]:
       algorithm = config.get("algorithm")
       if algorithm == "logistic_regression":
           algo_config = config.get("logistic_regression") or {}
           return {
               "random_state": int(algo_config.get("random_state", config.get("random_state", 42))),
               "C": float(algo_config.get("C", config.get("C", 1.0))),
               "max_iter": int(algo_config.get("max_iter", config.get("max_iter", 1000))),
           }
       if algorithm == "random_forest":
           algo_config = config.get("random_forest") or {}
           max_depth = algo_config.get("max_depth", config.get("max_depth", 10))
           return {
               "random_state": int(algo_config.get("random_state", config.get("random_state", 42))),
               "n_estimators": int(algo_config.get("n_estimators", config.get("n_estimators", 200))),
               "max_depth": int(max_depth) if max_depth is not None else None,
               "min_samples_leaf": int(algo_config.get("min_samples_leaf", config.get("min_samples_leaf", 1))),
           }
   ```
2. **Filtrado de ramas inactivas para MLflow (`mlflow_tracker.py`):**
   ```python
   def filter_inactive_algorithm_params(params: dict[str, Any]) -> dict[str, Any]:
       filtered = {k: (dict(v) if isinstance(v, dict) else v) for k, v in params.items()}
       model_section = filtered.get("model")
       if isinstance(model_section, dict):
           model_copy = dict(model_section)
           algorithm = model_copy.get("algorithm")
           if algorithm == "logistic_regression":
               model_copy.pop("random_forest", None)
           elif algorithm == "random_forest":
               model_copy.pop("logistic_regression", None)
           filtered["model"] = model_copy
       return filtered

   def effective_tracking_params(params: dict[str, Any]) -> dict[str, Any]:
       return flatten_params(filter_inactive_algorithm_params(params))
   ```

---

### 4. Estructura Final Relevante de `experiment_manifest.json`

`artifacts/reports/experiment_manifest.json` ahora documenta el contexto completo sin contaminar `metrics.json`:

```json
{
  "algorithm": "logistic_regression",
  "candidate": true,
  "dataset_fingerprint": "75a1d74a59df9cb0a78ecc0a36085bf8a85239557f178d3b8d758e88b26416e1",
  "dvc_revision": "c593d2d8c1960d939bd1f4ee8068d1a217e1b4776b3fd0960d450ac7fb7227a6",
  "git_commit": "eb39618767de4eb8c5fc77a5ec62bb94e46dfbb0",
  "git_dirty": true,
  "model": {
    "algorithm": "logistic_regression",
    "parameters": {
      "C": 1.0,
      "max_iter": 1000,
      "random_state": 42
    }
  },
  "params_hash": "6495e64810a5ecf5946447816c0a10026c7901271fc309838805ac5eb0f74069",
  "primary_metric": "f1",
  "primary_metric_value": 0.9861111111111112
}
```

---

### 5. Ejemplo de Parámetros Registrados para Logistic Regression

```text
data.random_state: 42
data.test_size: 0.2
evaluation.minimum_score: 0.8
evaluation.primary_metric: f1
model.algorithm: logistic_regression
model.logistic_regression.C: 1.0
model.logistic_regression.max_iter: 1000
model.random_state: 42
```
*(Ningún parámetro con prefijo `model.random_forest` fue registrado).*

---

### 6. Ejemplo de Parámetros Registrados para Random Forest

```text
data.random_state: 42
data.test_size: 0.2
evaluation.minimum_score: 0.8
evaluation.primary_metric: f1
model.algorithm: random_forest
model.random_forest.max_depth: 10
model.random_forest.min_samples_leaf: 1
model.random_forest.n_estimators: 200
model.random_state: 42
```
*(Ningún parámetro con prefijo `model.logistic_regression` fue registrado).*

---

### 7. Tags Utilizados en MLflow

* **Tag de algoritmo:** `algorithm = "logistic_regression"` / `algorithm = "random_forest"`
* **Nombre del Run (`run_name`):** `logistic_regression` / `random_forest`
* **Tags de linaje:**
  * `lineage.git_commit`
  * `lineage.git_dirty`
  * `lineage.dvc_revision`
  * `lineage.params_hash`
  * `lineage.dataset_fingerprint`
  * `lineage.dvc_lock_hash`

---

### 8. Resultado de `pytest`

Ejecutado con `.venv/bin/pytest`:

```text
============================= test session starts ==============================
platform linux -- Python 3.11.2, pytest-8.4.2, pluggy-1.6.0
rootdir: /home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml
configfile: pyproject.toml
testpaths: tests
plugins: hydra-core-1.3.6, anyio-4.14.2
collected 25 items                                                             

tests/contract/test_model_contract.py ..                                 [  8%]
tests/integration/test_pipeline.py ..                                    [ 16%]
tests/unit/test_candidate.py ..                                          [ 24%]
tests/unit/test_factory.py .........                                     [ 60%]
tests/unit/test_lineage.py .                                             [ 64%]
tests/unit/test_metrics.py .                                             [ 68%]
tests/unit/test_registry.py .                                            [ 72%]
tests/unit/test_settings.py .                                            [ 76%]
tests/unit/test_tracking.py ....                                         [ 92%]
tests/unit/test_validation.py ..                                         [100%]

============================= 25 passed in 56.30s ==============================
```

---

### 9. Resultado de `dvc repro` para Ambos Algoritmos

* **Logistic Regression:**
  * Etapas `train` y `evaluate` ejecutadas:
    ```text
    INFO | ml_pipeline.modeling.train | train | algorithm=logistic_regression | input=... | output=...
    INFO | ml_pipeline.modeling.train | train | result=success | algorithm=logistic_regression | features=30
    INFO | ml_pipeline.modeling.evaluate | evaluate | result=success | algorithm=logistic_regression | f1=0.986111 | eligible=True
    ```
* **Random Forest:**
  * Al conmutar `model.algorithm: random_forest`, DVC invalidó y reconstruyó únicamente `train` y `evaluate`:
    ```text
    INFO | ml_pipeline.modeling.train | train | algorithm=random_forest | input=... | output=...
    INFO | ml_pipeline.modeling.train | train | result=success | algorithm=random_forest | features=30
    INFO | ml_pipeline.modeling.evaluate | evaluate | result=success | algorithm=random_forest | f1=0.965517 | eligible=True
    ```

---

### 10. Run ID de Prueba de Logistic Regression

* **Run ID:** `a3d3795374b24e5397938854f37f2ac0`
* **Run Name:** `logistic_regression`
* **Model URI:** `models:/m-dd69e881e67243bc8ec1e5382c442317`

---

### 11. Run ID de Prueba de Random Forest

* **Run ID:** `d14611795c58493799a24faa39b9d552`
* **Run Name:** `random_forest`
* **Model URI:** `models:/m-7b20501bc7074954833a8e5e59b8831c`

---

### 12. Confirmación de Hiperparámetros Exclusivos por Run

Se inspeccionó directamente la base de datos de MLflow:
* En el Run `a3d3795374b24e5397938854f37f2ac0` (`logistic_regression`), las únicas claves bajo `model.` fueron:
  * `model.algorithm`, `model.random_state`, `model.logistic_regression.C`, `model.logistic_regression.max_iter`.
  * **Confirmado:** 0 parámetros de `random_forest`.
* En el Run `d14611795c58493799a24faa39b9d552` (`random_forest`), las únicas claves bajo `model.` fueron:
  * `model.algorithm`, `model.random_state`, `model.random_forest.n_estimators`, `model.random_forest.max_depth`, `model.random_forest.min_samples_leaf`.
  * **Confirmado:** 0 parámetros de `logistic_regression`.

---

### 13. Confirmación de Registro de `data.*` y `evaluation.*`

En ambos Runs se conservaron íntegramente:
* `data.test_size: 0.2`
* `data.random_state: 42`
* `evaluation.primary_metric: f1`
* `evaluation.minimum_score: 0.8`

---

### 14. Confirmación de Artifacts `model`

En ambos Runs se verificó la existencia y recuperabilidad del modelo registrado:
* Run LR: `model_uri = models:/m-dd69e881e67243bc8ec1e5382c442317`
* Run RF: `model_uri = models:/m-7b20501bc7074954833a8e5e59b8831c`
* Ambos exponen su firma (`infer_signature`), ejemplo de entrada (`head(5)`), serialización con `skops` y artefactos de reporte bajo `reports/`.

---

### 15. Deuda Técnica Identificada

1. **Gestión de tipos de hiperparámetros opcionales (`None`):**
   Actualmente si `max_depth` es `None`, en MLflow y en el manifiesto se serializa como string `'None'` o `null`. En caso de soportar en el futuro algoritmos con parámetros opcionales más complejos, convendrá estandarizar si se omiten o si se preserva `null`.
2. **Configuración por defecto restaurada:**
   `ml/params.yaml` quedó restaurado y versionado con `algorithm: logistic_regression` como baseline canónico.
