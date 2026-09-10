# Reporte Final: Evolución Multi-Algoritmo del Pipeline `ml/`

Se completó la evolución controlada del módulo `ml/` para soportar tanto `logistic_regression` como `random_forest`, manteniendo la ejecución de exactamente un modelo por corrida y sin introducir sobreingeniería.

---

### 1. Archivos Modificados y Creados

* [`ml/params.yaml`](ml/params.yaml): Evolucionado para estructurar la selección de algoritmo (`algorithm`) y aislar las secciones de hiperparámetros específicos.
* [`ml/src/ml_pipeline/modeling/factory.py`](ml/src/ml_pipeline/modeling/factory.py) *(Nuevo)*: Función factoría `build_model` que construye el `sklearn.pipeline.Pipeline` correspondiente.
* [`ml/src/ml_pipeline/modeling/train.py`](ml/src/ml_pipeline/modeling/train.py): Refactorizado para delegar la construcción en la factoría e incorporar logging del algoritmo seleccionado.
* [`ml/dvc.yaml`](ml/dvc.yaml): Agregada la dependencia de `src/ml_pipeline/modeling/factory.py` en la etapa `train`. El grafo conserva sus 5 etapas originales.
* [`ml/dvc.lock`](ml/dvc.lock): Actualizado con los nuevos hashes de dependencias y parámetros de `train`.
* [`ml/tests/unit/test_factory.py`](ml/tests/unit/test_factory.py) *(Nuevo)*: Suite unitaria que valida la construcción de ambos modelos, pasos del pipeline, hiperparámetros, manejo de errores y preprocesamiento encapsulado con NaNs.
* [`ml/tests/contract/test_model_contract.py`](ml/tests/contract/test_model_contract.py): Parametrizado para validar el contrato de inferencia y pasos específicos de preprocesamiento tanto para `logistic_regression` como para `random_forest`.
* [`ml/tests/integration/test_pipeline.py`](ml/tests/integration/test_pipeline.py): Parametrizado para validar la reproducción end-to-end de las 5 etapas DVC con ambos algoritmos.
* [`ml/README.md`](ml/README.md): Documentado el soporte de ambos clasificadores, la configuración en `params.yaml`, la restricción de un solo modelo por corrida y el rol futuro de MLflow.

---

### 2. Estructura Final de `params.model`

En `ml/params.yaml`:

```yaml
model:
  algorithm: logistic_regression
  random_state: 42

  logistic_regression:
    C: 1.0
    max_iter: 1000

  random_forest:
    n_estimators: 200
    max_depth: 10
    min_samples_leaf: 1
```

* Los parámetros comunes a nivel global de modelo (`random_state`) residen en `model.random_state`.
* Cada algoritmo mantiene sus hiperparámetros en una sección dedicada e independiente (`model.logistic_regression` y `model.random_forest`).

---

### 3. Ubicación y Responsabilidad de la Factoría / Construcción del Modelo

* **Ubicación:** `ml/src/ml_pipeline/modeling/factory.py`
* **Responsabilidad:**
  * Función principal `build_model(config: dict[str, Any]) -> Pipeline`.
  * Inspecciona `config.get("algorithm")`:
    * `"logistic_regression"`: Construye un `Pipeline` con `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()` $\rightarrow$ `LogisticRegression(C, max_iter, random_state)`.
    * `"random_forest"`: Construye un `Pipeline` con `SimpleImputer(strategy="median")` $\rightarrow$ `RandomForestClassifier(n_estimators, max_depth, min_samples_leaf, random_state)` (sin `StandardScaler`).
    * Algoritmo no soportado o `None`: Lanza un `ValueError` descriptivo (`Unsupported algorithm: ... Supported algorithms are 'logistic_regression' and 'random_forest'`).
  * Expone el alias retrocompatible `build_reference_model = build_model`.

---

### 4. Tests Añadidos o Modificados

1. **`ml/tests/unit/test_factory.py`** (7 tests unitarios nuevos):
   * `test_build_model_logistic_regression`: Comprueba la construcción del pipeline, tipos de pasos (`imputer`, `scaler`, `classifier`) e hiperparámetros.
   * `test_build_model_random_forest`: Comprueba pasos (`imputer`, `classifier`), ausencia de escalador e hiperparámetros de árboles.
   * `test_build_model_unsupported_algorithm` (3 casos): Comprueba que algoritmos inválidos o vacíos levanten `ValueError`.
   * `test_models_fit_predict_binary_with_encapsulated_preprocessing` (2 casos): Comprueba que ambos modelos puedan ajustar y predecir sobre datos con valores faltantes (`NaN`), retornando etiquetas binarias $\{0, 1\}$.
2. **`ml/tests/contract/test_model_contract.py`**:
   * Se actualizó la configuración generada en el fixture al nuevo formato estructurado.
   * Se desacopló de la lista rígida de 3 pasos y se parametrizó para validar el contrato específico de cada clasificador (`["imputer", "scaler", "classifier"]` para LR y `["imputer", "classifier"]` para RF), asegurando que ambos cumplan el contrato de inferencia.
3. **`ml/tests/integration/test_pipeline.py`**:
   * Se parametrizó sobre `["logistic_regression", "random_forest"]` para validar el recorrido completo de las 5 etapas DVC (`collect`, `validate`, `preprocess`, `train`, `evaluate`), generación de todos los artefactos y pase de elegibilidad de candidato para ambos algoritmos.

---

### 5. Resultado de `pytest`

Ejecutado con `.venv/bin/pytest`:

```text
============================= test session starts ==============================
platform linux -- Python 3.11.2, pytest-8.4.2, pluggy-1.6.0
rootdir: /home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml
configfile: pyproject.toml
testpaths: tests
plugins: hydra-core-1.3.6, anyio-4.14.2
collected 19 items                                                             

tests/contract/test_model_contract.py ..                                 [ 10%]
tests/integration/test_pipeline.py ..                                    [ 21%]
tests/unit/test_candidate.py ..                                          [ 31%]
tests/unit/test_factory.py .......                                       [ 68%]
tests/unit/test_lineage.py .                                             [ 73%]
tests/unit/test_metrics.py .                                             [ 78%]
tests/unit/test_registry.py .                                            [ 84%]
tests/unit/test_settings.py .                                            [ 89%]
tests/unit/test_validation.py ..                                         [100%]

============================== 19 passed in 8.59s ==============================
```

---

### 6. Resultado del Pipeline con Logistic Regression

Ejecución mediante `dvc repro`:
* **Etapas ejecutadas por DVC:** `train` y `evaluate` (las etapas `collect`, `validate` y `preprocess` se conservaron cacheadas sin recomputaciones innecesarias).
* **Logs:**
  ```text
  INFO | ml_pipeline.modeling.train | train | algorithm=logistic_regression | input=.../data/processed/train.csv | output=.../artifacts/work/model/model.joblib
  INFO | ml_pipeline.modeling.train | train | result=success | algorithm=logistic_regression | features=30
  INFO | ml_pipeline.modeling.evaluate | evaluate | result=success | f1=0.986111 | eligible=True
  ```
* **Métricas en `artifacts/reports/metrics.json`:**
  * `accuracy`: 0.982456
  * `precision`: 0.986111
  * `recall`: 0.986111
  * `f1`: 0.986111
* **Candidate Gate en `artifacts/reports/candidate.json`:** `{"eligible": true, "reasons": []}`

---

### 7. Resultado del Pipeline con Random Forest

Ejecución tras cambiar `model.algorithm: random_forest`:
* **Etapas ejecutadas por DVC:** DVC detectó el cambio en `params.yaml:model` y reconstruyó únicamente `train` y `evaluate`.
* **Logs:**
  ```text
  INFO | ml_pipeline.modeling.train | train | algorithm=random_forest | input=.../data/processed/train.csv | output=.../artifacts/work/model/model.joblib
  INFO | ml_pipeline.modeling.train | train | result=success | algorithm=random_forest | features=30
  INFO | ml_pipeline.modeling.evaluate | evaluate | result=success | f1=0.965517 | eligible=True
  ```
* **Métricas generadas:**
  * `accuracy`: 0.956140
  * `precision`: 0.958904
  * `recall`: 0.972222
  * `f1`: 0.965517
* **Candidate Gate:** `{"eligible": true, "reasons": []}` (supera el umbral mínimo `f1 >= 0.80`).

---

### 8. Diferencias Observadas Respecto al Baseline Anterior

* El baseline anterior con `logistic_regression` mantiene exactamente las mismas métricas (`f1: 0.986111`), validando que no hubo regresión cuantitativa ni cualitativa.
* `params.yaml` quedó restaurado al final con `algorithm: logistic_regression` como configuración por defecto.
* El pipeline serializa en `artifacts/work/model/model.joblib` únicamente el modelo seleccionado, garantizando que el downstream (`evaluate.py`) permanezca completamente agnóstico al algoritmo.

---

### 9. Deuda Técnica y Consideraciones para la Futura Fase 2 (MLflow Tracking)

1. **Parámetros registrados en MLflow (`flatten_params`):**
   La función recursiva `flatten_params` en `mlflow_tracker.py` registra actualmente todas las ramas de `params.yaml` (ej. `model.logistic_regression.C` y `model.random_forest.n_estimators`), incluso cuando sólo uno de los algoritmos fue ejecutado. En la Fase 2 convendrá filtrar para registrar únicamente los hiperparámetros del algoritmo activo (`model.algorithm`), o etiquetarlo explícitamente para evitar ruido en la UI de MLflow.
2. **Model Flavors / Metadata:**
   Ambos modelos son `sklearn.pipeline.Pipeline`, por lo que `mlflow.sklearn.log_model` funciona idénticamente para los dos. En la Fase 2 se podrán añadir tags como `algorithm: logistic_regression` o `algorithm: random_forest` en los tags del run para facilitar el filtrado y comparación en el dashboard de MLflow.
