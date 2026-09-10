# FASE 6 — MLflow PyFunc + Tracking del Modelo Salarial

## 1. Resumen Ejecutivo

En la **Fase 6**, se ha implementado el empaquetado del modelo salarial definitivo mediante un wrapper desacoplado **MLflow PyFunc** (`SalaryPredictorModel`) y se ha ejecutado el registro completo de un **MLflow Run** sobre el servidor activo de seguimiento (`http://localhost:5000`):

$$\text{artifacts/work/model/model.joblib} \longrightarrow \text{SalaryPredictorModel (PyFunc)} \longrightarrow \text{MLflow Run (Tracking)}$$

### Principios y Reglas de Gobernanza Cumplidas:
1. **Inviolabilidad del Modelo y Evaluaciones Previas**:
   - Se utilizó **exactamente** el modelo `artifacts/work/model/model.joblib` entrenado en la Fase 4.
   - No se reentrenó, no se recalibró y no se volvieron a evaluar particiones de datos.
   - Se consumieron directamente las métricas e invariantes generadas en la Fase 5.
2. **Paridad Numérica Estricta (Local vs PyFunc)**:
   - Las predicciones obtenidas localmente con `model.joblib` y las producidas por el wrapper PyFunc (tanto en memoria como recargado desde el Model Store de MLflow) son numéricamente idénticas ($\Delta < 10^{-5}$ USD en todos los cuantiles y promedios).
3. **Tracking Desacoplado de DVC**:
   - El pipeline DVC mantiene sus 6 etapas reproducibles (`collect -> validate -> preprocess -> qualify -> train -> evaluate`).
   - El comando `track` se ejecuta como un efecto externo explícito, sin contaminar el grafo DVC ni alterar `dvc.lock`.
4. **Delimitación de Alcance**:
   - No se creó ninguna versión en el **Model Registry** (responsabilidad reservada exclusivamente para la Fase 7).
   - No se modificaron contratos ni código en `backend/` ni `model_provider/`.

---

## 2. Wrapper PyFunc (`SalaryPredictorModel`)

El wrapper fue implementado en [`ml/src/ml_pipeline/modeling/pyfunc.py`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml/src/ml_pipeline/modeling/pyfunc.py) heredando de `mlflow.pyfunc.PythonModel`.

### 2.1. Arquitectura y Flujo de Inferencia
El modelo recibe un DataFrame con variables crudas y ejecuta internamente el flujo canónico:

$$\text{raw DataFrame} \xrightarrow{\text{prepare\_features()}} X_{24} \xrightarrow{\text{bundle.joblib}} [\text{log1p}(\hat{y}_{\min}), \text{log1p}(\hat{y}_{\max})] \xrightarrow{\text{expm1}} \text{clip}[\text{floor}, \text{ceiling}] \xrightarrow{\text{ordenamiento}} [\hat{y}_{\min}, \hat{y}_{\max}, \hat{y}_{\text{midpoint}}]$$

1. **Reutilización de Lógica**: No se duplicó código; el wrapper reutiliza [`prepare_features()`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml/src/ml_pipeline/features.py) y [`predict_range()`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml/src/ml_pipeline/modeling/train.py).
2. **Carga de Contexto**: En `load_context(context)`, el artefacto `model_bundle` es deserializado vía `joblib.load()` garantizando que el modelo sea completamente autónomo y serializable.
3. **Cálculo de Punto Medio**: El punto medio se calcula estrictamente como:
   $$\text{salary\_midpoint\_usd} = \frac{\text{salary\_min\_usd} + \text{salary\_max\_usd}}{2}$$
4. **Soporte de Entrada**: Admite tanto inferencias unitarias (1 fila) como inferencias en lote (múltiples filas), preservando el índice original del DataFrame de entrada.

---

## 3. Contrato de Entrada y Salida (Signature e Input Example)

### 3.1. Contrato de Entrada (Variables Crudas)
El wrapper exige exactamente las variables necesarias para alimentar `prepare_features()`:

| Columna | Tipo | Requerido | Descripción |
| :--- | :---: | :---: | :--- |
| `title` | string | Sí | Título del puesto de trabajo |
| `company` | string | Sí | Nombre de la empresa ofertante |
| `company_is_agency` | boolean | Sí | Flag de agencia o reclutador |
| `countries` | string | Sí | País o lista de países separados por pipe (`\|`) |
| `regions` | string | **Sí** | Región continental o geográfica (requerida por contrato) |
| `experience_level` | string | Sí | Nivel de experiencia (`junior`, `mid`, `senior`, `lead`) |
| `experience_years` | double | Sí | Años de experiencia declarados (acotados a $[0, 50]$) |
| `has_remote` | boolean | Sí | Flag de disponibilidad de trabajo remoto |
| `work_mode` | double | Sí | Código numérico de modalidad laboral |
| `tags` | string | Sí | Lista de tecnologías/habilidades separadas por pipe |
| `published` | string | Sí | Timestamp de publicación en formato ISO-8601 |

> [!WARNING]
> **Gap Vigente de `regions`**: `prepare_features()` eleva `KeyError` si la columna `regions` está ausente. El backend todavía no envía esta columna en su contrato actual. Este desacoplamiento es conocido y **no fue modificado** en esta fase para preservar la integridad de la frontera entre módulos.

### 3.2. Contrato de Salida Exacto
El PyFunc produce un DataFrame con exactamente 3 columnas continuas en dólares estadounidenses ($USD$):

| Columna | Tipo | Descripción |
| :--- | :---: | :--- |
| `salary_min_usd` | double | Límite inferior salarial predicho y acotado por `floor` |
| `salary_max_usd` | double | Límite superior salarial predicho y acotado por `ceiling` |
| `salary_midpoint_usd` | double | Punto medio exacto $(\text{min} + \text{max}) / 2$ |

### 3.3. Input Example Registrado
Se registró un ejemplo representativo de dos observaciones en MLflow:
```python
[
    {
        "title": "Senior Machine Learning Engineer",
        "company": "Tech Innovators Inc",
        "company_is_agency": False,
        "countries": "United States",
        "regions": "Americas",
        "experience_level": "senior",
        "experience_years": 6.0,
        "has_remote": True,
        "work_mode": 2.0,
        "tags": "Python|Machine Learning|Docker|Kubernetes|AWS",
        "published": "2026-08-16T10:00:00Z"
    },
    {
        "title": "Backend Software Developer",
        "company": "Digital Services Corp",
        "company_is_agency": False,
        "countries": "Germany",
        "regions": "Europe",
        "experience_level": "mid",
        "experience_years": 3.0,
        "has_remote": False,
        "work_mode": 1.0,
        "tags": "Python|SQL|FastAPI",
        "published": "2026-08-20T14:30:00Z"
    }
]
```

---

## 4. Verificación de Paridad Obligatoria (Local vs PyFunc)

La paridad fue evaluada con la función de auditoría `verify_local_pyfunc_parity` comparando la inferencia de `model.joblib` contra el modelo PyFunc rehidratado directamente desde MLflow:

| Invariante Evaluada | Criterio de Aceptación | Resultado Local | Resultado PyFunc | Paridad |
| :--- | :--- | :---: | :---: | :---: |
| **Mismo Mínimo** | $\vert \hat{y}_{\min}^{\text{loc}} - \hat{y}_{\min}^{\text{pyfunc}} \vert < 10^{-5}$ | $29,762.96 | $29,762.96 | ✅ **Idéntico** |
| **Mismo Máximo** | $\vert \hat{y}_{\max}^{\text{loc}} - \hat{y}_{\max}^{\text{pyfunc}} \vert < 10^{-5}$ | $57,257.03 | $57,257.03 | ✅ **Idéntico** |
| **Mismo Midpoint** | $\vert \hat{y}_{\text{mid}}^{\text{loc}} - \hat{y}_{\text{mid}}^{\text{pyfunc}} \vert < 10^{-5}$ | $43,509.99 | $43,509.99 | ✅ **Idéntico** |
| **Consistencia Midpoint** | $\text{mid} == (\text{min} + \text{max}) / 2$ | Exacto | Exacto | ✅ **Cumple** |
| **Valores Finitos** | Sin `NaN` ni `Inf` | 100% | 100% | ✅ **Cumple** |
| **Valores Positivos** | Predicciones $> 0$ | 100% | 100% | ✅ **Cumple** |
| **Rangos Ordenados** | $\text{min} \le \text{max}$ | 100% | 100% | ✅ **Cumple** |
| **Límites Operativos** | $[\$10,935.57, \$720,000.00]$ | 100% dentro | 100% dentro | ✅ **Cumple** |

---

## 5. Contenido Registrado en MLflow Tracking

Se ejecutó exitosamente el comando de tracking sobre el servidor en ejecución:
```bash
python -m ml_pipeline track
```

- **Tracking URI**: `http://localhost:5000`
- **Experimento**: `salary-prediction` (Experiment ID: `2`)
- **Run ID**: `db1fd1c17c8e499d976c8446fe17e3ec`
- **Model URI**: `models:/m-3cc347ba70af4dc782ba5aa2a9b1c5b8`
- **Model ID**: `m-3cc347ba70af4dc782ba5aa2a9b1c5b8`
- **Artifact Location**: `mlflow-artifacts:/2/db1fd1c17c8e499d976c8446fe17e3ec/artifacts`

### 5.1. Parámetros Registrados (24 parámetros)
- `model.algorithm`: `lightgbm`
- `model.random_state`: `42`
- `model.feature_count`: `24`
- `model.targets`: `y_min_usd, y_max_usd`
- `model.lightgbm.objective`: `regression_l1`
- `model.lightgbm.n_estimators`: `700`
- `model.lightgbm.num_leaves`: `95`
- `model.lightgbm.learning_rate`: `0.06`
- `model.lightgbm.min_child_samples`: `20`
- `model.lightgbm.subsample`: `0.8`
- `model.lightgbm.subsample_freq`: `1`
- `model.lightgbm.colsample_bytree`: `1.0`
- `model.lightgbm.reg_lambda`: `3.0`
- `model.lightgbm.verbosity`: `-1`
- `model.lightgbm.n_jobs`: `-1`
- `qualification.min_improvement_vs_baseline`: `0.10`
- `qualification.max_temporal_gap`: `0.25`
- `qualification.backtest_train_ratio`: `0.80`
- `qualification.uncertainty_quantile`: `0.80`
- `data.train_ratio`: `0.70`
- `data.validation_ratio`: `0.15`
- `data.test_ratio`: `0.15`
- `data.target_scope`: `reportado`
- `evaluation.primary_metric`: `mae_promedio`

### 5.2. Métricas Consolidadas (24 métricas)
Consumidas directamente de los reportes deterministas sin reevaluar:
- **Test Ciego (Fase 5)**:
  - `test_mae_promedio`: `27081.221469`
  - `test_mae_min`: `22487.039193`
  - `test_mae_max`: `31675.403746`
  - `test_rmse_min`: `40174.003587`
  - `test_rmse_max`: `54490.051851`
  - `test_mape_min`: `0.265057`
  - `test_mape_max`: `0.238765`
  - `test_r2_min`: `0.572696`
  - `test_r2_max`: `0.628212`
  - `test_mae_amplitud`: `21127.059368`
  - `test_cobertura_intervalo`: `0.113918`
  - `test_incoherencia_raw`: `0.020478`
  - `test_prediccion_no_positiva`: `0.0`
  - `uncertainty_margin`: `50927.2876`
  - `uncertainty_nominal_coverage`: `0.80`
  - `uncertainty_test_coverage`: `0.789332`
- **Validación y Calificación (Fase 3)**:
  - `val_mae_promedio`: `26140.8116`
  - `baseline_mae`: `56034.6145`
  - `improvement_vs_baseline`: `0.533488`
  - `backtest_mae`: `25244.334`
  - `temporal_gap`: `0.035512`
- **Volumetría de Observaciones**:
  - `train_rows`: `38054.0`
  - `validation_rows`: `8154.0`
  - `final_training_rows`: `46208.0`
  - `test_rows`: `8155.0`

### 5.3. Lineage y Tags de Gobernanza
- `algorithm`: `lightgbm`
- `candidate_eligible`: `True`
- `dataset_fingerprint`: `541f47bf1bb49a6a0e5947d49dade61c444e74cd5350cfbb68a185506cd957ed`
- `primary_metric`: `mae_promedio`
- `train_rows`: `38054`
- `validation_rows`: `8154`
- `final_training_rows`: `46208`
- `test_rows`: `8155`

### 5.4. Artefactos Persistidos en el Run
Bajo el directorio lógico `reports/`:
1. `qualification.json`
2. `uncertainty_calibration.json`
3. `training.json`
4. `metrics.json`
5. `candidate.json`
6. `experiment_manifest.json`
7. `audit_segments.json`
8. `audit_novelty.json`
9. `audit_sensitivity.json`
10. `feature_importance.json`

Bajo el artefacto lógico `model`:
- Empaquetado PyFunc completo con `MLmodel`, código fuente en `code/ml_pipeline` y el bundle serializado `model.joblib`.

### 5.5. Reporte de Tracking Local (`artifacts/reports/tracking.json`)
```json
{
  "algorithm": "lightgbm",
  "artifact_location": "mlflow-artifacts:/2/db1fd1c17c8e499d976c8446fe17e3ec/artifacts",
  "dataset_fingerprint": "541f47bf1bb49a6a0e5947d49dade61c444e74cd5350cfbb68a185506cd957ed",
  "eligible": true,
  "experiment_id": "2",
  "experiment_name": "salary-prediction",
  "model_id": "m-3cc347ba70af4dc782ba5aa2a9b1c5b8",
  "model_uri": "models:/m-3cc347ba70af4dc782ba5aa2a9b1c5b8",
  "primary_metric": "mae_promedio",
  "primary_metric_value": 27081.221469,
  "run_id": "db1fd1c17c8e499d976c8446fe17e3ec"
}
```

---

## 6. Verificación de Aislamiento de Dependencias y Desacoplamiento

1. **Inclusión de Código (`code_paths`)**:
   - Se incluyó explícitamente `src/ml_pipeline` en `code_paths`.
   - MLflow empaqueta los módulos de pipeline dentro del modelo en `code/ml_pipeline`, permitiendo que `from ml_pipeline.features import prepare_features` resuelva correctamente en cualquier entorno donde se descargue el modelo.
2. **Requisitos de Entorno (`pip_requirements`)**:
   - `mlflow==3.15.1`
   - `pandas>=2.0.0`
   - `numpy>=1.24.0`
   - `scikit-learn>=1.4.0`
   - `lightgbm>=4.0.0`
   - `joblib>=1.3.0`
3. **Ausencia de Dependencia de CWD**:
   - Los artefactos del modelo se referencian mediante rutas relativas al contexto (`context.artifacts["model_bundle"]`).

---

## 7. Pruebas Automatizadas

Se añadieron pruebas unitarias exhaustivas y se verificó la suite completa:

```bash
ml/.venv/bin/pytest tests/ -v
```

- **Resultado**: `73 passed, 3 skipped in 29.13s`.
- **Pruebas añadidas / adaptadas**:
  - `tests/unit/test_pyfunc.py`:
    - `test_pyfunc_input_example_and_signature`: valida contrato de entrada y salida nombrada.
    - `test_pyfunc_predict_single_and_batch_rows`: valida inferencia de 1 fila y batch.
    - `test_pyfunc_fails_when_regions_missing`: confirma que la ausencia de `regions` eleva `KeyError`.
    - `test_pyfunc_parity_verification_helper`: audita las 8 invariantes de paridad.
    - `test_pyfunc_save_load_and_mlflow_roundtrip`: ciclo completo de log, recarga y paridad.
  - `tests/unit/test_tracking.py`:
    - `test_effective_tracking_params_for_lightgbm`: valida extracción de parámetros LightGBM.
    - `test_track_fails_if_model_or_metrics_missing`: falla controlada si faltan artefactos.
    - `test_track_creates_mlflow_run_with_pyfunc_params_metrics_and_artifacts`: valida ejecución end-to-end de `track`, ausencia de llamadas a `train`/`evaluate` y verificación de 0 versiones en Model Registry.

---

## 8. Verificación de Idempotencia DVC

El grafo de DVC permanece inalterado:
```bash
$ PATH="$PWD/.venv/bin:$PATH" dvc status
Data and pipelines are up to date.

$ PATH="$PWD/.venv/bin:$PATH" dvc repro
Stage 'collect' didn't change, skipping
Stage 'validate' didn't change, skipping
Stage 'preprocess' didn't change, skipping
Stage 'qualify' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Data and pipelines are up to date.
```

---

## 9. Estado Final de la Fase 6

El empaquetado PyFunc y el tracking del modelo salarial han sido completados con total fidelidad funcional, paridad numérica rigurosa y registro verificable en MLflow. Quedan listos todos los metadatos e identificadores en `tracking.json` para abordar la Fase 7 (Model Registry).

```text
PHASE_6_STATUS=READY_FOR_PHASE_7
```
