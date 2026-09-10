# Módulo ML

Primera versión funcional del pipeline que prepara datos, entrena, evalúa, registra
experimentos y crea **versiones candidatas**. Los clasificadores de referencia
soportados (`LogisticRegression` y `RandomForestClassifier`) validan la arquitectura;
no limitan las tecnologías que pueden usarse después.

## Responsabilidades

```text
notebooks       EDA, hipótesis, visualizaciones y prototipado
DVC             datasets y pipeline reproducible
MLflow Tracking experimentos, métricas, lineage y artifacts
MLflow Registry versiones de modelos candidatos
model_provider  promoción, alias champion, despliegue y serving
```

Este módulo termina al registrar un candidato. Entrenar no significa registrar;
registrar no significa promover; promover no significa desplegar. `ml/` no importa
código, copia artifacts ni ejecuta operaciones sobre `model_provider/`.

## Instalación aislada

Desde la raíz del monorepo:

```bash
cd ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
cp .env.example .env
```

`requirements.lock.txt` instala también el paquete local en modo editable, por lo
que la CLI queda disponible sin configurar `PYTHONPATH`.

## Pipeline reproducible (Fase 5: Ingesta, Validación, Preprocesamiento, Calificación, Entrenamiento y Evaluación Final sobre TEST)

Actualmente el pipeline DVC gobierna de forma reproducible las 6 etapas completas del ciclo de modelado salarial:

```text
collect -> validate -> preprocess -> qualify -> train -> evaluate
```

```bash
python -m ml_pipeline collect
python -m ml_pipeline validate
python -m ml_pipeline preprocess
python -m ml_pipeline qualify
python -m ml_pipeline train
python -m ml_pipeline evaluate
# O mediante DVC:
dvc repro
```

### Fuente de datos y regla de entrada
- **Fuente RAW canónica**: `data/raw/foorilla/jobs_*.csv`.
- **Inclusión automática**: Todos los archivos `jobs_*.csv` materializados en la carpeta forman automáticamente el dataset de entrada. No se requiere mantener listas de archivos en `params.yaml`.
- **Preflight de integridad**: Si existe un archivo de puntero `jobs_X.csv.dvc` pero el archivo físico `jobs_X.csv` no se encuentra materializado en el disco local, `collect` falla inmediatamente para evitar procesar un dataset parcial no deliberado.
- **Descarga de snapshots**: Si un snapshot falta en el entorno local, debe materializarse explícitamente mediante DVC antes de ejecutar la ingesta:
  ```bash
  dvc pull data/raw/foorilla/<snapshot>.csv.dvc
  ```

### Salidas y Trazabilidad
- **Interim**: `data/interim/foorilla_consolidated.parquet` (dataset consolidado, deduplicado y con filtros de coherencia salarial e inliers).
- **Validated**: `data/validated/dataset.parquet` (dataset que supera todas las invariantes de calidad y esquema).
- **Processed**: `data/processed/train.parquet`, `validation.parquet`, `test.parquet` (particiones con ordenamiento temporal estricto 70/15/15 sobre universo `target_scope: reportado`, conteniendo metadatos `id`, `published`, las 24 features y targets salariales `y_min_usd`, `y_max_usd`).
- **Límites de entrenamiento**: `artifacts/reports/train_limits.json` calculado exclusivamente con la partición de train (cuantiles 0.001 y 0.999).
- **Calificación y Calibración de Incertidumbre**:
  - `artifacts/reports/qualification.json`: métricas del baseline (DummyRegressor), validación LightGBM, retrotest temporal en train (80/20) y verificación de criterios (`improvement_vs_baseline >= 0.10`, `abs(temporal_gap) <= 0.25`).
  - `artifacts/reports/uncertainty_calibration.json`: margen de incertidumbre conjunto sobre validación (`quantile(joint_error, 0.80, method="higher")`).
- **Modelo Definitivo y Reporte de Entrenamiento**:
  - `artifacts/work/model/model.joblib`: bundle serializado con modelos duales (`pipeline_min`, `pipeline_max`), límites operativos, margen de incertidumbre calibrado en Fase 3 y contrato de 24 features entrenado sobre 46,208 filas (`train + validation`).
  - `artifacts/reports/training.json`: reporte determinista con metadatos del ajuste definitivo sin marcas de tiempo dinámicas.
- **Evaluación Final sobre TEST Ciego y Auditorías Diagnósticas**:
  - `artifacts/reports/metrics.json`: métricas de regresión sobre `test.parquet` (8,155 filas), margen y cobertura de incertidumbre (78.93%), invariantes de calidad y diagnósticos.
  - `artifacts/reports/candidate.json`: preserva la calificación aprobada (`eligible: true`), metadatos de entrenamiento y resultado final de test.
  - `artifacts/reports/experiment_manifest.json`: manifiesto integral con lineage Git/DVC, hiperparámetros efectivos y métricas de test.
  - `artifacts/reports/audit_segments.json`: desglose de desempeño por segmentos (`country`, `experience_level`, `years_group`, `work_mode`).
  - `artifacts/reports/audit_novelty.json`: evaluación de error en categorías conocidas vs no observadas (`title`, `company`, `country`).
  - `artifacts/reports/audit_sensitivity.json`: auditorías de sensibilidad (perfil sintético no observado, monotonicidad en progresión de experiencia, escenarios geográficos y perfiles de habilidades).
  - `artifacts/reports/feature_importance.json`: ranking de importancia promedio entre los dos estimadores LightGBM.
- **Manifiestos y reportes de datos**:
  - `artifacts/reports/data_manifest.json`: huella SHA-256 por snapshot y `dataset_fingerprint`.
  - `artifacts/reports/validation.json`: conteos por `target_source` y estadísticas de validación.
  - `artifacts/reports/preprocess.json`: resumen de splits, rangos de fechas, contrato de 24 features y metadata de reproducibilidad.

*Nota de gobernanza*: La partición `test.parquet` fue evaluada por primera y única vez en `evaluate`, sin retroalimentación, sin recalibración y sin ajuste de hiperparámetros. Las fases subsiguientes abordarán el registro del candidato y el tracking en MLflow.

## Tracking y empaquetado PyFunc (Fase 6)

Configure `.env` para apuntar a un servidor MLflow:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=salary-prediction
MLFLOW_MODEL_NAME=salary-predictor
ML_REQUIRE_CLEAN_GIT=false
```

O use un backend SQLite local cambiando solo `MLFLOW_TRACKING_URI` (por ejemplo, `sqlite:///mlflow.db`). Luego ejecute:

```bash
python -m ml_pipeline track
```

`track` es **intencionalmente externo a `dvc repro`**: crea un MLflow Run explícito sin reentrenar ni reevaluar. Registra:
- **Wrapper PyFunc**: encapsula el modelo como `SalaryPredictorModel(mlflow.pyfunc.PythonModel)`, recibiendo datos crudos (incluyendo `regions`), ejecutando `prepare_features()`, inferencia dual con límites operativos y produciendo exactamente `[salary_min_usd, salary_max_usd, salary_midpoint_usd]`.
- **Signature e Input Example**: mapea las columnas crudas de entrada hacia las 3 columnas salariales nombradas.
- **Parámetros efectivos**: algoritmo (`lightgbm`), targets (`y_min_usd, y_max_usd`), conteo de features (24), hiperparámetros de LightGBM y umbrales de calificación.
- **Métricas consolidadas**: consume directamente las métricas calculadas en validación y test ciego en la Fase 5.
- **Lineage y tags**: commit/dirty Git, `dataset_fingerprint`, revisión DVC, hash de parámetros y conteos de filas.
- **Artifacts**: los 10 reportes de auditoría y calificación (`qualification.json`, `uncertainty_calibration.json`, `training.json`, `metrics.json`, `candidate.json`, `experiment_manifest.json`, `audit_segments.json`, `audit_novelty.json`, `audit_sensitivity.json`, `feature_importance.json`) bajo `reports/` y el modelo PyFunc bajo `model`.
- **Reporte de Tracking**: genera `artifacts/reports/tracking.json` con `run_id`, `model_uri`, `model_id` y metadatos para la Fase 7.

*Nota de gobernanza*: `track` **no** crea versiones en el Model Registry ni promueve modelos; esa responsabilidad corresponde exclusivamente a la Fase 7 (`register-candidate`).

## Lineage y outputs

`tracking/lineage.py` centraliza commit/estado Git, revisión de `dvc.yaml`, hash de
parámetros y fingerprint del dataset. Tracking añade además el hash del `dvc.lock`
ya finalizado. Así se evita incluir en un output un hash del lock que ese mismo
output modificaría. Los valores no disponibles se conservan como `null` en el
manifest y como `unknown` en tags MLflow.

Los outputs reproducibles son datos raw/validated/processed, el working artifact
`artifacts/work/model/model.joblib` y reports de validación, métricas, elegibilidad
y `artifacts/reports/experiment_manifest.json` (que resume el contexto completo del experimento:
algoritmo, hiperparámetros efectivos del modelo, métrica primaria y estado candidato).
El working artifact no es la copia canónica desplegable: esa función corresponde al
artifact MLflow registrado.

## Notebooks y evolución

`notebooks/` queda reservado para EDA y prototipos. Cuando una decisión se
estabiliza, solo la lógica necesaria para reproducir entrenamiento y evaluación se
traslada a código Python y al pipeline DVC; no se convierten celdas mecánicamente.

Para sustituir sklearn, `modeling/train.py` y el flavor usado dentro de
`tracking/mlflow_tracker.py` son los puntos tecnológicos a adaptar. Las fronteras
de datos, evaluación/candidatura, lineage, tracking y registro permanecen. Esto
permite incorporar después XGBoost, TensorFlow, PyTorch, Transformers u otros sin
convertir hoy el proyecto en un framework genérico.

## Tests

```bash
pytest
```

Las pruebas unitarias cubren validación, métricas, factoría de modelos (`logistic_regression` y `random_forest`), gate, configuración, lineage, normalización de parámetros y
rechazo de registro. La integración recorre las cinco etapas DVC en un directorio
temporal para ambos algoritmos y cubre el contrato automatizado `track -> register-candidate` sobre un backend SQLite temporal aislado. El contrato verifica el esquema de entrada, preprocessing encapsulado y
predicciones binarias. Ninguna prueba requiere MLflow remoto ni `model_provider/`.
