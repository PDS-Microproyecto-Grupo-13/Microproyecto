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

## Pipeline reproducible

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

*Nota de gobernanza*: `track` **no** crea versiones en el Model Registry ni promueve modelos; esa responsabilidad corresponde exclusivamente a `register-candidate`.

## Registro de Versión Candidata en Model Registry (Fase 7)

Para registrar la versión candidata producida por el tracking run:

```bash
python -m ml_pipeline register-candidate
```

`register-candidate` es **intencionalmente externo a `dvc repro`**:
- **Precondiciones estrictas**: valida la presencia de `artifacts/reports/candidate.json` y `artifacts/reports/tracking.json`, verifica que ambos declaren `eligible: true`, comprueba la coincidencia del `dataset_fingerprint`, verifica la existencia del Run en MLflow y comprueba que el modelo PyFunc origen pueda resolverse y cargarse.
- **Idempotencia y prevención de duplicados**: antes de registrar, inspecciona las versiones registradas para el modelo (`salary-predictor`). Si ya existe una versión registrada asociada al mismo `run_id` o `model_uri`, reutiliza dicha versión en lugar de crear versiones duplicadas.
- **Tags de candidato**: registra metadatos estructurados en la versión (`candidate: "true"`, `eligible: "true"`, `algorithm: "lightgbm"`, `dataset_fingerprint`, `run_id`, `primary_metric: "mae_promedio"`, `primary_metric_value`).
- **Gobernanza de etapas**: **no** asigna alias de producción (`champion`) ni realiza transiciones de etapa. La promoción y asignación de alias corresponde a etapas posteriores en el ciclo de gobernanza.
- **Verificación obligatoria de paridad**: carga el modelo directamente desde su URI canónica de registro (`models:/salary-predictor/<version>`) y valida paridad numérica estricta contra el modelo de tracking sobre `salary_min_usd`, `salary_max_usd`, `salary_midpoint_usd`, verificando además valores finitos, predicciones positivas, ordenamiento monotónico $y_{min} \le y_{max}$, punto medio exacto y respeto a los límites operativos de entrenamiento.
- **Reporte de Registro**: persiste localmente `artifacts/reports/registration.json` con `model_name`, `version`, `run_id`, `exact_registry_uri`, `dataset_fingerprint` y métricas asociadas.

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
pytest tests/ -v
```

Las pruebas unitarias y de integración cubren:
- Ingesta multi-snapshot, deduplicación y preflight de snapshots DVC (`test_collect.py`).
- Validación de datos y consistencia de esquemas (`test_validation.py`).
- Paridad exacta del contrato de 24 features con el notebook de referencia (`test_features.py`).
- Preprocesamiento, particionado temporal y límites operativos de entrenamiento (`test_preprocess.py`).
- Calificación del modelo LightGBM y calibración de incertidumbre conjunta (`test_qualify.py`).
- Entrenamiento final sobre `train + validation` y persistencia del bundle (`test_train.py`).
- Evaluación final sobre test ciego e invariantes operacionales (`test_evaluate.py`).
- Encapsulamiento PyFunc, serialización, input example y firmas (`test_pyfunc.py`).
- Tracking en MLflow con métricas, tags, lineage y reportes (`test_tracking.py`).
- Validación de precondiciones, idempotencia, tags y paridad numérica en Model Registry (`test_registry.py`).
- Integración end-to-end de todo el ciclo de pipeline y contrato `track -> register-candidate` (`test_pipeline.py`, `test_tracking_registry.py`).

