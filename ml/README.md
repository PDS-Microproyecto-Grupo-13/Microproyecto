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

```bash
python -m ml_pipeline --help
dvc repro
```

DVC ejecuta únicamente:

```text
collect -> validate -> preprocess -> train -> evaluate
```

También se puede ejecutar cada etapa explícitamente:

```bash
python -m ml_pipeline collect
python -m ml_pipeline validate
python -m ml_pipeline preprocess
python -m ml_pipeline train
python -m ml_pipeline evaluate
```

La semilla, partición, algoritmo y gate de evaluación viven en `params.yaml`.
Actualmente se soportan dos algoritmos de clasificación, seleccionables mediante `model.algorithm`:
- `logistic_regression` (con `SimpleImputer` y `StandardScaler`)
- `random_forest` (con `SimpleImputer`)

Cada ejecución entrena y evalúa exactamente **un único algoritmo** seleccionado en `params.yaml`. Todavía no existe selección automática de modelos ni comparación dentro de una misma ejecución; MLflow Tracking se utilizará posteriormente para comparar los distintos runs experimentales.

El split ocurre antes de entrenar. Las transformaciones necesarias para inferencia forman parte del `sklearn.pipeline.Pipeline` serializado (`artifacts/work/model/model.joblib`), evitando que el consumidor tenga que reproducir transformaciones ocultas.

## Tracking y registro

Configure `.env` para apuntar a un servidor compartido:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=toy-classification
MLFLOW_MODEL_NAME=toy-classifier
ML_REQUIRE_CLEAN_GIT=false
```

O use un backend local cambiando solo `MLFLOW_TRACKING_URI`, por ejemplo
`sqlite:///mlflow.db`. Luego ejecute:

```bash
python -m ml_pipeline track
python -m ml_pipeline register-candidate
```

`track` es intencionalmente externo a `dvc repro`: crea un MLflow Run explícito
(con `run_name` igual al algoritmo utilizado). Registra:
- el tag y parámetro `algorithm` utilizado;
- únicamente los hiperparámetros efectivos del algoritmo activo (provenientes de una única fuente de verdad `effective_model_params`, filtrando ramas de algoritmos inactivos);
- parámetros reproducibles de datos y evaluación (`data.*`, `evaluation.*`);
- valores opcionales `None` normalizados determinísticamente como `"null"` en MLflow (conservados como `null` en manifests JSON);
- métricas generadas en la evaluación (`accuracy`, `precision`, `recall`, `f1`);
- lineage y tags de procedencia;
- reports generados (`validation.json`, `metrics.json`, `candidate.json`, `experiment_manifest.json`);
- signature, ejemplo de entrada y el modelo bajo el artifact `model`.

Múltiples experimentos se comparan mediante distintos Runs en MLflow, sin entrenar automáticamente
todos los modelos en una sola ejecución.

`register-candidate` exige elegibilidad, un `run_id` y dicho artifact antes de crear una versión en
`MLFLOW_MODEL_NAME`. No asigna aliases ni stages, no promueve, no despliega y no sirve inferencias. En
entornos controlados, `ML_REQUIRE_CLEAN_GIT=true` exige un árbol de trabajo limpio.

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
