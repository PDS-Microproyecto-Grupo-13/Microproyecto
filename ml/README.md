# Módulo ML — Pipeline de Modelado Salarial y Registro

Módulo central de Machine Learning para **SalaryPredict v1.0**. Implementa el ciclo de vida cuantitativo reproducible desde la ingesta de vacantes crudas hasta el registro formal de modelos candidatos en MLflow.

---

## 1. Responsabilidad

El módulo `ml/` cubre de forma autónoma:
```text
Datos Crudos ──> Validación ──> Preprocesamiento ──> Calificación ──> Entrenamiento ──> Evaluación ──> Analytics
                                                                                                           └──> publicación explícita
Evaluación ──> Tracking ──> Registro de Candidato
```

- **Frontera de responsabilidad**: El flujo de este módulo **termina estrictamente en `register-candidate`**.
- **Fuera de alcance de `ml/`**:
  - **NO** promueve modelos al alias `champion`.
  - **NO** despliega ni reinicia servicios de inferencia.
  - **NO** atiende peticiones de inferencia en tiempo real.
  - Esas responsabilidades pertenecen exclusivamente a `model_provider/`.

---

## 2. Preparación del Entorno

Desde la carpeta `ml/`:

```bash
# 1. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias fijadas y paquete ml_pipeline en modo editable
pip install --upgrade pip
pip install -r requirements.lock.txt

# 3. Configurar variables de entorno
cp .env.example .env
```

### Variables de Entorno Vigentes (`ml/.env`)

| Variable | Valor por Defecto | Descripción |
| :--- | :--- | :--- |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | Servidor central de MLflow |
| `MLFLOW_EXPERIMENT_NAME` | `salary-prediction` | Experimento de seguimiento de corridas |
| `MLFLOW_MODEL_NAME` | `salary_predict_model` | Nombre canónico del modelo en el Registry |
| `ML_REQUIRE_CLEAN_GIT` | `false` | Exigir árbol Git limpio antes de registrar |
| `ANALYTICS_PUBLISH_URL` | sin default operativo | Endpoint administrativo `POST /api/v1/analytics/snapshots` |
| `ANALYTICS_PUBLISH_TOKEN` | sin default | Bearer token; obligatorio solo para `publish-analytics` y nunca versionado |
| `ANALYTICS_PUBLISH_TIMEOUT_SECONDS` | `10` | Timeout HTTP de publicación |

---

## 3. Fuente de Datos y Remoto DVC

- **Ruta canónica**: `data/raw/foorilla/jobs_*.csv`.
- Los archivos CSV de corte son la materia prima inmutable de entrada y los snapshots son administrados y versionados mediante DVC.
- **Agnosticismo de backend**: El pipeline ML no necesita conocer qué backend de almacenamiento utiliza DVC.
- **Obtención normal de datos**: El usuario obtiene los datos normalmente mediante:
  ```bash
  dvc pull
  ```
- **Credenciales y autenticación**: Las credenciales dependen del remoto elegido:
  - En un servidor autogestionado Nginx/SSH, los consumidores pueden hacer `dvc pull` por HTTP sin credenciales, mientras que los escritores utilizan SSH autenticado para `dvc push`.
  - En un remoto administrado como Google Drive, la autenticación se gestiona vía OAuth según el proveedor.
- **Regla de preflight de integridad**: Todos los snapshots referenciados por punteros `.dvc` deben encontrarse materializados físicamente en la carpeta. Si falta algún archivo CSV, la etapa `collect` aborta automáticamente para evitar procesar un dataset parcial no planificado.
- **Configuración del remoto**: La configuración concreta del remoto DVC se encuentra en `docs/`:
  - [Configuración de DVC con Google Drive](docs/CONFIGURACION_DVC_GOOGLE_DRIVE.md)
  - [Servidor DVC autogestionado con Docker, Nginx y SSH](docs/DVC_NGINX_SERVER.md)
  - [Configuración y uso de clientes DVC (HTTP / SSH)](docs/DVC_NGINX_CONFIG.md)
- Para la guía general de despliegue y bootstrap, consulte [`../DEPLOYMENT.md`](../DEPLOYMENT.md).

---

## 4. Pipeline Reproducible DVC

El ciclo de preparación, modelado, evaluación y resumen analítico está gobernado por DVC a través de 7 etapas deterministas:

```text
collect ──> validate ──> preprocess ──> qualify ──> train ──> evaluate ──> analytics
```

Para reproducir el pipeline completo:
```bash
dvc repro
```

O ejecutando individualmente cada etapa mediante la CLI de `ml_pipeline`:
```bash
python -m ml_pipeline collect       # Consolida snapshots en data/interim/
python -m ml_pipeline validate      # Comprueba esquema y genera data/validated/
python -m ml_pipeline preprocess    # Split temporal 70/15/15 y límites de train
python -m ml_pipeline qualify       # Evaluación vs baseline Dummy y calibración de incertidumbre
python -m ml_pipeline train         # Ajuste dual LightGBM sobre train+val (46,208 filas)
python -m ml_pipeline evaluate      # Evaluación sobre test ciego (8,155 filas) y reportes
python -m ml_pipeline analytics     # Genera dashboard_summary.json sin red ni MLflow
```

> **Aviso**: `dvc repro` ejecuta **únicamente** estas 7 etapas reproducibles locales. No realiza operaciones con efectos remotos en MLflow ni publica analytics por HTTP.

### Home Analytics

La etapa reproducible puede ejecutarse o reconstruirse de forma aislada:

```bash
dvc repro analytics
```

Produce `artifacts/reports/dashboard_summary.json` schema 1.0 a partir de artifacts locales y de la población modelable completa. Salarios, seniority, work mode y tecnologías describen las filas con `target_source == data.target_scope`. Las tecnologías son **menciones detectadas por el extractor canónico** `skill_*`, no una ontología exhaustiva ni feature importance.

La publicación es una operación administrativa separada y fuera de DVC:

```bash
python -m ml_pipeline publish-analytics
```

El comando valida un artifact existente y lo envía al backend configurado. No calcula estadísticas, no ejecuta DVC y no consulta MLflow. No es necesario publicar después de cada `dvc repro`; un backend nuevo o vacío requiere una publicación inicial. Si el artifact ya existe, puede publicarse sin recalcular el pipeline.

---

## 5. Operaciones externas (Fuera de DVC)

Las etapas de logging y registro se ejecutan fuera de DVC porque interactúan con el servidor de MLflow y generan registros externos:

### 1. Registrar experimento en MLflow Tracking
```bash
python -m ml_pipeline track
```
- Lee los artefactos locales (`model.joblib` y los 10 reportes de auditoría en `artifacts/reports/`).
- Registra el Run en MLflow con hiperparámetros efectivos, métricas consolidadas y tags de linaje Git/DVC.
- Empaqueta el modelo como wrapper PyFunc (`SalaryPredictorModel`), asociando dependencias y código fuente de `src/ml_pipeline`.
- Genera el reporte local `artifacts/reports/tracking.json`.

### 2. Registrar versión candidata en Model Registry
```bash
python -m ml_pipeline register-candidate
```
- Valida las precondiciones de elegibilidad (`eligible: true` en `candidate.json` y `tracking.json`).
- Registra la versión en el MLflow Model Registry bajo el modelo `salary_predict_model`.
- Asigna los metadatos `candidate="true"` y `eligible="true"`.
- Valida la paridad numérica estricta entre el modelo de tracking y la versión registrada.
- Genera el reporte local `artifacts/reports/registration.json`.

> **Gobernanza**: `register-candidate` **no** asigna alias de producción (`champion`). La promoción corresponde a los operadores en `model_provider/`.

### 3. Publicar Home Analytics

```bash
python -m ml_pipeline publish-analytics
```

- Lee y valida `artifacts/reports/dashboard_summary.json`.
- Envía el JSON sin wrapper con bearer token.
- Acepta respuestas `created`, `replaced` o `unchanged` del backend.
- Las métricas `model` son métricas de evaluación asociadas al snapshot analítico; no prueban que el modelo evaluado sea el `champion` ni el modelo actualmente servido.

---

## 6. Arquitectura del Modelo Salarial

- **Ensamble Dual LightGBM**: Dos estimadores independientes optimizados con función de pérdida L1 (`objective='regression_l1'`), uno para salario mínimo (`y_min_usd`) y otro para salario máximo (`y_max_usd`).
- **Particionamiento Temporal**: División cronológica estricta sin mezcla futura (70% train, 15% validation, 15% test ciego).
- **Transformación de Características**:
  - `prepare_features()` estructura al vuelo las 24 variables canónicas a partir de los datos crudos del empleo.
  - El pipeline interno de Scikit-Learn aplica imputación most_frequent en variables categóricas, imputación mediana en numéricas, TargetEncoder categórico y LightGBM..
  - Postprocesamiento acotado a los límites operacionales aprendidos en train (`train_limits.json`).
- **Empaquetado PyFunc**: El wrapper encapsula el pipeline dual, la preparación de features y el margen de incertidumbre calibrado ($\pm$USD 50,927), produciendo exactamente las columnas `salary_min_usd`, `salary_max_usd` y `salary_midpoint_usd`.

---

## 7. Pruebas Automatizadas

Para ejecutar la suite de pruebas del módulo ML:

```bash
pytest tests/unit tests/contract
```

Las pruebas verifican:
- Ingesta multi-snapshot, deduplicación y preflight de snapshots (`test_collect.py`).
- Validación de datos e invariantes de calidad (`test_validation.py`).
- Paridad exacta del contrato de las 24 features (`test_features.py`).
- Preprocesamiento y cálculo de límites operacionales (`test_preprocess.py`).
- Calificación de modelos y calibración de incertidumbre (`test_qualify.py`).
- Entrenamiento definitivo y serialización del bundle (`test_train.py`).
- Evaluación sobre test ciego e invariantes (`test_evaluate.py`).
- Wrapper PyFunc, serialización, firmas e input examples (`test_pyfunc.py`).
- Tracking en MLflow y linaje (`test_tracking.py`).
- Idempotencia, tags y paridad en Model Registry (`test_registry.py`).

---

## 8. Documentación Relacionada

- [`../README.md`](../README.md) — Visión general del monorepo y arquitectura E2E.
- [`../DEPLOYMENT.md`](../DEPLOYMENT.md) — Manual reproducible de despliegue, bootstrap y operación.
- [`../model_provider/README.md`](../model_provider/README.md) — Infraestructura MLflow, promoción, serving inmutable y alineación.
- [`docs/CONFIGURACION_DVC_GOOGLE_DRIVE.md`](docs/CONFIGURACION_DVC_GOOGLE_DRIVE.md) — Guía de configuración para remoto DVC con Google Drive.
- [`docs/DVC_NGINX_SERVER.md`](docs/DVC_NGINX_SERVER.md) — Servidor DVC autogestionado con Docker, Nginx y SSH.
- [`docs/DVC_NGINX_CONFIG.md`](docs/DVC_NGINX_CONFIG.md) — Configuración y uso de clientes DVC (HTTP / SSH).
