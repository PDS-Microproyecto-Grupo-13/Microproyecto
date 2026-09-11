# MODEL_PROVIDER — REPORTE SUBETAPA 2: Adecuación del Entorno de Serving del Modelo Salarial

**Fecha:** 2026-09-11  
**Repositorio:** `PDS-Microproyecto-Grupo-13/Microproyecto`  
**Tracking URI:** `http://mlflow-tracking:5000` (interno red Docker) / `http://localhost:5000` (host)  
**Modelo:** `salary-predictor`  
**Alias:** `champion`  
**Versión Servida:** `1` (`models:/salary-predictor/1`)  
**Puerto de Serving:** `5001`  

---

## 1. Resumen Ejecutivo

En esta Subetapa 2 se adecuó exitosamente el entorno de ejecución del servicio de inferencia ([model_provider/inference](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/inference)), permitiendo que el contenedor Docker resuelva `salary-predictor@champion` hacia `salary-predictor:1` e inicie `mlflow models serve` sirviendo el modelo PyFunc salarial real con LightGBM.

El contenedor `mlops-inference` se encuentra en estado **healthy**, respondiendo HTTP 200 en `/health` y `/ping`. Asimismo, se ejecutó un smoke test directo contra `/invocations` demostrando que el modelo real procesa solicitudes válidas (incluyendo la variable `regions`) y produce estimaciones de salario (`salary_min_usd`, `salary_max_usd`, `salary_midpoint_usd`) numéricamente coherentes y matemáticamente consistentes.

---

## 2. Modificaciones en Configuración y Dependencias

### 2.1. Dockerfile ([model_provider/inference/Dockerfile](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/inference/Dockerfile))
- **Alineación del Runtime Python:** Se cambió la imagen base de `python:3.12-slim` a `python:3.11-slim`. Durante los diagnósticos de compatibilidad, se comprobó que Python 3.12 emitía múltiples advertencias críticas de deserialización (`InconsistentVersionWarning` en 4 estimadores de scikit-learn y advertencia de disparidad de versión menor en `mlflow.pyfunc`). Al alinear con Python 3.11 (el entorno en el que se entrenó y serializó el modelo), se eliminaron todas las alertas de incompatibilidad de runtime.
- **Soporte OpenMP / LightGBM:** Se incorporó el paquete del sistema operativo `libgomp1` mediante `apt-get install -y --no-install-recommends libgomp1`. Sin esta librería compartida, LightGBM fallaba inmediatamente al inicio con `OSError: libgomp.so.1: cannot open shared object file`.

### 2.2. Requirements ([model_provider/inference/requirements.txt](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/inference/requirements.txt))
- **Eliminación de CatBoost:** Se removió la dependencia obsoleta `catboost>=1.2.10,<2`, confirmando que no es requerida por el pipeline productivo de inferencia.
- **Inclusión de Dependencias Canónicas:** Se declararon explícitamente:
  - `mlflow>=3.15.0,<4`
  - `lightgbm>=4.0.0`
  - `scikit-learn>=1.5.0`
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`
  - `joblib>=1.3.0`
  - `httpx>=0.27.0`, `pydantic>=2.0.0`, `rich>=13.0.0`

### 2.3. Bloqueo de Versiones ([model_provider/inference/requirements.lock.txt](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/inference/requirements.lock.txt))
- El archivo de lock fue **regenerado formalmente** utilizando `pip-compile` dentro de un contenedor `python:3.11-slim` sin ninguna edición manual:
  ```bash
  pip-compile --output-file=requirements.lock.txt requirements.txt
  ```
- Versiones fijadas destacadas:
  - `mlflow==3.15.1`
  - `lightgbm==4.7.0`
  - `scikit-learn==1.9.0`
  - `joblib==1.5.3`
  - `pandas==2.3.3`
  - `numpy==2.4.6`
  - `cloudpickle==3.1.2`

### 2.4. Alinear MODEL_NAME
Se actualizó el nombre canónico del modelo a `salary-predictor`:
- En [docker-compose.yml](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/docker-compose.yml) (servicio `inference`):
  ```yaml
  environment:
    - MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://mlflow-tracking:5000}
    - MODEL_NAME=salary-predictor
    - MODEL_ALIAS=champion
    - INFERENCE_HOST=0.0.0.0
    - INFERENCE_PORT=5001
  ```
- En [model_provider/config/inference.env.example](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/config/inference.env.example):
  ```env
  MODEL_NAME=salary-predictor
  MODEL_ALIAS=champion
  ```

---

## 3. Resolución Inmutable y Logging de Startup

Se verificó y robusteció la trazabilidad en [model_provider/inference/start.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/inference/start.py). El proceso:
1. Conecta con el servidor MLflow Tracking.
2. Consulta el alias `champion` con `client.get_model_version_by_alias("salary-predictor", "champion")`.
3. Resuelve la versión concreta: `version 1`.
4. Construye la URI inmutable fijada: `models:/salary-predictor/1`.
5. Lanza `mlflow models serve --model-uri models:/salary-predictor/1 --host 0.0.0.0 --port 5001 --env-manager local`.

> [!IMPORTANT]
> **No Hot-Reload:** El servicio no utiliza la URI dinámica `models:/salary-predictor@champion` como argumento de ejecución, sino la versión resuelta al arranque. Cualquier reasignación posterior del alias en el Registry no afecta al proceso en ejecución hasta su reinicio deliberado.

### Logs Estructurados de Startup Observados en el Contenedor
```text
2026-09-11T07:10:54Z [INFO] service=inference event=tracking_connected tracking_uri=http://mlflow-tracking:5000
2026-09-11T07:10:54Z [INFO] service=inference event=model_resolved model_name=salary-predictor requested_alias=champion resolved_version=1 exact_model_uri=models:/salary-predictor/1 run_id=db1fd1c17c8e499d976c8446fe17e3ec
2026-09-11T07:10:54Z [INFO] service=inference event=model_server_starting model_name=salary-predictor requested_alias=champion resolved_version=1 exact_model_uri=models:/salary-predictor/1 command=/usr/local/bin/python -m mlflow models serve --model-uri models:/salary-predictor/1 --host 0.0.0.0 --port 5001 --env-manager local
2026/09/11 07:11:04 INFO mlflow.models.flavor_backend_registry: Selected backend for flavor 'python_function'
2026/09/11 07:11:06 INFO mlflow.pyfunc.backend: === Running command 'exec uvicorn --host 0.0.0.0 --port 5001 --workers 1 mlflow.pyfunc.scoring_server.app:app'
INFO:     Started server process [50]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5001 (Press CTRL+C to quit)
```

---

## 4. Estado de Contenedores y Healthchecks

### 4.1. Estado Docker
Comando: `docker ps --filter name=mlops-inference`
```text
CONTAINER ID   IMAGE                     COMMAND             CREATED          STATUS                    PORTS                                         NAMES
f85aeaac5461   microproyecto-inference   "python start.py"   45 seconds ago   Up 43 seconds (healthy)   0.0.0.0:5001->5001/tcp, [::]:5001->5001/tcp   mlops-inference
```

### 4.2. Sondeo de Health y Ping
- `curl -i http://localhost:5001/health`
  ```text
  HTTP/1.1 200 OK
  server: uvicorn
  content-length: 1
  content-type: application/json
  ```
- `curl -i http://localhost:5001/ping`
  ```text
  HTTP/1.1 200 OK
  server: uvicorn
  content-length: 1
  content-type: application/json
  ```

---

## 5. Smoke Test Directo de Inferencia (`/invocations`)

Se realizaron peticiones HTTP POST reales directamente a `http://localhost:5001/invocations` utilizando el formato nativo de MLflow (`dataframe_split`) y cumpliendo con el contrato del PyFunc registrado (incluyendo la variable `regions`).

### 5.1. Prueba 1: Senior Machine Learning Engineer (Americas / US)
**Request Payload:**
```json
{
  "dataframe_split": {
    "columns": [
      "title",
      "company",
      "company_is_agency",
      "countries",
      "regions",
      "experience_level",
      "experience_years",
      "has_remote",
      "work_mode",
      "tags",
      "published"
    ],
    "data": [
      [
        "Senior Machine Learning Engineer",
        "Tech Innovators Inc",
        false,
        "United States",
        "Americas",
        "senior",
        6.0,
        true,
        2.0,
        "Python|Machine Learning|Docker|Kubernetes|AWS",
        "2026-08-16T10:00:00Z"
      ]
    ]
  }
}
```

**Response HTTP 200:**
```json
{
  "predictions": [
    {
      "salary_min_usd": 156871.77334692486,
      "salary_max_usd": 210478.6139860343,
      "salary_midpoint_usd": 183675.19366647958
    }
  ]
}
```

### 5.2. Verificación Matemática y de Calidad
- **Valores numéricos finitos:** Sí.
- **Valores positivos:** Sí (`min > 0`, `max > 0`, `midpoint > 0`).
- **Coherencia de rango (`min <= max`):**  
  `156,871.773347 <= 210,478.613986` (Cumplido).
- **Exactitud de Punto Medio (`midpoint == (min + max) / 2`):**  
  `(156871.77334692486 + 210478.6139860343) / 2 = 183675.19366647958` (Cumplido exactamente).

### 5.3. Prueba 2: Junior Backend Developer (Europe / Spain)
**Request Payload:**
```json
{
  "dataframe_split": {
    "columns": [
      "title",
      "company",
      "company_is_agency",
      "countries",
      "regions",
      "experience_level",
      "experience_years",
      "has_remote",
      "work_mode",
      "tags",
      "published"
    ],
    "data": [
      [
        "Junior Backend Developer",
        "Startup Labs",
        false,
        "Spain",
        "Europe",
        "junior",
        1.0,
        true,
        2.0,
        "Python|Django|SQL",
        "2026-08-25T08:00:00Z"
      ]
    ]
  }
}
```

**Response HTTP 200:**
```json
{
  "predictions": [
    {
      "salary_min_usd": 39170.34222972772,
      "salary_max_usd": 50235.76854631388,
      "salary_midpoint_usd": 44703.0553880208
    }
  ]
}
```
Verificación: `(39170.34222972772 + 50235.76854631388) / 2 = 44703.0553880208` (Exacto).

---

## 6. Resultados de Pruebas Unitarias

Se incorporaron tres nuevos casos de prueba en [model_provider/tests/test_start.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/tests/test_start.py):
- `test_load_config_from_env_salary_predictor`: Comprueba la carga de configuración canónica `salary-predictor`.
- `test_resolve_model_info_salary_predictor_champion`: Comprueba la resolución de `salary-predictor@champion` a la versión 1.
- `test_build_serve_command_exact_version_pinning_no_hot_reload`: Comprueba que la línea de comandos de `mlflow models serve` fija la versión estricta (`models:/salary-predictor/1`) y no el alias, impidiendo hot-reload implícito.

### Resultado de Pytest
Comando: `ml/.venv/bin/pytest model_provider/tests -v`
```text
============================= test session starts ==============================
platform linux -- Python 3.11.2, pytest-8.4.2, pluggy-1.6.0
rootdir: /home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider
configfile: pyproject.toml
collected 30 items

model_provider/tests/test_healthcheck.py::test_check_health_success PASSED [  3%]
model_provider/tests/test_healthcheck.py::test_check_health_fallback_to_ping PASSED [  6%]
model_provider/tests/test_healthcheck.py::test_check_health_all_endpoints_fail PASSED [ 10%]
model_provider/tests/test_model_info.py::test_format_timestamp PASSED    [ 13%]
model_provider/tests/test_model_info.py::test_inspect_model_general_success PASSED [ 16%]
model_provider/tests/test_model_info.py::test_inspect_model_by_alias_success PASSED [ 20%]
model_provider/tests/test_model_info.py::test_inspect_model_by_version_success PASSED [ 23%]
model_provider/tests/test_model_info.py::test_inspect_model_not_found_exits PASSED [ 26%]
model_provider/tests/test_model_info.py::test_inspect_model_alias_not_found_exits PASSED [ 30%]
model_provider/tests/test_promote_model.py::test_promote_model_version_success_with_previous_alias PASSED [ 33%]
model_provider/tests/test_promote_model.py::test_promote_model_version_initial_promotion_no_prev_alias PASSED [ 36%]
model_provider/tests/test_promote_model.py::test_promote_model_version_idempotent_already_assigned PASSED [ 40%]
model_provider/tests/test_promote_model.py::test_promote_model_version_registered_model_not_found PASSED [ 43%]
model_provider/tests/test_promote_model.py::test_promote_model_version_target_version_not_found PASSED [ 46%]
model_provider/tests/test_promote_model.py::test_promote_model_version_rejects_status_not_ready PASSED [ 50%]
model_provider/tests/test_promote_model.py::test_promote_model_version_rejects_missing_eligible_tag PASSED [ 53%]
model_provider/tests/test_promote_model.py::test_promote_model_version_rejects_eligible_false PASSED [ 56%]
model_provider/tests/test_promote_model.py::test_promote_model_version_set_alias_failure PASSED [ 60%]
model_provider/tests/test_start.py::test_load_config_from_env_valid PASSED [ 63%]
model_provider/tests/test_start.py::test_load_config_from_env_missing_model_name PASSED [ 66%]
model_provider/tests/test_start.py::test_load_config_from_env_invalid_port PASSED [ 70%]
model_provider/tests/test_start.py::test_wait_for_tracking_server_success PASSED [ 73%]
model_provider/tests/test_start.py::test_wait_for_tracking_server_failure PASSED [ 76%]
model_provider/tests/test_start.py::test_resolve_model_info_success PASSED [ 80%]
model_provider/tests/test_start.py::test_resolve_model_info_missing_registered_model PASSED [ 83%]
model_provider/tests/test_start.py::test_resolve_model_info_missing_alias PASSED [ 86%]
model_provider/tests/test_start.py::test_build_serve_command PASSED      [ 90%]
model_provider/tests/test_start.py::test_load_config_from_env_salary_predictor PASSED [ 93%]
model_provider/tests/test_start.py::test_resolve_model_info_salary_predictor_champion PASSED [ 96%]
model_provider/tests/test_start.py::test_build_serve_command_exact_version_pinning_no_hot_reload PASSED [100%]

============================== 30 passed in 3.28s ==============================
```

---

## 7. Alcance y Restricciones Cumplidas

- **backend/ y frontend/ Intactos:** No se realizaron cambios en archivos de backend (`app/core/config.py`, `InferenceClient`, schemas, etc.) ni de frontend.
- **ml/ Intacto:** No se modificaron pipelines, notebooks, `model.joblib` ni artefactos de tracking.
- **MLflow Registry Intacto:** No se crearon nuevos runs, no se registraron nuevas versiones y se mantuvo intacto el alias `champion -> 1`.

---

## 8. Riesgos Restantes para Subetapas Posteriores

1. **Desalineación del Contrato Backend (`regions` vs `region`):**
   - El modelo servido requiere obligatoriamente `regions` y `countries` en el payload de entrada.
   - El backend actual (`backend/app/schemas/prediction.py`) aún tiene definido `region: str` en lugar de `regions: str`. Esto deberá abordarse en la Subetapa 3 al conectar el cliente HTTP del backend con el servicio de inferencia.
2. **Nombre de Modelo en Configuración Backend:**
   - `backend/app/core/config.py` y el bloque `backend` de `docker-compose.yml` aún referencian `salary_predict_model`, lo cual deberá alinearse a `salary-predictor` en la Subetapa 3.

---

MODEL_PROVIDER_STAGE_2_STATUS=READY_FOR_STAGE_3
