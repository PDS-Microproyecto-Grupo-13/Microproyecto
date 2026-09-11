# REPORTE DE RECONOCIMIENTO OPERACIONAL: MODEL_PROVIDER

**Fecha de evaluación:** 2026-09-11  
**Módulo evaluado:** `model_provider/`, `backend/`, `ml/`, `docker-compose.yml`  
**Estado inicial de partida en MLflow Model Registry:**
- **Modelo registrado:** `salary-predictor`
- **Versión:** `1`
- **Estado (Status):** `READY`
- **Aliases:** Ninguno (`aliases: {}`)
- **Run ID origen:** `db1fd1c17c8e499d976c8446fe17e3ec` (Fase 6 Tracking)

---

## A. Arquitectura Actual Real de `model_provider`

El módulo `model_provider/` encapsula la infraestructura de MLflow Tracking, Model Registry y serving de inferencia para el sistema **SalaryPredict**.

```text
                        ml/ (Pipeline Reproducible DVC)
                                      │
                                      │ MLflow Client (Tracking & Registry)
                                      ▼
                ┌──────────────────────────────────────────┐
                │         mlflow-tracking (5000)           │
                │   • Tracking Server & MLflow UI          │
                │   • Model Registry                       │
                │   • SQLite: /var/lib/mlflow/db/mlflow.db │
                │   • Artifacts: /var/lib/mlflow/artifacts │
                └─────────────────────┬────────────────────┘
                                      │
                                      │ models:/salary-predictor/1
                                      │ alias: champion (pendiente de asignación)
                                      ▼
                ┌──────────────────────────────────────────┐
                │            inference (5001)              │
                │   • start.py (Model Resolver al arrancar)│
                │   • mlflow models serve                  │
                │   • Inmutable: Requiere reinicio         │
                │   • Healthcheck: GET /health, /ping      │
                └─────────────────────┬────────────────────┘
                                      │
                                      │ HTTP (POST /invocations)
                                      ▼
                ┌──────────────────────────────────────────┐
                │              backend (8000)              │
                │   • FastAPI REST Microservice            │
                │   • InferenceClient (HTTP Adapter)       │
                └─────────────────────┬────────────────────┘
                                      │
                                      │ HTTP Proxy (/api/...)
                                      ▼
                ┌──────────────────────────────────────────┐
                │             frontend (5173)              │
                │   • React + TypeScript Dashboard (Vite)  │
                └──────────────────────────────────────────┘
```

### Componentes y Responsabilidades

1. **`model_provider/tracking/`**:
   - **`Dockerfile`**: Imagen base `python:3.12-slim` con `curl` y dependencias de MLflow.
   - **`entrypoint.sh`**: Lanza el servidor MLflow en `0.0.0.0:5000` con backend SQLite (`/var/lib/mlflow/db/mlflow.db`), default artifact root (`/var/lib/mlflow/artifacts`) y flag `--serve-artifacts`.
   - **Almacenamiento**: Persistencia desacoplada mediante volúmenes de Docker (`mlops-mlflow-db-data` y `mlops-mlflow-artifact-data`).
2. **`model_provider/inference/`**:
   - **`Dockerfile`**: Imagen base `python:3.12-slim` que expone el puerto 5001 y ejecuta `python start.py`.
   - **`start.py`**: Envoltorio operacional (Model Resolver). Lee variables de entorno (`MODEL_NAME`, `MODEL_ALIAS`, `MLFLOW_TRACKING_URI`, `INFERENCE_HOST`, `INFERENCE_PORT`), verifica conectividad con el servidor de tracking, consulta en el Model Registry qué versión concreta tiene el alias objetivo (`champion`), y arranca como subproceso `python -m mlflow models serve --model-uri models:/<MODEL_NAME>/<VERSION> --host <HOST> --port <PORT> --env-manager local`. Captura `SIGTERM` y `SIGINT` para un apagado ordenado.
   - **`healthcheck.py`**: Prueba `GET http://<HOST>:<PORT>/health` y si falla realiza fallback a `GET /ping` con timeout de 3 segundos.
   - **`requirements.txt` / `requirements.lock.txt`**: Dependencias de inferencia (contiene MLflow 3.0.0, CatBoost, Scikit-learn, Pandas, NumPy, pero **carece de LightGBM**).
3. **`model_provider/scripts/`**:
   - **`promote_model.py`**: CLI para asignar un alias (por defecto `champion`) a una versión de modelo en el Model Registry.
   - **`model_info.py`**: CLI para inspeccionar metadatos, versiones registradas y resolución de alias.
4. **`model_provider/dev/`**:
   - **`register_demo_model.py`**: Script de desarrollo para registrar modelos sintéticos de juguete (`salary_predict_model`) con Scikit-learn (`LinearRegression` y `GradientBoostingRegressor`) sobre 3 features (`years_experience`, `is_remote`, `skills_count`).
5. **`model_provider/config/`**:
   - Archivos de ejemplo `inference.env.example` y `tracking.env.example`.

---

## B. Flujo Actual Registry → Inference

El ciclo de vida de un modelo desde el registro hasta la atención de peticiones opera de la siguiente manera:

```text
[MLflow Model Registry]
         │
         │ 1. Versión candidata '1' en estado READY (sin alias)
         ▼
[Promoción (scripts/promote_model.py o MLflow UI)]
         │
         │ 2. Asignación del alias 'champion' a la versión '1'
         ▼
[model_provider/inference/start.py]
         │
         │ 3. client.get_model_version_by_alias(MODEL_NAME, 'champion')
         │ 4. Resuelve versión concreta: '1'
         │ 5. Construye URI: 'models:/salary-predictor/1'
         ▼
[mlflow models serve Subprocess]
         │
         │ 6. Descarga artefactos (model.joblib, código ml_pipeline)
         │ 7. Carga SalaryPredictorModel (PyFunc context)
         │ 8. Escucha en http://0.0.0.0:5001/invocations
         ▼
[Backend FastAPI (InferenceClient)]
         │
         │ 9. POST http://inference:5001/invocations con payload dataframe_split
         ▼
[SalaryPredictorModel.predict()]
         │
         │ 10. features = prepare_features(model_input)
         │ 11. range_preds = predict_range(features, bundle)
         │ 12. Genera salary_min_usd, salary_max_usd, salary_midpoint_usd
         ▼
[Respuesta HTTP]
         │
         │ 13. {"predictions": [{"salary_min_usd": ..., "salary_max_usd": ..., "salary_midpoint_usd": ...}]}
         ▼
[Backend Response -> Frontend UI]
```

---

## C. Promoción Actual y Soporte UI/CLI

Existen conceptualmente dos vías de promoción para asignar el alias `champion`:

### Camino A: Interfaz Gráfica de MLflow (MLflow UI)
- Un operador accede a `http://localhost:5000/#/models/salary-predictor`.
- Selecciona la versión deseada (versión 1) y en la pestaña de **Aliases** añade el alias `champion`.
- La UI invoca el endpoint de MLflow `POST /api/2.0/mlflow/registered-models/alias`.
- El alias queda formalmente registrado en la base de datos SQLite.

### Camino B: Script CLI (`model_provider/scripts/promote_model.py`)
- Se ejecuta:
  ```bash
  python model_provider/scripts/promote_model.py \
    --model salary-predictor \
    --version 1 \
    --alias champion \
    --tracking-uri http://localhost:5000
  ```
- El script implementa `promote_model_version(client, model_name, version, alias)`.

### Análisis de Capacidades y Limitaciones de la Promoción Actual

| Pregunta de Control | Estado Actual en Código | Detalle Técnico |
| :--- | :---: | :--- |
| **¿Existe soporte CLI implementado?** | **SÍ** | En [`model_provider/scripts/promote_model.py`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/scripts/promote_model.py). |
| **¿Qué validaciones realiza?** | **MÍNIMAS** | Solo valida que la versión exista en MLflow (`client.get_model_version`). Si no existe, eleva `ValueError`. |
| **¿Puede promover cualquier versión?** | **SÍ (RIESGO)** | No valida calidad, origen, lineage, métricas ni tags de elegibilidad. |
| **¿Comprueba que la versión esté `READY`?** | **NO (RIESGO)** | No inspecciona `model_version.status == "READY"`. Si una versión estuviera en `PENDING_REGISTRATION` o `FAILED_REGISTRATION`, intentaría asignarle el alias igualmente. |
| **¿Comprueba tags de elegibilidad (`eligible==true`)?** | **NO (RIESGO)** | Ignora los tags `candidate` y `eligible` generados en la Fase 7. |
| **¿Mueve/reemplaza correctamente el alias?** | **SÍ** | `set_registered_model_alias` es atómica en MLflow: retira el alias de la versión previa y se lo asigna a la nueva. El script reporta `previous_alias` y `to_version`. |
| **¿Hace deployment automáticamente?** | **NO** | Solo modifica metadatos en el Registry. No reinicia el contenedor de inferencia ni recarga el modelo en ejecución. |

---

## D. Cómo se Resuelve y Carga `champion`

El mecanismo de arranque y resolución en [`model_provider/inference/start.py`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/inference/start.py) opera bajo las siguientes reglas:

1. **Resolución previa al arranque**:
   - `start.py` se conecta a MLflow y ejecuta:
     ```python
     model_version = client.get_model_version_by_alias(
         name=config.model_name,
         alias=config.model_alias, # por defecto "champion"
     )
     ```
   - Si el modelo registrado no existe o el alias no está configurado, `start.py` emite un error estructurado (`model_lookup_failed` o `alias_lookup_failed`) y termina con `sys.exit(1)`.
2. **Fijación de versión concreta (Inmutabilidad)**:
   - `start.py` **NO** pasa `models:/<name>@champion` al servidor de MLflow.
   - Pasa la URI con la versión fija: `models:/<name>/<version>` (por ejemplo `models:/salary-predictor/1`).
   - Comando ejecutado:
     ```bash
     python -m mlflow models serve \
       --model-uri models:/salary-predictor/1 \
       --host 0.0.0.0 \
       --port 5001 \
       --env-manager local
     ```
3. **Comportamiento ante cambios de alias en caliente**:
   - Si mientras el contenedor está en ejecución alguien mueve el alias `champion` (por UI o CLI), el servidor de inferencia **no sufre ninguna alteración**: continúa atendiendo con la versión fijada en su inicio.
   - No existe hot-reload ni polling de alias.
4. **Distinción entre "Registry champion" y "Loaded version"**:
   - **Registry champion:** la versión que en este instante tiene asignado el alias en MLflow.
   - **Loaded version:** la versión concreta que fue cargada en memoria por el subproceso `mlflow models serve`.
   - **Brecha:** El código actual **no distingue programáticamente** ambos estados una vez iniciado. Solo emite un log al arrancar (`service=inference event=model_resolved ... version=1`). Ni el servidor de inferencia ni el backend ofrecen un endpoint para consultar qué versión está efectivamente activa o si hay divergencia (*drift*) frente al Registry.

---

## E. Estrategia Actual de Deploy / Redeploy

1. **Despliegue inicial**:
   - Se asegura que el modelo tenga el alias `champion` en MLflow.
   - Se levanta el servicio: `docker compose up -d inference`.
2. **Redeploy tras cambio de `champion`**:
   - Como la versión en servicio está fijada por línea de comandos, el redespacho exige un reinicio explícito:
     ```bash
     docker compose restart inference
     ```
   - Durante el reinicio, `start.py` vuelve a consultar el alias `champion`, resuelve la nueva versión (e.g. v2) y levanta el nuevo proceso.
3. **Impacto operacional (Downtime)**:
   - Durante el reinicio del contenedor, hay una ventana de indisponibilidad de entre 10 y 45 segundos (descarga de artefactos, inicialización de Python, carga de bundles y comprobación de salud).
   - En ese intervalo, `backend` fallará con error de conexión o timeout si intenta invocar `/invocations`.
   - No existe soporte nativo para despliegues *blue-green*, *canary* ni *zero-downtime* en la configuración actual.

---

## F. Dependencias Requeridas por `salary-predictor`

El modelo registrado en Fase 7 (`salary-predictor:1`) es un modelo dual LightGBM encapsulado mediante MLflow PyFunc. Para operar en inferencia, el entorno requiere satisfacer estrictamente sus dependencias.

### Comparativa de Dependencias (Entorno `ml/` vs Entorno `inference`)

| Componente | Versión en `ml/` (Entrenamiento) | Entorno `inference/requirements.lock.txt` | Estado / Compatibilidad |
| :--- | :---: | :---: | :--- |
| **Python** | `3.11.2` | `3.12-slim` (Dockerfile) | ⚠️ Discrepancia menor (verificar deserialización joblib) |
| **`mlflow`** | `3.15.1` | `3.0.0` | ⚠️ MLflow 3.x compatible, pero versionado menor diferente |
| **`lightgbm`** | `4.7.0` | **¡TOTALMENTE AUSENTE!** | ❌ **INCOMPATIBILIDAD BLOQUEANTE**: `ModuleNotFoundError: No module named 'lightgbm'` al deserializar `model.joblib`. |
| **`scikit-learn`** | `1.9.0` | `1.9.0` | ✅ Compatible |
| **`pandas`** | `2.3.3` | `2.3.3` | ✅ Compatible |
| **`numpy`** | `2.4.2` | `2.4.3` | ✅ Compatible |
| **`catboost`** | *(no usado)* | `1.2.10` | ℹ️ Innecesario (residuo de la plantilla original) |
| **Librería de sistema `libgomp1`** | Presente en host | No instalada explícitamente en Dockerfile | ⚠️ Requerida en Debian-slim para que LightGBM cargue OpenMP. |

### Estrategia `env-manager: local`
`start.py` ejecuta `mlflow models serve --env-manager local`.  
Con esta opción, MLflow **no crea un entorno virtual aislado ni instala los requirements del modelo**: depende 100% de los paquetes ya instalados en el contenedor. Por tanto, la ausencia de `lightgbm` en `model_provider/inference/requirements.lock.txt` es un fallo fatal garantizado en el momento en que se intente servir `salary-predictor`.

---

## G. Tabla de Compatibilidad Backend ↔ PyFunc

El modelo PyFunc espera un DataFrame de entrada que alimenta a `prepare_features()`.  
A continuación se compara el contrato esperado por PyFunc contra lo que el backend FastAPI (`backend/app/schemas/prediction.py` y `backend/app/clients/inference_client.py`) genera actualmente en `to_mlflow_record()`:

| Campo PyFunc | Campo Backend (`SalaryPredictionRequest`) | Transformación en `to_mlflow_record()` | Clasificación | Impacto Operacional / Detalle |
| :--- | :--- | :--- | :---: | :--- |
| `title` | `title` | `self.title.strip()` | **Compatible** | `prepare_features` aplica `normalize_text`. |
| `company` | `company` | `(self.company or "Sin información").strip() or "Sin información"` | **Compatible** | `prepare_features` normaliza y reemplaza vacíos por `"desconocido"`. |
| `company_is_agency` | `company_is_agency` | `self.company_is_agency` | **Compatible** | Booleano convertido a entero `0/1`. |
| `countries` | `country` | `"countries": self.country.strip()` | **Compatible (adaptado)** | El schema backend usa singular (`country`), pero `to_mlflow_record()` ya lo mapea a plural (`countries`). |
| `regions` | *(ausente en schema)* | **¡NO SE GENERA!** | ❌ **AUSENTE (BLOQUEANTE)** | `prepare_features()` ejecuta `df["regions"]`. Al faltar la columna, eleva `KeyError: 'regions'` y tumba la inferencia. |
| `experience_level` | `experience_level` | `self.experience_level` | **Compatible** | Backend valida regex `^(EN\|MI\|SE\|EX)$`, que coincide exactamente con las 4 clases de entrenamiento (`EN`, `MI`, `SE`, `EX`). |
| `experience_years` | `experience_years` | `self.experience_years` | **Compatible** | Acepta float o `None` (el pipeline imputa y marca `experience_years_missing`). |
| `has_remote` | `is_remote` | `"has_remote": self.is_remote` | **Compatible (adaptado)** | Backend usa `is_remote`, mapeado a `has_remote`. |
| `work_mode` | *(ausente en schema)* | `"work_mode": None` | **Semántica distinta** | Al ser `None`, el pipeline deriva: si `has_remote=False` -> `"presencial"`; si `has_remote=True` -> `"remoto_sin_detalle"`. No falla, pero pierde categorización fina (`híbrido`, `remoto_global`). |
| `tags` | `technologies` | `"tags": "\|".join(self.technologies)` | **Compatible** | El modelo busca sus 13 tokens de habilidad en este string con regex minúscula. |
| `published` | *(ausente en schema)* | `"published": datetime.now(UTC).isoformat()` | **Compatible** | El backend genera el timestamp actual. `prepare_features` extrae año y mes. |
| *(no usado)* | `topics` | `"topics": "\|".join(self.topics)` | **Compatible (inocuo)** | Se envía en el DataFrame pero el modelo no lo consume. |

### Contrato de Salida (Inferencia → Backend)
- **PyFunc retorna:** DataFrame con `[salary_min_usd, salary_max_usd, salary_midpoint_usd]`.
- **Formato MLflow `/invocations`:**
  ```json
  {
    "predictions": [
      {
        "salary_min_usd": 65000.0,
        "salary_max_usd": 95000.0,
        "salary_midpoint_usd": 80000.0
      }
    ]
  }
  ```
- **Backend `InferenceClient`:** Extrae exactamente `float(prediction["salary_min_usd"])`, `float(prediction["salary_max_usd"])`, `float(prediction["salary_midpoint_usd"])`.
- **Estado de salida:** **100% Compatible**.

---

## H. Gap Específico de `regions`

### 1. Diagnóstico del Gap
En la Fase 2 de `ml/`, se implementó paridad estricta con el notebook de referencia:
```python
# ml/src/ml_pipeline/features.py (Línea 100-108)
out["region"] = (
    df["regions"]
    .fillna("desconocido")
    .astype(str)
    .str.split("|")
    .str[0]
    .str.strip()
    .replace("", "desconocido")
)
```
La regla explícita fue:
> "No implementar fallback regions -> region -> desconocido. Si falta la columna regions, fallar claramente. La compatibilidad con backend se resolverá en la fase de serving."

Actualmente, `to_mlflow_record()` en `backend/app/schemas/prediction.py` **no incluye `"regions"`**. Cuando el backend invoca a MLflow, este construye un DataFrame sin dicha columna, produciendo de inmediato:
```text
KeyError: 'regions'
```
Y el backend recibe un HTTP 500 / 400 de MLflow, respondiendo con `ExternalServiceError`.

### 2. Opciones de Resolución en la Fase de Serving

| Alternativa | Dónde se implementa | Descripción | Ventajas | Desventajas |
| :--- | :--- | :--- | :--- | :--- |
| **Opción 1: Inyección fija en `to_mlflow_record`** | `backend/app/schemas/prediction.py` | Añadir `"regions": "desconocido"` directamente al diccionario que se envía a MLflow. | Cero cambios en la API pública de backend; implementación de 1 sola línea; respeta contrato de `ml/`. | No permite que un cliente que conozca su región la informe. |
| **Opción 2: Extensión de schema público con valor por defecto** | `backend/app/schemas/prediction.py` | Añadir `region: str \| None = None` a `SalaryPredictionRequest`, y en `to_mlflow_record` mapear `"regions": (self.region or "desconocido").strip()`. | Permite a frontend enviar la región si la tiene, manteniendo compatibilidad hacia atrás si no se envía. | Requiere tocar schema de request (aunque no rompe si es opcional). |
| **Opción 3: Mapeo geográfico `country -> regions`** | `backend/app/schemas/prediction.py` o servicio intermedio | Diccionario que mapee países comunes (`Colombia -> Americas`, `Germany -> Europe`, etc.) y fallback a `"desconocido"`. | Mayor precisión predictiva para perfiles donde el país es conocido. | Añade lógica de negocio geográfica al backend. |

**Recomendación técnica para implementación:**  
La **Opción 2** combinada con fallback a `"desconocido"` es la solución más limpia y robusta, resolviendo el gap en la capa de adaptación del backend sin modificar el código de `ml/`.

---

## I. Mecanismos Actuales de Health / Status / Trazabilidad

1. **Health de Inferencia (`model_provider/inference/healthcheck.py`)**:
   - Sondea `/health` y `/ping`.
   - Verifica únicamente que el proceso Gunicorn/Uvicorn de MLflow esté respondiendo peticiones HTTP 200.
   - No verifica si el modelo en memoria corresponde al `champion` vigente en el Registry.
2. **Health del Backend (`backend/app/api/routes/health.py`)**:
   - Endpoint `/api/v1/health`.
   - Devuelve `{"status": "ok", "service": "mlops-backend", "version": "0.1.0"}`.
   - **Brecha:** Es un chequeo puramente interno del backend; no valida la conectividad ni salud de `inference`.
3. **Trazabilidad de Versión y Modelo (`backend/app/api/routes/predictions.py`)**:
   - En la respuesta de predicción (`SalaryPredictionResponse`):
     ```json
     {
       "prediction": { "minimum_usd": 65000.0, "maximum_usd": 95000.0, "midpoint_usd": 80000.0 },
       "model": { "name": "salary_predict_model", "alias": "champion" },
       "warnings": []
     }
     ```
   - **Brecha Crítica de Trazabilidad:**
     - El backend **no conoce la versión real cargada** (`loaded_version`) ni el `run_id`.
     - Solo devuelve las variables de configuración estáticas `settings.MODEL_NAME` y `settings.MODEL_ALIAS`.
     - Si el alias en MLflow cambia a la versión 2 pero el contenedor de inferencia sigue corriendo con la versión 1, el backend seguirá afirmando falsamente que la predicción provino de `champion`.
     - MLflow models serve en `/invocations` no incluye cabeceras ni metadatos con el `ModelVersion`.

---

## J. Estado de Docker Compose y Entorno Operacional

Al inspeccionar el estado real del sistema mediante `docker ps` y `docker compose logs`:

1. **Estado de Contenedores en Vivo:**
   - `mlops-tracking` (MLflow Server): **UP y Healthy** en `0.0.0.0:5000`. Contiene el modelo `salary-predictor` versión 1 en estado `READY`.
   - `mlops-inference`: **Restarting (crash-loop continuo)**.
2. **Causa Raíz del Crash-Loop de `inference`:**
   En los logs de `mlops-inference` se evidencia el fallo reiterado:
   ```text
   2026-09-11T05:54:46Z [ERROR] service=inference event=model_lookup_failed model=salary_predict_model error=Registered model 'salary_predict_model' does not exist
   2026-09-11T05:54:46Z [ERROR] service=inference event=startup_failed error=Registered model 'salary_predict_model' does not exist: RESOURCE_DOES_NOT_EXIST: Registered Model with name=salary_predict_model not found
   ```
   - **Discrepancia de Nombres:**
     - En `docker-compose.yml` (líneas 45 y 83): `MODEL_NAME=salary_predict_model`.
     - En MLflow Model Registry: el modelo canónico registrado en Fase 7 es `salary-predictor`.
     - En `backend/app/core/config.py`: `MODEL_NAME: str = "salary_predict_model"`.
   - **Discrepancia de Alias:**
     - `docker-compose.yml` especifica `MODEL_ALIAS=champion`.
     - En MLflow Model Registry, `salary-predictor` v1 aún no tiene asignado el alias `champion`.
3. **Efecto Cascada:**
   - Como `backend` tiene `depends_on: inference: condition: service_healthy`, y `frontend` tiene `depends_on: backend: condition: service_healthy`, **el resto del sistema no puede iniciar**.

---

## K. Tests Existentes y Brechas

### Tests Existentes en `model_provider/tests/` (100% Mockeados)
- `test_promote_model.py` (4 tests): prueban asignación con/sin alias previo y manejo de errores de MLflow con `MagicMock`.
- `test_start.py` (9 tests): prueban carga de variables de entorno, reintentos de conexión, resolución de modelo y construcción del comando CLI usando mocks.
- `test_model_info.py` (6 tests): prueban formateo de timestamps e inspección por versión o alias usando mocks.
- `test_healthcheck.py` (3 tests): prueban `/health`, fallback a `/ping` y fallo de conexión con mocks de `urllib`.

### Brechas Identificadas en Tests
1. **Cero pruebas de integración reales:** No hay tests que levanten un MLflow local en SQLite y verifiquen que `start.py` resuelva y arranque un modelo de verdad.
2. **Ausencia de tests con LightGBM:** Ningún test en `model_provider` carga un modelo que requiera LightGBM; todos los tests de demo usaban Scikit-learn.
3. **Ausencia de tests para el contrato de `salary-predictor`:** No se valida si el schema de entrada del backend es compatible con el PyFunc de inferencia.
4. **Ausencia de tests para validación de elegibilidad en promoción:** No se prueba qué pasa si se intenta promover una versión con `eligible == false` o `status != READY`.

---

## L. Riesgos Reales Identificados

1. **Riesgo 1 (Crítico - Crash Inmediato): Ausencia de `lightgbm` en Serving:**
   El contenedor de inferencia fallará al importar LightGBM en cuanto intente servir `salary-predictor:1`.
2. **Riesgo 2 (Crítico - Incompatibilidad de Contrato): Ausencia de `regions`:**
   Cualquier petición del backend fallará con `KeyError: 'regions'` en `prepare_features()`.
3. **Riesgo 3 (Crítico - Discrepancia de Configuración): Nombre de Modelo Desalineado:**
   `salary_predict_model` en Compose/Backend vs `salary-predictor` en Registry.
4. **Riesgo 4 (Gobernanza): Promoción sin Salvaguardas:**
   `promote_model.py` permite promover cualquier versión sin comprobar que esté `READY` ni que sus tags certifiquen `eligible == true`.
5. **Riesgo 5 (Operacional): Drift de Versión Silencioso:**
   El backend reporta `alias: champion` sin conocer la versión real cargada en memoria, ocultando desfases si el contenedor no fue reiniciado tras una promoción.
6. **Riesgo 6 (Disponibilidad): Downtime por Reinicio Obligatorio:**
   La estrategia de inmutabilidad exige reiniciar el contenedor para cambiar de modelo, interrumpiendo el servicio a los clientes.

---

## M. Cambios Mínimos Recomendados

1. **En `model_provider/inference`:**
   - Añadir `lightgbm>=4.0.0` a `requirements.txt` y regenerar `requirements.lock.txt`.
   - En `Dockerfile`, añadir `libgomp1` al `apt-get install` para soporte OpenMP de LightGBM en Linux slim.
   - Remover `catboost` de los requirements de inferencia para aligerar la imagen.
2. **En `model_provider/scripts/promote_model.py`:**
   - Añadir verificación de `model_version.status == "READY"`.
   - Añadir verificación de que `model_version.tags.get("eligible") == "true"` (impidiendo promover modelos que fallaron calificación o evaluación).
3. **En Configuración (`docker-compose.yml`, `config/inference.env.example`, `backend/app/core/config.py`):**
   - Unificar `MODEL_NAME` a `salary-predictor`.
4. **En `backend/app/schemas/prediction.py`:**
   - En `to_mlflow_record()`, inyectar `"regions": (self.region or "desconocido").strip()` (y opcionalmente declarar `region: str | None = None` en `SalaryPredictionRequest`).
5. **En Trazabilidad:**
   - Documentar explícitamente en `DEPLOYMENT.md` y logs que la versión servida queda fijada al iniciar y requiere `docker compose restart inference` para actualizarse.

---

## N. Plan Propuesto de Implementación por Subetapas

Se recomienda estructurar la implementación en las siguientes 5 subetapas progresivas:

### Subetapa 1: Robustecimiento de Promoción y Asignación de Champion
- Actualizar `model_provider/scripts/promote_model.py` para validar `status == "READY"` y `eligible == "true"`.
- Añadir tests unitarios para estas salvaguardas en `model_provider/tests/test_promote_model.py`.
- Ejecutar la promoción formal de `salary-predictor` versión 1 a `champion` mediante la CLI y comprobar que en MLflow UI se refleje el alias.

### Subetapa 2: Adecuación del Entorno de Serving (`model_provider/inference`)
- Modificar `model_provider/inference/requirements.txt` (incorporando `lightgbm>=4.0.0`, eliminando `catboost`).
- Actualizar `requirements.lock.txt` y `Dockerfile` (incluyendo `libgomp1`).
- Alinear `MODEL_NAME=salary-predictor` en `inference.env.example` y `docker-compose.yml`.
- Reconstruir la imagen de `inference` y verificar localmente que `start.py` resuelva `champion`, levante `mlflow models serve` y responda 200 en `/health`.

### Subetapa 3: Alineación del Contrato Backend ↔ Inferencia
- Modificar `backend/app/schemas/prediction.py` para resolver el gap de `regions` en `to_mlflow_record()`.
- Actualizar `backend/app/core/config.py` con `MODEL_NAME="salary-predictor"`.
- Validar mediante prueba unitaria/de cliente en `backend/tests/test_predictions.py` que el payload enriquecido sea aceptado por el cliente de inferencia.

### Subetapa 4: Despliegue y Verificación E2E en Docker Compose
- Levantar la infraestructura completa: `docker compose up -d --build`.
- Verificar la cadena de salud: `mlflow-tracking` (healthy) -> `inference` (healthy) -> `backend` (healthy) -> `frontend` (up).
- Realizar prueba de predicción end-to-end desde el navegador (`http://localhost:5173`) y vía `curl` contra el backend (`http://localhost:8000/api/v1/predictions`).
- Validar respuesta salarial esperada en dólares ($USD$) con rangos e intervalos válidos.

### Subetapa 5: Procedimiento Operacional de Cambio de Champion y Rollback
- Documentar y validar el flujo de promoción de una nueva versión o reversión de alias.
- Verificar el comportamiento de inmutabilidad y reinicio controlado (`docker compose restart inference`).
- Actualizar `DEPLOYMENT.md` reflejando el modelo salarial LightGBM real y los nombres canónicos de servicio.

---

```text
MODEL_PROVIDER_RECON_STATUS=READY_FOR_IMPLEMENTATION_PLAN
```
