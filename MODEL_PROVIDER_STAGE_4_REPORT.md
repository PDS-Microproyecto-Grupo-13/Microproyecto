# MODEL_PROVIDER — ETAPA 4 REPORT: Integración E2E Completa y Trazabilidad Runtime

## 1. Resumen Ejecutivo

La **Subetapa 4 de MODEL_PROVIDER** ha consolidado la integración extremo a extremo del sistema de predicción salarial, estableciendo una arquitectura robusta, trazable y resiliente que enlaza:

```
[Usuario / Navegador]
        │
        ▼
[Frontend React / Nginx] (Puerto 5173)
        │  /api/v1/predictions
        ▼
[Backend FastAPI] (Puerto 8000)
        │  POST /invocations (Puerto 5001)
        │  GET /status (Puerto 5002)
        ▼
[Serving MLflow PyFunc] (Inference)
        │  models:/salary-predictor/1
        ▼
[MLflow Model Registry / Tracking] (Puerto 5000)
        alias: champion -> version 1
```

Además, se cerró formalmente la brecha de trazabilidad operacional distinguiendo entre:
- **Registry champion version**: la versión que el alias `champion` apunta en el Registry (`v1`).
- **Runtime loaded version**: la versión inmutable fijada al momento de arranque del proceso servidor de inferencia (`v1`).

Se implementó el servidor de estado desacoplado en el puerto `5002` dentro del contenedor de inferencia, la herramienta CLI de verificación operacional `model_provider/scripts/check_alignment.py`, y se levantó la pila completa de 4 contenedores Docker Compose con salud confirmada y pruebas funcionales E2E satisfactorias.

---

## 2. Trazabilidad Operacional: Registry vs Serving Runtime

### 2.1 El Principio de Inmutabilidad en Serving
En un entorno de producción MLOps, cuando un contenedor de inferencia arranca resolviendo `salary-predictor@champion`, fija la versión exacta (`models:/salary-predictor/1`). Si posteriormente en el Registry el alias `champion` se reasigna a una nueva versión candidata (e.g. `v2`), el contenedor en ejecución continúa sirviendo de manera inmutable la versión `v1`.

Consultar al Registry en cada inferencia runtime induciría a un error de observabilidad (reportar falsamente que se sirve `v2` cuando en memoria se ejecuta `v1`).

### 2.2 Servidor de Estado Runtime (`model_provider/inference/start.py`)
Para exponer la versión real en memoria sin competir con el tráfico de inferencia ni introducir librerías externas pesadas, `start.py` integra un servidor HTTP basado en `http.server.ThreadingHTTPServer` (estándar de Python) que corre en un hilo daemon en el puerto `5002` (configurable vía `INFERENCE_STATUS_PORT`).

#### Endpoints expuestos:
- `GET /status`: Devuelve el payload estructurado con HTTP 200 (o HTTP 503 si el proceso de inferencia subyacente ha muerto):
```json
{
  "status": "ok",
  "model_name": "salary-predictor",
  "requested_alias": "champion",
  "resolved_version": "1",
  "loaded_version": "1",
  "exact_model_uri": "models:/salary-predictor/1",
  "run_id": "db1fd1c17c8e499d976c8446fe17e3ec",
  "source": "models:/m-3cc347ba70af4dc782ba5aa2a9b1c5b8",
  "model_server_pid": 17,
  "model_server_running": true,
  "started_at": "2026-09-11T07:45:17Z"
}
```
- `GET /health` / `GET /ping`: Comprobación rápida de vitalidad del proceso sirviente.

### 2.3 Herramienta de Alineación (`model_provider/scripts/check_alignment.py`)
Se implementó un script CLI que compara bidireccionalmente el Registry contra el runtime:
- **`synchronized` (Código de salida `0`)**: `registry_champion_version == runtime_loaded_version` y el proceso servidor está activo.
- **`redeploy_required` (Código de salida `2`)**: El alias `champion` en MLflow apunta a una versión diferente a la que el contenedor tiene en memoria, requiriendo un reinicio/despliegue del contenedor.
- **`runtime_unhealthy` / error (Código de salida `1`)**: El contenedor de serving no responde o el proceso está detenido.

Soporta flag `--json` para integración continua y automatización de alertas.

---

## 3. Adaptaciones en Backend y Frontend

### 3.1 Backend FastAPI
1. **Configuración (`backend/app/core/config.py`)**:
   - Incorporación de `INFERENCE_STATUS_URL: str = "http://inference:5002/status"`.
2. **Esquema (`backend/app/schemas/prediction.py`)**:
   - `ModelDeployment` enriquecido con `version: str | None = None`.
3. **Cliente (`backend/app/clients/inference_client.py`)**:
   - Inclusión de `get_runtime_status()` con sondeo no bloqueante, timeout de 2.0s y captura total de excepciones para no interrumpir inferencias si el status server experimenta latencia.
4. **Rutas (`backend/app/api/routes/predictions.py`)**:
   - `POST /api/v1/predictions`: Adjunta la versión runtime servida (`"version": "1"`) dentro del objeto `model`.
   - `GET /api/v1/predictions/model`: Reporta `name`, `alias` y `version`.
   - `GET /api/v1/predictions/status`: Expone el estado global de alineación operativa.

### 3.2 Frontend React
1. **Tipado de API (`frontend/src/services/salaryApi.ts`)**:
   - `SalaryPredictionResponse.model` incluye `version?: string | null`.
2. **Página de Predicción (`frontend/src/pages/SalaryPrediction/SalaryPredictionPage.tsx`)**:
   - Nombre de modelo unificado a `salary-predictor`.
   - Visualización explícita de la etiqueta y valor de la `Versión` del modelo servido en la tarjeta de metadatos del despliegue.

---

## 4. Estado de los Servicios en Docker Compose

Pila de contenedores validada y activa mediante `docker compose up -d --build`:

| Contenedor | Servicio | Imagen | Puertos Mapeados | Estado |
| :--- | :--- | :--- | :--- | :--- |
| `mlops-tracking` | MLflow Tracking & Registry | `microproyecto-mlflow-tracking` | `0.0.0.0:5000->5000/tcp` | Up (healthy) |
| `mlops-inference`| MLflow PyFunc Serving + Status | `microproyecto-inference` | `0.0.0.0:5001-5002->5001-5002/tcp` | Up (healthy) |
| `mlops-backend`  | FastAPI Application Backend | `microproyecto-backend` | `0.0.0.0:8000->8000/tcp` | Up (healthy) |
| `mlops-frontend` | React UI Dashboard + Nginx | `microproyecto-frontend` | `0.0.0.0:5173->80/tcp` | Up |

---

## 5. Verificación de Alineación Operacional

Ejecución de `model_provider/scripts/check_alignment.py`:
```
=================================================================
 MLflow Serving Operational Alignment Check
=================================================================
 Model Name:             salary-predictor
 Target Alias:           champion
 Registry Version:       1
 Runtime Loaded Version: 1
 Runtime Server Alive:   True
-----------------------------------------------------------------
 Alignment Status:       SYNCHRONIZED (serving latest champion)
=================================================================
```
Formato JSON (`--json`):
```json
{
  "model_name": "salary-predictor",
  "alias": "champion",
  "registry_version": "1",
  "runtime_loaded_version": "1",
  "runtime_server_running": true,
  "alignment_status": "synchronized",
  "runtime_metadata": {
    "status": "ok",
    "model_name": "salary-predictor",
    "requested_alias": "champion",
    "resolved_version": "1",
    "loaded_version": "1",
    "exact_model_uri": "models:/salary-predictor/1",
    "run_id": "db1fd1c17c8e499d976c8446fe17e3ec",
    "source": "models:/m-3cc347ba70af4dc782ba5aa2a9b1c5b8",
    "model_server_pid": 18,
    "model_server_running": true,
    "started_at": "2026-09-11T07:47:41Z"
  }
}
```

---

## 6. Pruebas E2E y Validación Numérica

### 6.1 Inferencia E2E a través del Proxy Frontend (Puerto 5173)
Solicitud enviada a `http://localhost:5173/api/v1/predictions`:
```json
{
  "title": "Senior Data Scientist",
  "experience_level": "SE",
  "experience_years": 5.0,
  "country": "Colombia",
  "is_remote": true,
  "company": "Tech Corp",
  "company_is_agency": false,
  "technologies": ["Python", "SQL", "Docker", "AWS"],
  "topics": ["Machine Learning", "Data Science"]
}
```

Respuesta recibida:
```json
{
  "prediction": {
    "minimum_usd": 68858.60147927287,
    "maximum_usd": 113123.68324692493,
    "midpoint_usd": 90991.1423630989
  },
  "model": {
    "name": "salary-predictor",
    "alias": "champion",
    "version": "1"
  },
  "warnings": []
}
```

### 6.2 Comprobaciones Numéricas y Físicas
- **Consistencia de orden**: `minimum_usd (68858.60) < maximum_usd (113123.68)` -> **CORRECTO**.
- **Punto medio**: `(68858.60147927287 + 113123.68324692493) / 2 = 90991.1423630989` -> **EXACTO**.
- **Límites de entrenamiento canónicos**:
  - `floor`: $10,935.57 USD <= 68,858.60 USD -> **VÁLIDO**.
  - `ceiling`: $720,000.00 USD >= 113,123.68 USD -> **VÁLIDO**.
- **Metadatos enlazados**: Nombre `"salary-predictor"`, alias `"champion"`, versión `"1"`.

---

## 7. Pruebas de Resiliencia y Control de Errores

| Escenario de Falla | Comportamiento Observado | Resultado |
| :--- | :--- | :--- |
| **Payload Inválido** (`title` corto, `experience_level` inválido) | Backend intercepta en validación Pydantic y responde `HTTP 422 Unprocessable Entity` con detalle de campos. No alcanza al modelo ni genera errores no controlados. | **Aprobado** |
| **Inference Down** (`docker compose stop inference`) | Backend responde de inmediato `HTTP 502 Bad Gateway` con esquema `ExternalServiceError`, mensaje claro y `request_id`. El endpoint `/predictions/status` responde con `runtime_inference: {"status": "unavailable"}` sin colapsar. | **Aprobado** |
| **Detección por CLI** (`check_alignment.py` con inference down) | Script falla controladamente con `[ERROR] Alignment check failed: Unable to connect...` y código de salida `1`. | **Aprobado** |
| **Recuperación Automática** (`docker compose start inference`) | Al restaurarse el contenedor, el healthcheck se vuelve verde, el status server responde en el puerto 5002, `check_alignment.py` vuelve a `SYNCHRONIZED` y las predicciones operan inmediatamente. | **Aprobado** |

---

## 8. Batería de Pruebas Automatizadas

Se ejecutaron las suites de pruebas unitarias y de integración tanto para el backend como para el proveedor del modelo:

### 8.1 Model Provider (`model_provider/tests/`)
- `test_check_alignment.py`: 7 pruebas (verificación sincronizada, redeploy requerido, estado degraded, fallos de red).
- `test_healthcheck.py`: 3 pruebas (vitalidad de servidor MLflow).
- `test_model_info.py`: 6 pruebas (inspección de metadatos y resolución de versión).
- `test_promote_model.py`: 9 pruebas (gobernanza, validación de candidato y promoción a champion).
- `test_start.py`: 15 pruebas (resolución de alias, configuración de entorno, servidor de estado HTTP en puerto secundario).
**Total**: **40 pasadas**, 0 fallidas (5.19s).

### 8.2 Backend (`backend/tests/`)
- `test_clients.py`: 8 pruebas (inyección de X-Request-ID, mapeo de errores HTTP, validaciones del cliente de inferencia).
- `test_health.py`: 5 pruebas (salud y metadatos del backend).
- `test_predictions.py`: 9 pruebas (contrato de 11 columnas, validación Pydantic, enriquecimiento de versión runtime, endpoints `/model` y `/status`).
**Total**: **22 pasadas**, 0 fallidas (4.35s).

---

## 9. Conclusión

El ciclo de servicio de inferencia está 100% operativo, cerrado e integrado de punta a punta: desde la interfaz gráfica de usuario en el frontend hasta el PyFunc del modelo salarial gobernado por MLflow en el backend.

```
MODEL_PROVIDER_STAGE_4_STATUS=READY_FOR_STAGE_5
```
