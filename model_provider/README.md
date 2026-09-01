# SalaryPredict — Infraestructura MLOps (MLflow + Docker Compose)

Infraestructura base del módulo `model_provider/` para la plataforma **SalaryPredict**, proporcionando **MLflow Tracking Server**, **Model Registry**, **MLflow Model Serving (Inference)**, scripts de promoción/inspección y orquestación reproducible con **Docker Compose**.

---

## 1. Arquitectura MLOps

```text
                       ml/ (Entrenamiento & Registro)
                                     │
                                     │ MLflow Client
                                     ▼
               ┌──────────────────────────────────────────┐
               │         mlflow-tracking (5000)           │
               │   • Tracking Server & UI                 │
               │   • Model Registry                       │
               │   • SQLite: /var/lib/mlflow/db           │
               │   • Artifacts: /var/lib/mlflow/artifacts │
               └─────────────────────┬────────────────────┘
                                     │
                                     │ model: salary_predict_model
                                     │ alias: champion (v1 -> v2)
                                     ▼
               ┌──────────────────────────────────────────┐
               │            inference (5001)              │
               │   • start.py (Model Resolver)            │
               │   • MLflow Model Serving                 │
               │   • Inmutable: Requiere reinicio         │
               └─────────────────────┬────────────────────┘
                                     │
                                     │ HTTP (POST /invocations)
                                     ▼
               ┌──────────────────────────────────────────┐
               │              backend (8000)              │
               │   • FastAPI Microservice                 │
               └─────────────────────┬────────────────────┘
                                     │
                                     │ HTTP
                                     ▼
               ┌──────────────────────────────────────────┐
               │             frontend (5173)              │
               │   • React + TypeScript Dashboard         │
               └──────────────────────────────────────────┘
```

### Separación de Responsabilidades

* **`model_provider/tracking/`**: Servidor de tracking y Model Registry persistente sobre SQLite y volumen de artefactos.
* **`model_provider/inference/`**: Servidor oficial de inferencia (`mlflow models serve`) con wrapper operacional (`start.py`) que resuelve el alias `champion` a una versión concreta al arrancar.
* **`model_provider/scripts/`**: Utilidades CLI para promoción de versiones a alias e inspección de metadatos.
* **`model_provider/dev/`**: Utilidad bootstrap para registrar modelos demo con el fin de validar el ciclo end-to-end.
* **`model_provider/config/`**: Plantillas de variables de entorno para tracking e inferencia.

---

## 2. Persistencia de Datos

Para garantizar que los datos sobreviven al ciclo de vida de los contenedores, se utilizan volúmenes nombrados de Docker:

| Recurso | Tipo de Almacenamiento | Ruta en Contenedor | Volumen Docker |
| :--- | :--- | :--- | :--- |
| **Metadatos & Runs** | SQLite (`mlflow.db`) | `/var/lib/mlflow/db` | `mlops-mlflow-db-data` |
| **Artefactos del Modelo** | Filesystem Local | `/var/lib/mlflow/artifacts` | `mlops-mlflow-artifact-data` |

> [!NOTE]
> El contenedor de `inference` monta `mlops-mlflow-artifact-data` en modo solo lectura (`:ro`) para cargar artefactos directamente sin duplicar almacenamiento.

---

## 3. Guía de Puesta en Marcha y Verificación End-to-End

### Paso 1: Arrancar el Servidor de MLflow Tracking

```bash
docker compose up -d mlflow-tracking
```

* **MLflow UI:** `http://localhost:5000`
* **Healthcheck:** El contenedor esperará hasta responder HTTP 200 en `http://localhost:5000/`.

---

### Paso 2: Registrar el Modelo Demo (Versión 1)

Ejecuta el script de demostración para entrenar un modelo sintético en memoria, registrar el run en MLflow y asignarle el alias `champion`:

```bash
python model_provider/dev/register_demo_model.py --version-tag v1
```

Salida esperada:
```text
✓ Model logged to run ID: <run_id>
✓ Registered Model Version: 1
✓ Alias 'champion' assigned to Version 1
```

---

### Paso 3: Iniciar el Servicio de Inferencia

```bash
docker compose up -d inference
```

Al iniciar, `start.py` resolverá el alias `champion` (versión 1) y lanzará el servidor de inferencia.

Inspecciona los logs operacionales:
```bash
docker compose logs -f inference
```

Salida esperada:
```text
[INFO] service=inference event=tracking_connected tracking_uri=http://mlflow-tracking:5000
[INFO] service=inference event=model_resolved model=salary_predict_model alias=champion version=1 run_id=...
[INFO] service=inference event=model_server_starting model=salary_predict_model version=1
```

---

### Paso 4: Verificar Healthcheck de Inferencia

```bash
docker compose ps inference
```
El estado debe indicar `(healthy)`.

---

### Paso 5: Realizar una Predicción de Prueba (`/invocations`)

Envía un payload de ejemplo en formato `dataframe_split`:

```bash
curl -s -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["years_experience", "is_remote", "skills_count"],
      "data": [
        [5.0, 1, 4],
        [2.0, 0, 2]
      ]
    }
  }'
```

Respuesta esperada:
```json
{"predictions": [78300.0, 46800.0]}
```

---

### Paso 6: Registrar una Nueva Versión (Versión 2)

Genera una nueva versión del modelo (por ejemplo, con un algoritmo `GradientBoostingRegressor`):

```bash
python model_provider/dev/register_demo_model.py --version-tag v2 --no-set-champion
```

Salida:
```text
✓ Registered Model Version: 2
```

---

### Paso 7: Promover la Versión 2 a `champion`

Utiliza el script administrativo `promote_model.py`:

```bash
python model_provider/scripts/promote_model.py \
  --model salary_predict_model \
  --version 2 \
  --alias champion
```

Salida esperada en logs:
```text
[INFO] event=model_promotion_started model=salary_predict_model version=2 alias=champion
[INFO] event=previous_alias model=salary_predict_model alias=champion version=1
[INFO] event=model_promoted model=salary_predict_model alias=champion from_version=1 to_version=2
```

---

### Paso 8: Comprobar que NO hay Hot Reload Inesperado

Realiza una petición a `/invocations` sin reiniciar el contenedor: el servidor **continúa sirviendo la versión 1** de manera inmutable y segura.

---

### Paso 9: Reiniciar Inferencia para Desplegar el Nuevo Champion

```bash
docker compose restart inference
```

Inspecciona los logs para confirmar el nuevo despliegue:
```bash
docker compose logs -f inference
```

Salida:
```text
[INFO] service=inference event=model_resolved model=salary_predict_model alias=champion version=2 run_id=...
[INFO] service=inference event=model_server_starting model=salary_predict_model version=2
```

---

## 4. Scripts Administrativos

### Inspeccionar Metadatos del Modelo (`model_info.py`)

```bash
# Ver información general y todas las versiones registradas
python model_provider/scripts/model_info.py --model salary_predict_model

# Resolver qué versión tiene actualmente el alias 'champion'
python model_provider/scripts/model_info.py --model salary_predict_model --alias champion

# Inspeccionar una versión concreta
python model_provider/scripts/model_info.py --model salary_predict_model --version 2
```

### Promover Modelo (`promote_model.py`)

```bash
python model_provider/scripts/promote_model.py \
  --model salary_predict_model \
  --version <VERSION_NUM> \
  --alias champion \
  --tracking-uri http://localhost:5000
```

---

## 5. Ejecución del Stack Completo

Para iniciar todos los servicios (`mlflow-tracking`, `inference`, `backend`, `frontend`):

```bash
docker compose up -d
```

| Servicio | URL / Endpoint | Descripción |
| :--- | :--- | :--- |
| **Frontend** | `http://localhost:5173` | React Dashboard UI |
| **Backend** | `http://localhost:8000/docs` | FastAPI REST API (Swagger Docs) |
| **MLflow UI** | `http://localhost:5000` | Experimentos y Model Registry |
| **Inference** | `http://localhost:5001/invocations` | MLflow Model Serving |

---

## 6. Testing

Ejecutar la suite de tests unitarios del módulo MLOps:

```bash
pytest model_provider/tests -v
```

---

## 7. Troubleshooting

| Problema | Causa Probable | Solución |
| :--- | :--- | :--- |
| `Unable to connect to MLflow Tracking Server` | El contenedor `mlflow-tracking` no ha iniciado o no está saludable. | Verificar logs con `docker compose logs mlflow-tracking` y confirmar que responde en el puerto 5000. |
| `Registered model '...' does not exist` | El modelo aún no ha sido registrado en MLflow Model Registry. | Ejecutar `python model_provider/dev/register_demo_model.py` antes de iniciar el contenedor de inferencia. |
| `Alias 'champion' is not configured` | El modelo existe pero no tiene asignado el alias objetivo. | Promover una versión con `python model_provider/scripts/promote_model.py --model <NAME> --version 1 --alias champion`. |
| `Inference container unhealthy` | El modelo tardó en cargar o falló la verificación de `/health`. | Revisar `docker compose logs inference` y el log de `healthcheck.py`. |
| `Port conflict on 5000 / 5001` | Otro proceso local está ocupando el puerto. | Modificar `ports` en `docker-compose.yml` o detener el proceso en conflicto. |
