# SalaryPredict — Infraestructura MLOps y Serving

Módulo de infraestructura, gobernanza de modelos y servicio de inferencia para **SalaryPredict v1.0**. Provee el servidor central de MLflow, el contenedor de serving inmutable y las utilidades operacionales de promoción, detección de drift y operaciones que soportan el procedimiento operacional de rollback..

---

## 1. Responsabilidad

El módulo `model_provider/` se encarga exclusivamente de:
- Proveer el servidor persistente de **MLflow Tracking Server** y **Model Registry**.
- Gestionar la resolución y fijación del modelo **`salary_predict_model`** con el alias **`champion`**.
- Servir inferencias HTTP de forma inmutable mediante `mlflow models serve` (`:5001/invocations`).
- Exponer el estado y metadatos del runtime de inferencia en tiempo real (`:5002/status`).
- Ofrecer herramientas CLI para inspección de versiones, promoción, verificación de alineación que soportan el procedimiento operacional de rollback.

> **Frontera de responsabilidad**: `model_provider/` **NO entrena modelos**, no prepara variables de ingeniería ni evalúa candidatos; esas tareas corresponden a `ml/`.

---

## 2. Organización del Módulo

```text
model_provider/
├── tracking/            # Dockerfile y configuración del servidor MLflow (puerto 5000)
├── inference/           # Dockerfile, wrapper operacional start.py y healthchecks (puertos 5001 y 5002)
├── scripts/             # Herramientas CLI (promote_model.py, check_alignment.py, model_info.py)
├── config/              # Plantillas de variables de entorno complementarias
└── tests/               # Suite de pruebas unitarias de infraestructura y serving
```

---

## 3. Mecanismo de Serving e Inmutabilidad

El servicio de serving implementa una política estricta de estabilidad operativa:

1. **Resolución en Arranque**: Al inicializarse el contenedor (`start.py`), consulta el Model Registry para resolver qué versión numérica corresponde a `salary_predict_model@champion`.
2. **Fijación de Versión**: Lanza el servidor subyacente apuntando directamente a la URI inmutable `models:/salary_predict_model/<VERSION>`.
3. **Inmutabilidad en Runtime**: La versión cargada en memoria permanece invariable durante toda la vida del proceso, asegurando que ninguna promoción externa altere peticiones en curso.
4. **Status Server Integrado**: En paralelo, un servidor HTTP ligero en el puerto `5002` expone `GET /status` reportando la versión servida, PID, estado del proceso y hora de inicio.

> [!IMPORTANT]
> **Principio de gobierno**: `PROMOTE != DEPLOY`. Promover una versión en el Registry solo actualiza la asignación lógica del alias `champion`. Para aplicar el cambio al tráfico real, es indispensable reiniciar el servicio de inferencia (`docker compose restart inference`).

---

## 4. Scripts Operacionales Vigentes

Todas las herramientas operacionales operan contra el nombre canónico **`salary_predict_model`**:

### Inspección del Modelo y Versiones (`model_info.py`)
```bash
python model_provider/scripts/model_info.py --model salary_predict_model
```
Permite auditar el estado del modelo, listar todas las versiones registradas y conocer qué versión tiene actualmente asignado el alias `champion`.

### Promoción a Producción (`promote_model.py`)
```bash
python model_provider/scripts/promote_model.py \
  --model salary_predict_model \
  --version <VERSION> \
  --alias champion
```
Valida que la versión exista, tenga estado `READY` y cuente con el tag `eligible="true"` antes de asignarle el alias `champion`.

### Comprobación de Alineación Operacional (`check_alignment.py`)
```bash
python model_provider/scripts/check_alignment.py
```
Compara la versión del alias `champion` en el Registry contra la versión reportada por el status server (`:5002`):
- **Código 0 (`SYNCHRONIZED`)**: El contenedor sirve exactamente la versión champion actual.
- **Código 2 (`REDEPLOY REQUIRED`)**: Existe drift operacional; el Registry fue promovido pero el contenedor aún sirve la versión previa.
- **Código 1 (`RUNTIME UNHEALTHY`)**: El servicio de inferencia no responde o está caído.

### Redeploy Explícito
```bash
docker compose restart inference
```
Aplica de forma controlada la nueva versión promovida, descargando y recargando el artefacto en memoria (~5–15 s de ventana de inicialización).

---

## 5. Puertos y Servicios de Red

| Puerto | Servicio | Protocolo / Ruta | Acceso |
| :--- | :--- | :--- | :--- |
| **`5000`** | MLflow Tracking & Registry | HTTP / UI | Restringido al equipo de ingeniería |
| **`5001`** | MLflow Model Serving | HTTP POST `/invocations` | **Privado** (consumido solo por el Backend) |
| **`5002`** | Inference Status Server | HTTP GET `/status`, `/health` | **Privado** (consumido por Backend y scripts) |

---

## 6. Pruebas Automatizadas

Para validar los scripts operacionales, healthchecks y wrappers de inferencia:

```bash
pytest model_provider/tests
```

---

## 7. Fuera de Alcance del Módulo

- **Entrenamiento y optimización**: No ajusta hiperparámetros ni compila estimadores.
- **Selección de features y evaluación**: No calcula métricas sobre particiones ni calibra incertidumbre.
- **Lógica de negocio**: No valida peticiones de usuarios finales (responsabilidad del Backend).
- **Interfaz de usuario**: No provee componentes web (responsabilidad del Frontend).

Para la puesta en marcha completa del stack junto con backend y frontend, consulte [`../DEPLOYMENT.md`](../DEPLOYMENT.md).
