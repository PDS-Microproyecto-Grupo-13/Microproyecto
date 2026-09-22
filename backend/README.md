# SalaryPredict — FastAPI Backend Microservice

Microservicio backend base en Python con FastAPI para la plataforma MLOps **SalaryPredict**. Diseñado con una arquitectura modular y desacoplada, preparado para orquestar peticiones hacia el servicio de inferencia de modelos ML vía HTTP.

---

## 1. Requisitos Previos

* **Python 3.12** (o 3.11+)
* **Docker** (opcional para despliegue en contenedores)

---

## 2. Puesta en Marcha Local (Desarrollo)

### Crear entorno virtual

```bash
python -m venv .venv
```

### Activar entorno virtual

* **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

* **Windows (PowerShell / CMD):**
  ```powershell
  .venv\Scripts\activate
  ```

### Instalar dependencias de desarrollo

```bash
pip install -r requirements-dev.lock.txt
```

### Configurar variables de entorno

```bash
cp .env.example .env
```

### Iniciar el servidor de desarrollo

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 3. Endpoints y Documentación

* **Healthcheck:** `GET http://localhost:8000/api/v1/health`
* **Swagger UI (OpenAPI):** `http://localhost:8000/docs`
* **Esquema OpenAPI JSON:** `http://localhost:8000/openapi.json`
* **Predicción salarial:** `POST http://localhost:8000/api/v1/predictions`
* **Selector del modelo:** `GET http://localhost:8000/api/v1/predictions/model`
* **Estado del runtime:** `GET http://localhost:8000/api/v1/predictions/status`

### Ejemplo de respuesta de Healthcheck

```json
{
  "status": "ok",
  "service": "mlops-backend",
  "version": "0.1.0"
}
```

---

## 4. Tests y Calidad de Código

### Ejecutar tests automatizados

```bash
pytest -v
```

### Ejecutar linter y formateador (Ruff)

```bash
ruff check .
ruff format --check .
```

---

## 5. Gestión y Regeneración de Locks (pip-tools)

Para actualizar o sincronizar los archivos lock reproducibles:

```bash
# Regenerar lock de producción
pip-compile requirements.txt --output-file=requirements.lock.txt

# Regenerar lock de desarrollo
pip-compile requirements-dev.txt --output-file=requirements-dev.lock.txt
```

---

## 6. Ejecución con Docker

### Construir imagen

```bash
docker build -t mlops-backend .
```

### Ejecutar contenedor

```bash
docker run --rm -p 8000:8000 --env-file .env mlops-backend
```

---

## 7. Arquitectura, Contratos y Fronteras

### Interacción con el Servicio de Inferencia

El backend no carga artefactos de ML directamente (`model.joblib` permanece en el servicio de inferencia). Su función es orquestar y validar:

1. **Traducción de Contrato**: Recibe el payload JSON del frontend y lo traduce a la estructura `dataframe_split` requerida por el wrapper PyFunc de MLflow.
2. **Inferencia HTTP (`:5001`)**: Envía la solicitud a `POST http://inference:5001/invocations`.
3. **Sonda de Estado (`:5002`)**: Consulta de forma no bloqueante `GET http://inference:5002/status` para adjuntar la versión concreta cargada en memoria (`loaded_version`).
4. **Validación de Invariantes**: Comprueba que las salidas salariales sean valores finitos, positivos, que el mínimo no supere al máximo y que el punto medio coincida con $(y_{min} + y_{max}) / 2$.

### Modelo Canónico

El backend opera contra el modelo canónico registrado:
- **`MODEL_NAME`**: `salary_predict_model`
- **`MODEL_ALIAS`**: `champion`

### Fuera de Alcance del Backend

- **NO entrena modelos**: El ciclo de entrenamiento y evaluación pertenece a `ml/`.
- **NO carga modelos en memoria**: La ejecución del wrapper PyFunc reside en `model_provider/inference`.
- **NO promueve modelos**: La asignación del alias `champion` corresponde a `model_provider/scripts/promote_model.py`.
- **NO sirve interfaces de usuario**: La capa visual corresponde a `frontend/`.

Para la orquestación completa del monorepo, consulte [`../README.md`](../README.md) y [`../DEPLOYMENT.md`](../DEPLOYMENT.md).
