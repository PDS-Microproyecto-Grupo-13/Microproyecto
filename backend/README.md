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

El backend no carga artefactos de ML directamente. Traduce el contrato público al
protocolo de MLflow y consulta `INFERENCE_BASE_URL`. `MODEL_NAME` y `MODEL_ALIAS`
identifican el selector desplegado que se devuelve al tablero. Consulte
[`../DEPLOYMENT.md`](../DEPLOYMENT.md) para ejecutar el flujo completo.
