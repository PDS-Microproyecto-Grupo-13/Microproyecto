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
* **Predicción Breast Cancer:** `POST http://localhost:8000/api/v1/predictions`
* **Swagger UI (OpenAPI):** `http://localhost:8000/docs`
* **Esquema OpenAPI JSON:** `http://localhost:8000/openapi.json`

### Ejemplo de respuesta de Healthcheck

```json
{
  "status": "ok",
  "service": "mlops-backend",
  "version": "0.1.0"
}
```

### Ejemplo de predicción

```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    "mean_perimeter": 122.8,
    "mean_area": 1001.0,
    "mean_smoothness": 0.1184,
    "mean_compactness": 0.2776,
    "mean_concavity": 0.3001,
    "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871,
    "radius_error": 1.095,
    "texture_error": 0.9053,
    "perimeter_error": 8.589,
    "area_error": 153.4,
    "smoothness_error": 0.006399,
    "compactness_error": 0.04904,
    "concavity_error": 0.05373,
    "concave_points_error": 0.01587,
    "symmetry_error": 0.03003,
    "fractal_dimension_error": 0.006193,
    "worst_radius": 25.38,
    "worst_texture": 17.33,
    "worst_perimeter": 184.6,
    "worst_area": 2019.0,
    "worst_smoothness": 0.1622,
    "worst_compactness": 0.6656,
    "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654,
    "worst_symmetry": 0.4601,
    "worst_fractal_dimension": 0.1189
  }'
```

```json
{"prediction": 0}
```

La ruta recibe las 30 features con nombres `snake_case` y delega su adaptación a
`BreastCancerPredictionService`. Este utiliza `InferenceService`, el cliente HTTP
genérico del único proveedor `model_provider/inference`. El backend no carga
modelos ni consulta MLflow Model Registry.

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
