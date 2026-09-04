# Despliegue de SalaryPredict

La integración mantiene estable el contrato `frontend -> backend -> MLflow serving`.
Los experimentos generan versiones inmutables del modelo registrado
`salary_predict_model`; el alias `champion` decide qué versión se sirve.

## 1. Preparar el entorno de entrenamiento

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.venv\Scripts\python.exe -m pip install -r ml\catboost_baseline\requirements.txt
dvc pull
```

## 2. Iniciar MLflow

```powershell
docker compose up -d --build mlflow-tracking
```

La interfaz queda disponible en `http://localhost:5000`. El servidor recibe los
artefactos por HTTP y los conserva en volúmenes Docker.

## 3. Entrenar, empaquetar y registrar

Para registrar una versión sin desplegarla:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
.venv\Scripts\python.exe -m ml.catboost_baseline.run `
  --register-model `
  --run-name catboost-candidato
```

El paquete MLflow incluye los dos modelos CatBoost, la transformación de variables,
la firma del contrato y las versiones de sus dependencias. Por eso el servicio de
inferencia no necesita conocer la implementación de cada nuevo experimento.

Para el primer despliegue también puede asignarse el alias durante el entrenamiento:

```powershell
.venv\Scripts\python.exe -m ml.catboost_baseline.run `
  --register-model `
  --model-alias champion `
  --run-name catboost-inicial
```

## 4. Promover un experimento validado

Compare las métricas de validación en MLflow. No use el conjunto de prueba para
escoger hiperparámetros. Después promueva la versión elegida:

```powershell
.venv\Scripts\python.exe model_provider\scripts\promote_model.py `
  --model salary_predict_model `
  --version VERSION `
  --alias champion `
  --tracking-uri http://localhost:5000
```

## 5. Desplegar el producto

```powershell
docker compose up -d --build inference backend frontend
```

Servicios:

- Tablero: `http://localhost:5173`
- API y Swagger: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`
- Inferencia interna: `http://localhost:5001/invocations`

Prueba directa del API público:

```powershell
$body = @{
  title = "Data Scientist"
  experience_level = "SE"
  experience_years = 6
  country = "Colombia"
  is_remote = $true
  company = "Example Corp"
  company_is_agency = $false
  technologies = @("Python", "SQL", "AWS")
  topics = @("Data Science", "Machine Learning")
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/predictions `
  -ContentType application/json -Body $body
```

## 6. Desplegar una versión posterior

1. Ejecute un nuevo experimento con `--register-model`, sin alias.
2. Compare el candidato con el campeón usando métricas de validación y segmentos.
3. Promueva la versión con `promote_model.py`.
4. Reinicie solo inferencia: `docker compose restart inference`.
5. Ejecute la prueba del API. Backend y frontend no requieren cambios.

El servicio resuelve `champion` al arrancar y fija esa versión durante toda su vida.
Esto evita que una promoción cambie silenciosamente un proceso que ya está atendiendo
solicitudes y permite una reversión rápida reasignando el alias a la versión anterior.

## Verificación local

```powershell
.venv\Scripts\python.exe -m pytest -q ml\catboost_baseline\tests
Push-Location backend
..\.venv\Scripts\python.exe -m pytest -q
Pop-Location
Push-Location frontend
npm ci
npm run lint
npm run build
Pop-Location
docker compose config
```
