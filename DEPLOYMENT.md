# Guía Operacional de Despliegue — SalaryPredict v1.0

Manual operativo para levantar, reconstruir y validar **SalaryPredict v1.0** sin depender de conocimiento previo del monorepo.

La idea central es simple:

```text
ML / DVC
→ MLflow Tracking + Registry
→ alias champion
→ inference
→ backend
→ frontend
```

Home Analytics sigue un flujo separado:

```text
DVC analytics
→ dashboard_summary.json
→ publish-analytics
→ backend
→ frontend Home
```

> **Importante:** `publish-analytics` no forma parte de DVC y las métricas del snapshot analítico no identifican necesariamente al modelo `champion`.

---

## 1. Requisitos

### Para levantar la aplicación

- Git
- Docker Engine 24+
- Docker Compose v2

No se requiere Node.js en el host cuando se utiliza Docker Compose.

### Para reconstruir el modelo desde cero

Además:

- Python 3.11 recomendado (3.12 también soportado por el proyecto)
- entorno virtual Python
- dependencias de `ml/`
- acceso a los datos mediante DVC o copia manual de los snapshots CSV

Instalación del entorno ML:

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.lock.txt
cd ..
```

---

## 2. Configuración inicial: decidir una sola vez

Antes de elegir la modalidad de despliegue, configure el entorno y decida **qué servidor MLflow utilizará durante toda la sesión**.

### 2.1 Variables raíz

Desde la raíz:

```bash
cp .env.example .env
```

Valores principales:

```dotenv
MODEL_NAME=salary_predict_model
MODEL_ALIAS=champion
CORS_ORIGINS=http://localhost:5173,http://localhost:80,http://localhost
```

Si despliega en otra máquina, agregue su IP o dominio a `CORS_ORIGINS`.

### 2.2 Elegir MLflow local o remoto

Esta decisión es independiente de las modalidades A/B.

| Opción | Cuándo usarla | `MLFLOW_TRACKING_URI` desde el host |
|---|---|---|
| **MLflow local** | Quiere que este checkout levante Tracking/Registry | `http://localhost:5000` |
| **MLflow remoto** | Ya existe un servidor MLflow compartido | `http(s)://<MLFLOW_HOST>[:PUERTO]` |

Todos estos componentes deben apuntar al **mismo MLflow objetivo**:

```text
ml_pipeline track
ml_pipeline register-candidate
model_provider scripts
inference
```

`backend` y `frontend` no necesitan conectarse directamente a MLflow.

#### Si usa MLflow local

El servicio `mlflow-tracking` se levanta con Docker Compose y el host utiliza:

```dotenv
MLFLOW_TRACKING_URI=http://localhost:5000
```

Dentro de Docker, inference utiliza normalmente:

```text
http://mlflow-tracking:5000
```

#### Si usa MLflow remoto

Configure `ml/.env` con el servidor real:

```dotenv
MLFLOW_TRACKING_URI=https://<MLFLOW_HOST>
MLFLOW_EXPERIMENT_NAME=salary-prediction
MLFLOW_MODEL_NAME=salary_predict_model
ML_REQUIRE_CLEAN_GIT=false
```

Compruebe acceso antes de continuar:

```bash
export MLFLOW_TRACKING_URI=https://<MLFLOW_HOST>
python model_provider/scripts/model_info.py --model salary_predict_model
```

> **Advertencia:** cambiar solo `ml/.env` no garantiza que `inference` utilice el mismo servidor. Antes de trabajar con MLflow remoto, ejecute `docker compose config` y confirme la URI efectiva de `inference`. Si Compose fuerza `http://mlflow-tracking:5000`, deberá parametrizar esa configuración o usar un override antes de considerar soportado el modo remoto.

A partir de aquí, el manual se referirá simplemente al **MLflow objetivo** ya elegido. No vuelva a cambiarlo entre `track`, `register-candidate`, promoción e inference.

### 2.3 Home Analytics

El backend necesita un token compartido únicamente para proteger el POST administrativo de analytics.

Para una prueba local puede usar un valor simple:

```dotenv
ANALYTICS_PUBLISH_TOKEN=dev-analytics-token
ANALYTICS_STORAGE_PATH=/app/data/analytics_summary.json
```

Para un servidor compartido puede generar uno:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

El publisher de `ml/` debe usar el mismo token:

```dotenv
ANALYTICS_PUBLISH_URL=http://localhost:8000/api/v1/analytics/snapshots
ANALYTICS_PUBLISH_TOKEN=dev-analytics-token
ANALYTICS_PUBLISH_TIMEOUT_SECONDS=10
```

No versione secretos reales en Git.

---

## 3. Elegir modalidad de despliegue

Solo hay dos recorridos.

### Modalidad A — Ya existe `champion`

Use esta modalidad cuando el MLflow objetivo ya contiene:

```text
salary_predict_model@champion
```

Esto puede ocurrir porque:

- reutiliza volúmenes persistentes locales; o
- utiliza un MLflow remoto con el modelo ya registrado y promovido.

### Modalidad B — Bootstrap completo

Use esta modalidad cuando el MLflow objetivo todavía:

- no contiene `salary_predict_model`; o
- no tiene el alias `champion`.

En esta modalidad se reconstruyen datos/modelo, se registra una versión y se promueve explícitamente.

---

# 4. Modalidad A — Registry / Champion existente

## Paso 1. Clonar y validar configuración

```bash
git clone git@github.com:PDS-Microproyecto-Grupo-13/Microproyecto.git
cd Microproyecto

docker compose config
```

## Paso 2. Comprobar el MLflow objetivo

```bash
python model_provider/scripts/model_info.py --model salary_predict_model
```

Confirme que existe el alias:

```text
champion
```

Si no existe, utilice **Modalidad B**.

## Paso 3. Levantar la aplicación

### MLflow local

```bash
docker compose up -d --build
```

### MLflow remoto

No levante un Tracking Server local. Una vez confirmado que `inference` apunta al MLflow remoto:

```bash
docker compose up -d --build inference backend frontend
```

## Paso 4. Validar servicios

```bash
docker compose ps
```

Comprobaciones:

```bash
curl -s http://localhost:8000/api/v1/health
curl -s http://localhost:5002/status
```

Abra:

```text
http://localhost:5173
```

## Paso 5. Comprobar Home Analytics

```bash
curl -i http://localhost:8000/api/v1/analytics/summary
```

- `200`: Home Analytics ya está inicializado.
- `404 analytics_not_published`: siga la sección **6. Publicar Home Analytics**.

No necesita reconstruir el modelo solo porque falte analytics.

---

# 5. Modalidad B — Bootstrap completo desde una máquina limpia

Siga los pasos en orden.

> `inference` resuelve `champion` al arrancar. No lo levante antes de registrar y promover una versión.

## Paso 1. Preparar el MLflow objetivo

Si eligió MLflow local:

```bash
docker compose up -d --build mlflow-tracking
curl -sI http://localhost:5000/ | head -n 5
```

Debe responder HTTP 200.

Si eligió MLflow remoto, no levante `mlflow-tracking`; use directamente el servidor configurado en la sección 2.

## Paso 2. Preparar `ml/`

```bash
cd ml

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.lock.txt

cp .env.example .env
```

Verifique que `ml/.env` contiene el `MLFLOW_TRACKING_URI` elegido en la sección 2.

## Paso 3. Disponibilizar los datos

Elija **una** de estas alternativas.

### Opción 1 — DVC remote configurado

Desde `ml/`:

```bash
dvc pull
```

El pipeline no depende de un proveedor específico.

Implementaciones documentadas:

- Google Drive: [`docs/CONFIGURACION_DVC_GOOGLE_DRIVE.md`](ml/docs/CONFIGURACION_DVC_GOOGLE_DRIVE.md)
- Nginx + SSH:
  - [`docs/DVC_NGINX_SERVER.md`](ml/docs/DVC_NGINX_SERVER.md)
  - [`docs/DVC_NGINX_CONFIG.md`](ml/docs/DVC_NGINX_CONFIG.md)

La autenticación depende del remoto elegido.

### Opción 2 — Copia manual de snapshots

Si no desea conectarse a ningún servidor DVC, copie físicamente todos los:

```text
jobs_*.csv
```

requeridos por el checkout en:

```text
ml/data/raw/foorilla/
```

Todos los snapshots requeridos deben estar presentes. Si existe un puntero `.dvc` para un CSV que no está materializado, el preflight de `collect` abortará para evitar procesar un dataset parcial.

## Paso 4. Ejecutar el pipeline reproducible

Desde `ml/`:

```bash
dvc repro
```

El grafo actual contiene siete etapas:

```text
collect
→ validate
→ preprocess
→ qualify
→ train
→ evaluate
→ analytics
```

Resumen:

1. `collect`: consolida y deduplica snapshots.
2. `validate`: valida esquema e invariantes.
3. `preprocess`: prepara población modelable y split temporal 70/15/15.
4. `qualify`: compara baseline, verifica gap temporal y calibra incertidumbre.
5. `train`: ajusta el modelo final LightGBM.
6. `evaluate`: evalúa sobre test ciego y genera reportes.
7. `analytics`: genera `artifacts/reports/dashboard_summary.json`.

`analytics` no publica datos fuera de `ml/`.

> **Regla:** `dvc repro != publish-analytics`.

## Paso 5. Registrar la corrida en MLflow

```bash
python -m ml_pipeline track
```

Esto registra métricas, parámetros, reportes y el modelo PyFunc.

## Paso 6. Registrar la versión candidata

```bash
python -m ml_pipeline register-candidate
```

Esto crea una nueva versión en Model Registry, pero **no la promueve**.

## Paso 7. Inspeccionar la versión

Desde la raíz:

```bash
cd ..
python model_provider/scripts/model_info.py --model salary_predict_model
```

Identifique la versión candidata:

```text
Version <N>
```

## Paso 8. Promover a `champion`

```bash
python model_provider/scripts/promote_model.py   --model salary_predict_model   --version <VERSION>   --alias champion
```

## Paso 9. Levantar aplicación

Una vez que `salary_predict_model@champion` existe:

```bash
docker compose up -d --build inference backend frontend
```

Si utiliza MLflow local y `mlflow-tracking` ya estaba iniciado, permanecerá activo.

## Paso 10. Verificar alineación

```bash
python model_provider/scripts/check_alignment.py
```

Resultado esperado:

```text
Alignment Status: SYNCHRONIZED
Registry Version:       <VERSION>
Runtime Loaded Version: <VERSION>
```

Código de salida esperado:

```text
0
```

## Paso 11. Publicar Home Analytics

Continúe con la sección siguiente.

---

# 6. Publicar Home Analytics

Este paso es independiente de MLflow.

Compruebe primero:

```bash
curl -i http://localhost:8000/api/v1/analytics/summary
```

Si obtiene `200`, ya existe un snapshot y puede continuar.

Si obtiene:

```text
404 analytics_not_published
```

publique el artifact existente.

Desde `ml/`:

```bash
ANALYTICS_PUBLISH_URL=http://localhost:8000/api/v1/analytics/snapshots ANALYTICS_PUBLISH_TOKEN='dev-analytics-token' python -m ml_pipeline publish-analytics
```

Use el mismo token configurado en el backend.

Después:

```bash
curl -s http://localhost:8000/api/v1/analytics/summary
```

Debe responder HTTP 200 e incluir:

```json
{
  "schema_version": "1.0"
}
```

Consideraciones:

- si `dashboard_summary.json` ya existe, no necesita ejecutar ML nuevamente;
- si falta, ejecute:

```bash
dvc repro analytics
```

- repetir la misma publicación es seguro y devuelve `unchanged`;
- frontend no necesita reinicio: recargue Home o utilice su botón de reintento;
- las métricas `model` del snapshot son métricas de evaluación, no identifican al `champion`.

El snapshot publicado se conserva en el volumen:

```text
mlops-backend-analytics-data
```

Sobrevive a restart, recreación y `docker compose down/up`.

Se elimina con:

```bash
docker compose down -v
```

En ese caso basta volver a publicar un artifact existente.

---

# 7. Resumen rápido: Modalidad A vs Modalidad B

| Pregunta | Modalidad A | Modalidad B |
|---|---|---|
| ¿Existe `salary_predict_model@champion`? | Sí | No |
| ¿Ejecuta DVC pipeline? | No, salvo que necesite regenerar analytics | Sí |
| ¿Ejecuta `track`? | No | Sí |
| ¿Ejecuta `register-candidate`? | No | Sí |
| ¿Promueve una versión? | No | Sí |
| ¿Puede requerir `publish-analytics`? | Sí, si backend está vacío | Sí, en primer bootstrap |
| Resultado final | reutiliza un modelo existente | construye y registra uno nuevo |

En ambos casos, MLflow local/remoto se decide **una sola vez en la sección 2** y todos los componentes deben apuntar al mismo servidor.

---

# 8. Prueba End-to-End de predicción

Con la aplicación activa:

```bash
curl -s -X POST http://localhost:8000/api/v1/predictions   -H "Content-Type: application/json"   -d '{
    "title": "Data Scientist",
    "experience_level": "SE",
    "experience_years": 6,
    "country": "Colombia",
    "is_remote": true,
    "company": "Example Corp",
    "company_is_agency": false,
    "technologies": ["Python", "SQL", "AWS"]
  }'
```

Respuesta esperada:

```text
{
  "prediction": {
    "minimum_usd": <MINIMUM_USD>,
    "maximum_usd": <MAXIMUM_USD>,
    "midpoint_usd": <MIDPOINT_USD>
  },
  "model": {
    "name": "salary_predict_model",
    "alias": "champion",
    "version": "<VERSION>"
  },
  "warnings": []
}
```

También valide Home:

```text
http://localhost:5173
```

Debe mostrar estadísticas reales o, si aún no fueron publicadas, el estado explícito “Analítica aún no publicada”.

---

# 9. Operación: promoción, drift y rollback

La regla operacional sigue siendo:

```text
PROMOTE != DEPLOY
```

`inference` resuelve `champion` solo al arrancar.

## Promover otra versión

```bash
python model_provider/scripts/promote_model.py   --model salary_predict_model   --version <NUEVA_VERSION>   --alias champion
```

## Detectar drift

```bash
python model_provider/scripts/check_alignment.py
```

- exit `0`: runtime sincronizado.
- exit `2`: Registry apunta a otra versión y requiere redeploy.
- exit `1`: runtime no saludable o error.

## Desplegar el nuevo champion

```bash
docker compose restart inference
python model_provider/scripts/check_alignment.py
```

## Rollback

```bash
python model_provider/scripts/promote_model.py   --model salary_predict_model   --version <VERSION_ANTERIOR>   --alias champion

docker compose restart inference

python model_provider/scripts/check_alignment.py
```

---

# 10. Despliegue en servidor remoto / EC2

Docker Compose continúa siendo el mecanismo canónico.

## CORS

Ejemplo:

```dotenv
CORS_ORIGINS=http://localhost:5173,http://localhost:80,http://localhost,http://<IP_PUBLICA>:5173
```

## Puertos

| Puerto | Exposición recomendada | Propósito |
|---|---|---|
| `5173` | Pública | Frontend |
| `8000` | Opcional / restringida | API backend |
| `5000` | Restringida | MLflow UI |
| `5001` | Cerrada al exterior | inference |
| `5002` | Cerrada al exterior | runtime status |

Si MLflow UI se expone por IP/dominio, puede ser necesario configurar `MLFLOW_ALLOWED_HOSTS` según el servidor utilizado.

## Persistencia

Mantenga almacenamiento persistente para:

```text
mlops-mlflow-db-data
mlops-mlflow-artifact-data
mlops-backend-analytics-data
```

En EC2, use disco raíz persistente o EBS no efímero.

---

# 11. Troubleshooting

| Síntoma | Causa probable | Acción |
|---|---|---|
| `alias_lookup_failed` | No existe `salary_predict_model@champion` en el MLflow objetivo | Use Modalidad B o corrija el servidor MLflow seleccionado |
| `Inference container unhealthy` | El modelo tarda en inicializar o hay error runtime | `docker compose logs inference` |
| Backend devuelve 502 | Backend no alcanza `inference:5001` | Revise `docker compose ps` y logs |
| Puerto ocupado | Otro proceso usa 5000/8000/5173 | Libere el puerto o ajuste Compose |
| `libgomp.so.1` ausente | LightGBM se ejecuta directamente en Linux sin OpenMP | `sudo apt-get install -y libgomp1` |
| `champion` existe pero inference no lo encuentra | Componentes apuntan a distintos MLflow | Revise `MLFLOW_TRACKING_URI` efectivo con `docker compose config` |
| Home muestra “Analítica aún no publicada” | Backend no tiene snapshot | Ejecute `publish-analytics` |
| `publish-analytics` devuelve 401 | Tokens diferentes | Use el mismo `ANALYTICS_PUBLISH_TOKEN` en backend y publisher |
| `dvc pull` no es posible | Remoto no disponible/credenciales | Use la copia manual de todos los `jobs_*.csv` requeridos |
