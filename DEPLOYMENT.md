# Despliegue de SalaryPredict

La integración mantiene estable el contrato `frontend -> backend -> MLflow serving`.
Los experimentos generan versiones inmutables del modelo registrado
`salary_predict_model`; el alias `champion` decide qué versión se sirve.

Hay dos formas de levantar el producto. Sirven el mismo modelo; lo que cambia es
quién arranca los procesos.

| Ruta | Cuándo usarla |
| --- | --- |
| **Rápida, sin Docker** (sección 2) | Ver el tablero funcionando contra el MLflow del grupo. Es la que se usó para la Entrega 2. |
| **Docker Compose** (sección 6) | Levantar el stack completo, MLflow incluido, en una sola máquina. |

## 1. Dónde vive MLflow

El servidor de seguimiento del grupo corre en una instancia EC2 y **su IP pública
cambia cada vez que se reinicia el laboratorio**. Por eso la dirección no está en
el código: se lee de la variable de entorno `MLFLOW_TRACKING_URI`.

```powershell
Copy-Item .env.example .env
notepad .env    # poner la IP vigente
```

`.env` está en `.gitignore`: no se sube al repositorio y cada quien mantiene el
suyo. Si no hay servidor remoto disponible, deje la variable sin definir y todo
apuntará al MLflow local que levanta Docker Compose.

Para comprobar que el servidor responde, abra `http://<IP>:5000` en el navegador.

## 2. Levantar el tablero sin Docker (ruta rápida)

Requisitos previos: el entorno virtual de la sección 3 creado, y `npm install`
ejecutado dentro de `frontend/`.

Son **tres procesos en tres terminales**, en este orden. Cada terminal queda
ocupada mientras el proceso corre: no la cierre ni lo interrumpa.

### Terminal 1 — servicio de inferencia (puerto 5001)

```powershell
$env:PATH = "$PWD\.venv\Scripts;" + $env:PATH
$env:MLFLOW_TRACKING_URI = "http://<IP>:5000"
$env:MODEL_NAME = "salary_predict_model"
$env:MODEL_ALIAS = "champion"
$env:INFERENCE_HOST = "127.0.0.1"
.venv\Scripts\python.exe model_provider\inference\start.py
```

El script resuelve el alias contra el Model Registry, fija la versión concreta
que le corresponde y la sirve. En su salida queda registrado qué versión quedó
atendiendo, que es la forma de confirmar que se está sirviendo lo que se cree.

La primera línea antepone el entorno virtual al `PATH`. Es necesaria porque
MLflow lanza un proceso hijo de `uvicorn` buscándolo en el `PATH`; si hay una
instalación de Anaconda por delante, Windows resuelve la de conda y el proceso
falla con `ModuleNotFoundError: No module named 'mlflow'`.

Arrancar tarda cerca de un minuto: hay que descargar el modelo empaquetado desde
MLflow y cargarlo en memoria. Espere a que anuncie que está escuchando.

### Terminal 2 — API del proyecto (puerto 8000)

```powershell
$env:INFERENCE_BASE_URL = "http://127.0.0.1:5001"
Push-Location backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

`INFERENCE_BASE_URL` no se puede omitir. El valor por defecto es
`http://inference:5001`, que es el nombre del servicio **dentro de la red de
Docker**; corriendo a mano ese nombre no existe.

Y tiene que decir `127.0.0.1`, no `localhost`. La terminal 1 deja el servicio
escuchando solo en IPv4, y en Windows `localhost` resuelve primero a `::1`, la
direccion IPv6, donde no hay nadie escuchando. El sintoma es que el tablero
responde `Server disconnected without sending a response` mientras
`http://127.0.0.1:5001/ping` devuelve 200 sin problema.

### Terminal 3 — tablero (puerto 5173)

```powershell
Push-Location frontend
npm run dev
```

Abra `http://localhost:5173`.

### Qué queda escuchando

| Puerto | Proceso | Quién le habla |
| --- | --- | --- |
| 5173 | Tablero (Vite) | El navegador |
| 8000 | API del proyecto (FastAPI) | El tablero, vía el proxy de Vite |
| 5001 | Inferencia (MLflow serving) | Sólo la API |
| 5000 | MLflow en EC2 | Inferencia, al arrancar; y el navegador para ver experimentos |

El tablero no escribe en ninguna parte la dirección de la API: pide `/api/...`,
una ruta relativa, y `frontend/vite.config.ts` la reenvía al puerto 8000. En
Docker el mismo papel lo cumple `frontend/nginx.conf`. Por eso el código del
tablero es idéntico en las dos rutas.

### Detalles que ahorran tiempo

- **El orden importa al usar el tablero, no al arrancarlo.** La API no busca al
  servicio de inferencia hasta que llega una petición; si no está arriba, la
  predicción devuelve 502 y el tablero muestra el error.
- **EC2 sólo hace falta al arrancar la terminal 1.** Una vez cargado el modelo en
  memoria, se puede apagar la instancia y el tablero sigue prediciendo. Útil
  cuando los créditos del laboratorio están escasos.
- **Si la IP de EC2 cambió**, hay que actualizarla en `.env` y en la variable de
  la terminal 1, y volver a arrancar ese proceso.

## 3. Preparar el entorno de entrenamiento

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.venv\Scripts\python.exe -m pip install -r ml\catboost_baseline\requirements.txt
dvc pull
```

El proyecto tiene **dos familias de modelos** que compiten por el mismo alias:

| Paquete | Estado |
| --- | --- |
| `ml/catboost_baseline` | En esta rama. Produjo las versiones 1 y 2. |
| `ml/lightgbm_baseline` | En la rama `feature/lightgbm-baseline`, pendiente de fusionar. Produjo las versiones 3 a 6. |

**Para levantar el tablero no importa en qué rama esté.** El modelo que sirve la
sección 2 viaja empaquetado dentro de MLflow con su propio código y sus
dependencias fijadas; el servicio de inferencia no necesita tener a la vista el
paquete que lo entrenó. La rama sólo importa para *entrenar* un modelo nuevo.

## 4. Entrenar, empaquetar y registrar

Para registrar una versión sin desplegarla:

```powershell
$env:MLFLOW_TRACKING_URI = "http://<IP>:5000"
.venv\Scripts\python.exe -m ml.catboost_baseline.run `
  --register-model `
  --run-name catboost-candidato
```

El paquete MLflow incluye los dos modelos entrenados, la transformación de
variables, la firma del contrato y las versiones de sus dependencias. Por eso el
servicio de inferencia no necesita conocer la implementación de cada nuevo
experimento.

Para el primer despliegue también puede asignarse el alias durante el
entrenamiento:

```powershell
.venv\Scripts\python.exe -m ml.catboost_baseline.run `
  --register-model `
  --model-alias champion `
  --run-name catboost-inicial
```

Desde la rama de LightGBM el comando es el mismo cambiando el módulo por
`ml.lightgbm_baseline.run`.

## 5. Promover un experimento validado

Compare las métricas de **validación** en MLflow. No use el conjunto de prueba
para escoger hiperparámetros: si se elige mirando la prueba, la prueba deja de
medir generalización. Después promueva la versión elegida:

```bash
python model_provider/scripts/promote_model.py \
  --model salary-predictor \
  --version VERSION \
  --alias champion \
  --tracking-uri http://localhost:5000
```

Promover es mover una etiqueta en el Registry; no altera procesos en ejecución (`PROMOTE != DEPLOY`). Para que el cambio surta efecto hay que reiniciar explícitamente el servicio de inferencia: resuelve el alias **una sola vez, al arrancar**, y sirve esa versión durante toda su vida. Esto evita que una promoción cambie en silencio un proceso en producción y garantiza la inmutabilidad de serving.

## 6. Levantar todo con Docker Compose

```powershell
docker compose up -d --build mlflow-tracking
docker compose up -d --build inference backend frontend
```

Servicios:

- Tablero: `http://localhost:5173`
- API y Swagger: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`
- Inferencia interna: `http://localhost:5001/invocations`
- Inferencia status runtime: `http://localhost:5002/status`

El servicio de inferencia lee `MLFLOW_TRACKING_URI` del entorno, así que también
puede apuntarse al MLflow de EC2 definiendo la variable en `.env` antes de
levantar el contenedor.

## 7. Probar la API sin el tablero

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

Devuelve el rango mínimo, máximo y punto medio, junto con el nombre, alias
y versión del modelo que respondió.

## 8. Ciclo Operacional de Despliegue, Drift y Rollback

El servicio de serving fija la versión concreta del modelo al momento de arrancar (`start.py`) y expone su estado runtime en el puerto `5002`.

```
Promoción
   ↓ (promote_model.py)
Registry cambia champion
   ↓
Comprobación
   ↓ (check_alignment.py -> redeploy_required, exit code 2)
Redeploy explícito
   ↓ (docker compose restart inference)
Verificación
   ↓ (check_alignment.py -> synchronized, exit code 0)
Rollback (si se requiere revertir)
   ↓ (promote_model.py a versión previa)
   ↓ (check_alignment.py -> redeploy_required)
   ↓ (docker compose restart inference)
   ↓ (check_alignment.py -> synchronized)
```

### Paso a paso:

1. **Promoción de una nueva versión**:
   ```bash
   python model_provider/scripts/promote_model.py --model salary-predictor --version <NUEVA_VERSION> --alias champion
   ```
2. **Detección de Drift operacional**:
   ```bash
   python model_provider/scripts/check_alignment.py
   ```
   Retornará código de salida `2` y estado `REDEPLOY REQUIRED`. La inferencia continúa sirviendo la versión previa sin interrupción.
3. **Redeploy explícito**:
   ```bash
   docker compose restart inference
   ```
   > **Nota Operacional**: El reinicio produce una pequeña ventana de indisponibilidad temporal (~5 a 15 segundos) mientras el proceso uvicorn/LightGBM se descarga y recarga en memoria. Mecanismos como blue-green o zero-downtime quedan fuera de alcance para esta fase.
4. **Verificación de sincronización**:
   ```bash
   python model_provider/scripts/check_alignment.py
   ```
   Retornará código `0` y estado `SYNCHRONIZED`.
5. **Procedimiento de Rollback**:
   Para regresar a la versión previa:
   ```bash
   python model_provider/scripts/promote_model.py --model salary-predictor --version <VERSION_ANTERIOR> --alias champion
   python model_provider/scripts/check_alignment.py  # Reporta redeploy_required
   docker compose restart inference                 # Recarga versión anterior
   python model_provider/scripts/check_alignment.py  # Reporta synchronized
   ```

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
