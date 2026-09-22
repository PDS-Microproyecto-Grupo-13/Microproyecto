# Home Analytics Implementation Plan

Fecha de diseño: 2026-09-22  
Alcance: especificación contractual; no implementa código, endpoints, DVC, Docker ni frontend.

Este documento parte del checkout actual y de `HOME_ANALYTICS_SOURCE_AUDIT.md`. Los nombres y cifras de ejemplo se contrastaron con `ml/dvc.yaml`, `ml/params.yaml`, la CLI y el código de `ml_pipeline`, los artifacts presentes en `ml/artifacts/reports/`, las particiones Parquet actuales, FastAPI, React/Vite y `docker-compose.yml`.

## 1. Decisiones arquitectónicas

### 1.1 Flujo aceptado

```text
dataset versionado
    -> pipeline DVC en ml/
    -> etapa determinista analytics
    -> ml/artifacts/reports/dashboard_summary.json
    -> comando explícito publish-analytics (fuera de DVC)
    -> POST HTTP autenticado
    -> validación + persistencia propia del backend
    -> GET HTTP público/read-only
    -> frontend/Home
```

Las dos operaciones siguientes son deliberadamente independientes:

```bash
cd ml
dvc repro analytics
python -m ml_pipeline publish-analytics
```

- `dvc repro analytics` solo lee artifacts/datos locales y produce un JSON reproducible. No realiza red ni otro efecto externo.
- `publish-analytics` solo publica un artifact ya existente. No ejecuta DVC, no recalcula estadísticas y no consulta MLflow.
- El backend no conoce rutas de `ml/` y conserva su propia copia.
- El frontend solo conoce el GET del backend.

### 1.2 Poblaciones elegidas

Se distinguen tres conteos; no son intercambiables:

1. **Registros brutos leídos**: suma de `rows_raw` de los snapshots. Puede incluir una misma vacante en varios cortes y, por ello, no debe rotularse como “vacantes únicas”.
2. **Vacantes validadas**: filas después de deduplicación, higiene de targets y filtro de extremos, de cualquier `target_source`.
3. **Vacantes modelables**: subconjunto validado cuyo `target_source` coincide con `params.yaml:data.target_scope`, actualmente `reportado`; es la unión exacta de train, validation y test.

Salarios, histogramas, seniority, work mode y tecnologías usarán siempre la población **modelable**. Es una decisión importante: `validation.json.target_statistics` usa todas las filas validadas, incluidas las de salario estimado e híbrido, y no es la fuente correcta para esas estadísticas del dashboard. Los valores actuales de salario modelable se calculan de las tres particiones procesadas.

### 1.3 Decisiones de contrato

- Nombre del artifact: `ml/artifacts/reports/dashboard_summary.json`.
- Versión inicial: `schema_version: "1.0"`.
- El artifact no contiene `generated_at`: una hora de ejecución basada en reloj haría que dos ejecuciones con iguales entradas generen bytes distintos. Se expone `data_range` para vigencia de datos y linaje reproducible para trazabilidad. El backend sí registra `stored_at` en su respuesta administrativa, pero ese dato no forma parte del artifact.
- `git_commit` puede ser `null`, como permite el recolector de linaje existente. Si existe, representa el commit capturado por `experiment_manifest.json`, no una consulta nueva no declarada durante `analytics`.
- No se incluye un hash autorreferencial dentro del JSON. El backend calcula `artifact_hash` como SHA-256 de la representación JSON canónica ya validada; ese hash sirve para idempotencia y ETag.
- No se incluyen filas individuales de vacantes. El dashboard es agregado; publicar títulos, empresas o filas arbitrarias ampliaría el contrato y podría exponer datos sin necesidad.
- No se usa `feature_importance.json` como frecuencia. Importancia predictiva y frecuencia descriptiva son conceptos distintos.
- MLflow no participa en la publicación. Las métricas oficiales del modelo se leen del artifact reproducible `metrics.json`.

## 2. Responsabilidades por módulo

### `ml/`

Es dueño de la definición de poblaciones, fórmulas, cálculo estadístico, bins, extracción de skills, linaje del dataset y generación reproducible. También ofrece el cliente administrativo que valida y publica el JSON por HTTP. Solo conoce URL, token y contrato HTTP; no conoce almacenamiento del backend ni UI.

### `backend/`

Es dueño de los schemas HTTP, autenticación del POST, validación estructural y cruzada, idempotencia, escritura atómica, copia persistente y GET público. No lee CSV/Parquet de ML, no importa Pandas, no ejecuta DVC, no consulta MLflow y no monta `ml/artifacts`.

### `frontend/`

Es dueño del consumo y presentación. Solo invoca `GET /api/v1/analytics/summary`; formatea moneda/porcentajes, asigna etiquetas visuales y representa loading, error y estado no publicado. No conoce artifacts, DVC, MLflow ni publicación.

## 3. Schema `dashboard_summary.json`

### 3.1 Forma normativa

Todos los objetos son cerrados: campos extra deben rechazarse en ML y backend. Todos los números deben ser JSON finitos; no se permiten `NaN`, `Infinity` ni strings numéricos.

```text
AnalyticsSummary
  schema_version: literal "1.0"
  dataset: DatasetAnalytics
  model: ModelMetrics
  metadata: AnalyticsMetadata

DatasetAnalytics
  source: non-empty string
  target_scope: non-empty string
  counts:
    raw_snapshot_rows: integer >= 0
    validated_rows: integer >= 0
    modelable_rows: integer > 0
  data_range:
    published_min: UTC RFC 3339 datetime string
    published_max: UTC RFC 3339 datetime string
  salary_midpoint:
    currency: literal "USD"
    period: literal "annual"
    mean_usd: number >= 0
    median_usd: number >= 0
    minimum_usd: number >= 0
    maximum_usd: number >= 0
  salary_midpoint_distribution: array[SalaryBin], length >= 1
  seniority_distribution: array[CategoryCount], length >= 1
  work_mode_distribution: array[CategoryCount], length >= 1
  top_technologies: array[TechnologyCount]

SalaryBin
  lower_bound_usd: integer >= 0, inclusive
  upper_bound_usd: integer > lower_bound_usd or null, exclusive
  count: integer >= 0
  proportion: number in [0, 1]

CategoryCount
  category: non-empty string
  count: integer >= 0
  proportion: number in [0, 1]

TechnologyCount
  technology: one of the canonical SKILLS keys
  count: integer >= 0
  proportion: number in [0, 1]

ModelMetrics
  algorithm: non-empty string
  evaluation_rows: integer > 0
  mae_average_usd: number >= 0
  r2_salary_min: finite number
  r2_salary_max: finite number
  predicted_range_coverage: number in [0, 1]
  uncertainty_margin_usd: number >= 0
  uncertainty_nominal_coverage: number in [0, 1]
  uncertainty_test_coverage: number in [0, 1]

AnalyticsMetadata
  dataset_fingerprint: 64-character lowercase SHA-256 hex string
  snapshot_count: integer > 0
  git_commit: 40-character lowercase Git SHA hex string or null
  dvc_revision: 64-character lowercase SHA-256 hex string or null
  params_hash: 64-character lowercase SHA-256 hex string or null
```

Invariantes cruzados:

- `published_min <= published_max`.
- `minimum_usd <= median_usd <= maximum_usd` y `minimum_usd <= mean_usd <= maximum_usd`.
- La suma de `salary_midpoint_distribution[].count` es `modelable_rows`.
- Los bins están ordenados, son contiguos, no se solapan, comienzan en 0 y solo el último tiene `upper_bound_usd: null`.
- Cada distribución de seniority y work mode suma `modelable_rows`.
- En cada distribución, `proportion = count / modelable_rows`, redondeado a 6 decimales, con tolerancia de validación de `1e-6`.
- Una vacante puede contener varias tecnologías; por eso la suma de conteos/proporciones de tecnologías puede superar la población/1.0.
- `top_technologies` está ordenado por `count` descendente y luego por `technology` ascendente para desempate; no hay claves duplicadas.
- `evaluation_rows` coincide con `preprocess.json.split.test.rows`.

### 3.2 Ejemplo JSON completo

El siguiente ejemplo usa el checkout actual y redondeo contractual. Es ilustrativo del resultado esperado, no es un artifact generado en esta fase.

```json
{
  "dataset": {
    "counts": {
      "modelable_rows": 54363,
      "raw_snapshot_rows": 333773,
      "validated_rows": 260158
    },
    "data_range": {
      "published_max": "2026-09-01T05:37:04Z",
      "published_min": "2025-01-01T00:13:20Z"
    },
    "salary_midpoint": {
      "currency": "USD",
      "maximum_usd": 2190158.0,
      "mean_usd": 167538.99,
      "median_usd": 163000.0,
      "minimum_usd": 412.0,
      "period": "annual"
    },
    "salary_midpoint_distribution": [
      {"count": 994, "lower_bound_usd": 0, "proportion": 0.018284, "upper_bound_usd": 50000},
      {"count": 6125, "lower_bound_usd": 50000, "proportion": 0.112669, "upper_bound_usd": 100000},
      {"count": 14686, "lower_bound_usd": 100000, "proportion": 0.270147, "upper_bound_usd": 150000},
      {"count": 17903, "lower_bound_usd": 150000, "proportion": 0.329323, "upper_bound_usd": 200000},
      {"count": 9652, "lower_bound_usd": 200000, "proportion": 0.177547, "upper_bound_usd": 250000},
      {"count": 5003, "lower_bound_usd": 250000, "proportion": 0.09203, "upper_bound_usd": null}
    ],
    "seniority_distribution": [
      {"category": "SE", "count": 34214, "proportion": 0.629362},
      {"category": "MI", "count": 15792, "proportion": 0.290492},
      {"category": "EN", "count": 2968, "proportion": 0.054596},
      {"category": "EX", "count": 1342, "proportion": 0.024686},
      {"category": "desconocido", "count": 47, "proportion": 0.000865}
    ],
    "source": "foorilla",
    "target_scope": "reportado",
    "top_technologies": [
      {"count": 35936, "proportion": 0.661038, "technology": "python"},
      {"count": 23991, "proportion": 0.441311, "technology": "sql"},
      {"count": 20737, "proportion": 0.381454, "technology": "machine_learning"},
      {"count": 16706, "proportion": 0.307305, "technology": "aws"},
      {"count": 11876, "proportion": 0.218457, "technology": "spark"},
      {"count": 11601, "proportion": 0.213399, "technology": "azure"},
      {"count": 8818, "proportion": 0.162206, "technology": "kubernetes"},
      {"count": 7763, "proportion": 0.142799, "technology": "pytorch"}
    ],
    "work_mode_distribution": [
      {"category": "presencial", "count": 41934, "proportion": 0.77137},
      {"category": "remoto", "count": 10702, "proportion": 0.196862},
      {"category": "híbrido", "count": 902, "proportion": 0.016592},
      {"category": "remoto_sin_detalle", "count": 784, "proportion": 0.014422},
      {"category": "remoto_global", "count": 41, "proportion": 0.000754}
    ]
  },
  "metadata": {
    "dataset_fingerprint": "541f47bf1bb49a6a0e5947d49dade61c444e74cd5350cfbb68a185506cd957ed",
    "dvc_revision": "642fe55786ad977f3bf0a1462384c90c5269b3f472adccbae36901d60c5d461b",
    "git_commit": "b322e736b6ed6d82fc28b1200b48ed2d2acb783e",
    "params_hash": "4fcad423dba9e77cbd69319a1eb461d74aead31f1198647e972665a863a52892",
    "snapshot_count": 5
  },
  "model": {
    "algorithm": "lightgbm",
    "evaluation_rows": 8155,
    "mae_average_usd": 27081.22,
    "predicted_range_coverage": 0.113918,
    "r2_salary_max": 0.628212,
    "r2_salary_min": 0.572696,
    "uncertainty_margin_usd": 50927.29,
    "uncertainty_nominal_coverage": 0.8,
    "uncertainty_test_coverage": 0.789332
  },
  "schema_version": "1.0"
}
```

## 4. Diccionario semántico de métricas

Las cantidades monetarias se expresan en USD anuales. El generador redondea importes de resumen a 2 decimales y proporciones/métricas adimensionales a 6; los conteos permanecen enteros.

| Campo | Tipo / unidad | Significado, población y fórmula |
|---|---|---|
| `schema_version` | string | Versión mayor/menor del contrato, exactamente `1.0`. No es versión de modelo. |
| `dataset.source` | string | Identificador de la fuente de `data_manifest.json.source`; actualmente `foorilla`. |
| `dataset.target_scope` | string | Valor de `params.yaml:data.target_scope` aplicado por preprocess; actualmente `reportado`. |
| `raw_snapshot_rows` | int / filas leídas | `sum(data_manifest.snapshots[*].rows_raw)`. Cuenta registros de cada corte antes de deduplicar; una vacante repetida entre snapshots puede contarse varias veces. |
| `validated_rows` | int / vacantes | `validation.json.rows`, luego de deduplicación, higiene de targets y retiro IQR de extremos en collect, y validación exitosa. Incluye todos los `target_source`. |
| `modelable_rows` | int / vacantes | `preprocess.json.rows_modeling`, igual a `len(train)+len(validation)+len(test)`. Población con `target_source == target_scope`. |
| `published_min`, `published_max` | RFC 3339 UTC | Mínimo/máximo de `published` en la unión modelable. No son hora de generación ni de publicación HTTP. |
| `salary_midpoint.*_usd` | float / USD/año | Sobre cada fila modelable, `midpoint_i=(y_min_usd_i+y_max_usd_i)/2`; luego media, mediana, mínimo o máximo de esos midpoints. Se excluyen todas las filas no modelables; no se usan predicciones. |
| `salary_midpoint_distribution[].count` | int / vacantes | Número de midpoints en el intervalo `[lower_bound_usd, upper_bound_usd)`; el último es `[lower_bound_usd, +inf)`. |
| `salary_midpoint_distribution[].proportion` | float / fracción | `count/modelable_rows`. No es porcentaje 0–100; UI multiplica por 100 cuando corresponda. |
| `seniority_distribution` | categorías, conteos, fracciones | `value_counts(dropna=False)` de `experience_level` ya preparado, sobre toda la población modelable. `desconocido` es categoría explícita. Orden: conteo descendente, categoría ascendente en empate. |
| `work_mode_distribution` | categorías, conteos, fracciones | `value_counts` de la feature derivada por `prepare_features`: `presencial`, `híbrido`, `remoto`, `remoto_global` o `remoto_sin_detalle`; población modelable. |
| `top_technologies[].count` | int / vacantes | Suma de la columna binaria canónica `skill_<technology>` sobre la población modelable. Cada vacante aporta como máximo 1 a una tecnología y puede aportar a varias. |
| `top_technologies[].proportion` | float / fracción | `count/modelable_rows`. Es prevalencia de mención según el extractor canónico, no importancia del modelo. |
| `model.algorithm` | string | Algoritmo efectivo de `experiment_manifest.json.algorithm`; actualmente `lightgbm`. |
| `evaluation_rows` | int / vacantes | Filas del test ciego, desde `preprocess.json.split.test.rows` y contrastadas con `candidate.json.test_evaluation.test_rows` si se decide incluir candidate como dependencia. |
| `mae_average_usd` | float / USD/año | `metrics.json.mae_promedio = (MAE(y_min)+MAE(y_max))/2` sobre test ciego, después de postproceso. |
| `r2_salary_min` | float / adimensional | `metrics.json.r2_min`, coeficiente R² entre `y_min_usd` observado y mínimo predicho sobre test. Puede ser negativo; no se valida como porcentaje. |
| `r2_salary_max` | float / adimensional | `metrics.json.r2_max`, equivalente para `y_max_usd`. |
| `predicted_range_coverage` | float / fracción | `metrics.json.cobertura_intervalo = mean((y_min_observado >= pred_min) AND (y_max_observado <= pred_max))` sobre test. No confundir con cobertura de incertidumbre. |
| `uncertainty_margin_usd` | float / USD/año | Margen calibrado en validation: cuantil configurado del máximo error absoluto entre ambos endpoints; copiado de `metrics.json.uncertainty_margin`. |
| `uncertainty_nominal_coverage` | float / fracción | Cuantil objetivo de calibración, `metrics.json.uncertainty_nominal_coverage`. |
| `uncertainty_test_coverage` | float / fracción | `mean(max(abs(y_true - y_pred), axis=1) <= uncertainty_margin)` sobre test ciego. Esta es la “coverage” principal del intervalo de incertidumbre. |
| `dataset_fingerprint` | SHA-256 hex | Hash determinista que `collect` calcula sobre la secuencia ordenada `nombre_de_snapshot:fingerprint_del_archivo`. Identifica entradas, no el modelo. |
| `snapshot_count` | int / snapshots | Número de entradas de `data_manifest.json.snapshots`. |
| `git_commit` | SHA/null | Commit capturado durante evaluate en `experiment_manifest.json.lineage.git_commit`. `null` si Git no estaba disponible. |
| `dvc_revision` | SHA-256/null | Hash de bytes de `dvc.yaml` capturado por el linaje existente; no es el hash de `dvc.lock`. |
| `params_hash` | SHA-256/null | Hash de bytes de `params.yaml` capturado por el linaje existente. |

## 5. Fuentes reales de cada campo

| Grupo/campo | Artifact o dato fuente | Transformación de analytics |
|---|---|---|
| source, snapshot_count, raw_snapshot_rows, fingerprint | `artifacts/reports/data_manifest.json` | Lectura, suma y validación de consistencia. |
| validated_rows | `artifacts/reports/validation.json` | Lectura; exigir `valid == true`. |
| target_scope, modelable_rows, test rows, rango temporal auxiliar | `artifacts/reports/preprocess.json` | Lectura y controles cruzados. |
| fechas, midpoint, histogramas, seniority, work mode, skills | `data/processed/train.parquet`, `validation.parquet`, `test.parquet` | Concatenación estable; agregaciones únicamente. |
| claves y reglas de skills | `src/ml_pipeline/features.py:SKILLS` y columnas `skill_*` procesadas | Suma de flags existentes; no volver a interpretar tags en analytics. |
| métricas del modelo | `artifacts/reports/metrics.json` | Renombrado semántico y redondeo. |
| algorithm y linaje | `artifacts/reports/experiment_manifest.json` | Lectura de `algorithm` y `lineage`. |
| bins/top N | `params.yaml:analytics` | Parámetros explícitos versionados. |

### Tecnologías / skills

Las particiones procesadas ya contienen 13 columnas binarias generadas por `prepare_features`: Python, SQL, AWS, Azure, GCP, Spark, Docker, Kubernetes, Machine Learning, PyTorch, TensorFlow, Tableau y Power BI. Por tanto, los datos actuales sí permiten conteos reproducibles sin volver al CSV.

La etapa `analytics` debe sumar cada `skill_*` en train + validation + test y ordenar el resultado. Debe usar las columnas binarias, no `feature_importance.json`. El cálculo pertenece a la etapa analytics, después de preprocess, porque ahí existe la población modelable y el contrato de features ya fue aplicado.

Limitación conocida: el extractor actual usa búsqueda literal de substring sobre `tags` en minúsculas. Los tests confirman que `pythonista` cuenta como Python, mientras `machine-learning` no cuenta como `machine learning` y `powerbi` no cuenta como `power bi`. Home debe describir el dato como **“menciones detectadas por el extractor canónico”**, no como una ontología exhaustiva. Corregir tokenización es una mejora de features y cambiaría resultados upstream; no pertenece a esta iniciativa.

React y TypeScript no son skills canónicas actuales, así que no pueden conservarse en la lista real salvo una ampliación futura, versionada y recalculada, de `SKILLS`.

## 6. Diseño etapa DVC `analytics`

### 6.1 Ubicación en el DAG

```text
collect -> validate -> preprocess -> qualify -> train -> evaluate
                    \                         /
                     processed partitions   /
                                              -> analytics -> dashboard_summary.json
```

Debe ser una séptima etapa posterior a `evaluate`. Aunque las analíticas descriptivas nacen de preprocess, las métricas oficiales solo existen después de evaluate.

Comando propuesto, coherente con la CLI actual:

```bash
python -m ml_pipeline analytics
```

Definición conceptual para `ml/dvc.yaml`:

```yaml
analytics:
  cmd: python -m ml_pipeline analytics
  deps:
    - data/processed/train.parquet
    - data/processed/validation.parquet
    - data/processed/test.parquet
    - artifacts/reports/data_manifest.json
    - artifacts/reports/validation.json
    - artifacts/reports/preprocess.json
    - artifacts/reports/metrics.json
    - artifacts/reports/experiment_manifest.json
    - src/ml_pipeline/analytics.py
    - src/ml_pipeline/features.py
  params:
    - analytics
  outs:
    - artifacts/reports/dashboard_summary.json:
        cache: false
```

Parámetros propuestos:

```yaml
analytics:
  salary_bins_usd: [0, 50000, 100000, 150000, 200000, 250000]
  top_technologies_limit: 8
```

No se agrega `candidate.json` si `evaluation_rows` se contrasta directamente con la longitud de `test.parquet` y preprocess; esto evita una dependencia redundante. Si Fase 2 decide validar además candidate, debe declararlo explícitamente como `dep`.

### 6.2 Comportamiento

- Cargar y validar todos los inputs antes de escribir.
- Comprobar que los fingerprints presentes coinciden y que conteos de particiones coinciden con preprocess.
- Generar el objeto en memoria, aplicar invariantes y serializar con `sort_keys=True`, indentación estable UTF-8 y newline final, siguiendo `common/io.py`.
- Escribir a temporal en el mismo directorio y reemplazar el output solo tras éxito. Aunque DVC controla ejecución, esto evita dejar un artifact truncado si falla la etapa.
- Si el output existe y algún upstream cambió, DVC ejecuta la etapa y el archivo se reemplaza.
- Si inputs, código y parámetros no cambiaron y el output existe, DVC hace skip; el artifact permanece.
- Si el output falta, `dvc repro analytics` detecta el output ausente y reconstruye la etapa, aun con upstream sin cambios. `cache: false` es coherente con los demás reportes JSON actuales y obliga a reconstrucción local si se borra.
- Si falta un input o no concuerda el fingerprint/conteo, salir distinto de cero sin tocar un output previo válido.
- Prohibido usar reloj, aleatoriedad, HTTP, MLflow API o filesystem del backend.

## 7. Diseño `publish-analytics`

### 7.1 Comando y configuración

```bash
cd ml
python -m ml_pipeline publish-analytics
```

Se agrega a `COMMANDS` junto a `track` y `register-candidate`, pero nunca a `dvc.yaml`.

Variables de entorno:

```text
ANALYTICS_PUBLISH_URL=http://localhost:8000/api/v1/analytics/snapshots
ANALYTICS_PUBLISH_TOKEN=<secret>
ANALYTICS_PUBLISH_TIMEOUT_SECONDS=10
```

`URL` y token son obligatorios para publicar; timeout admite default 10. El token nunca se imprime, persiste ni commitea. `ml/.env.example` solo contiene placeholder vacío/documentación. Se recomienda declarar `httpx` como dependencia directa del módulo ML, aunque hoy llegue transitivamente en otros paquetes; no se debe depender de una dependencia transitiva.

### 7.2 Algoritmo

1. Resolver `artifacts/reports/dashboard_summary.json` con `Settings.path`.
2. Si no existe, fallar indicando `dvc repro analytics`; no ejecutar esa orden automáticamente.
3. Parsear JSON estricto y validarlo con el mismo modelo contractual de Fase 2 (campos, versión e invariantes).
4. Construir `POST` con `Content-Type: application/json`, `Accept: application/json` y `Authorization: Bearer <token>`.
5. Enviar el summary como body, sin wrapper y sin modificarlo.
6. Interpretar `201 created`, `200 replaced/unchanged` como éxito. Validar también el response schema.
7. Mostrar una línea segura con `status`, `artifact_hash` y `stored_at`; nunca token ni payload completo.
8. Para 4xx mostrar código/mensaje del `ErrorResponse`; para 5xx, red o timeout mostrar causa acotada y accionable.

Exit codes propuestos:

| Código | Significado |
|---:|---|
| 0 | Publicación creada, reemplazada o idempotentemente sin cambios. |
| 2 | Artifact ausente, JSON inválido o schema local incompatible. |
| 3 | Configuración local ausente/inválida (URL, token, timeout). |
| 4 | Rechazo HTTP 4xx (auth, schema o conflicto). |
| 5 | Error HTTP 5xx del backend. |
| 6 | Error de transporte: DNS, conexión, TLS o timeout. |

No hacer retries automáticos en v1. El operador puede repetir con seguridad gracias a la idempotencia; esto evita ocultar fallos y tormentas de reintentos.

## 8. Contrato HTTP Backend

Las rutas propuestas respetan `API_PREFIX=/api/v1` y el patrón actual de routers:

```http
POST /api/v1/analytics/snapshots
GET  /api/v1/analytics/summary
```

### 8.1 POST administrativo

Request:

```http
POST /api/v1/analytics/snapshots
Authorization: Bearer <ANALYTICS_PUBLISH_TOKEN>
Content-Type: application/json

<AnalyticsSummary>
```

El body es exactamente el schema de la sección 3. Pydantic debe configurar modelos cerrados (`extra="forbid"`), tipos estrictos, límites y validadores cruzados. Se acepta solo `schema_version == "1.0"`.

Respuesta:

```json
{
  "artifact_hash": "64-char-sha256",
  "schema_version": "1.0",
  "status": "created",
  "stored_at": "2026-09-22T18:30:00Z"
}
```

`status` es `created`, `replaced` o `unchanged`.

| Caso | HTTP | Respuesta |
|---|---:|---|
| Primer snapshot válido | 201 | `created`. |
| Snapshot distinto reemplaza al actual | 200 | `replaced`. |
| Mismo hash canónico | 200 | `unchanged`; no reescribe ni cambia `stored_at`. |
| Bearer ausente/incorrecto | 401 | `ErrorResponse`, `error=analytics_unauthorized`, más `WWW-Authenticate: Bearer`. |
| Token del servidor no configurado | 503 | `analytics_publish_unavailable`; nunca aceptar abierto. |
| JSON/schema/campos extra inválidos | 422 | convención existente `validation_error`. |
| `schema_version` desconocido | 422 | `unsupported_analytics_schema`, detalle de versiones soportadas. |
| Persistencia no escribible/falla I/O | 503 | `analytics_storage_unavailable`; conservar copia previa. |

No se crea historial: existe un único snapshot actual. “Latest” significa el último payload distinto aceptado, no el mayor timestamp. Esto evita reglas engañosas cuando se reevalúa un modelo con el mismo dataset.

### 8.2 GET público/read-only

```http
GET /api/v1/analytics/summary
Accept: application/json
```

- `200`: body exacto `AnalyticsSummary` persistido.
- `404`: si nunca se publicó, `ErrorResponse` con `error=analytics_not_published`; no devolver ceros ni mocks con 200.
- `503`: si existe archivo pero no puede leerse/validarse, `analytics_storage_unavailable`; no devolver datos parciales.
- Header `ETag: "<artifact_hash>"`.
- Header `Cache-Control: public, max-age=60`. Es suficientemente corto para Home y reduce lecturas; no requiere Redis.
- Soportar `If-None-Match`/304 es recomendado pero puede posponerse dentro de Fase 3 sin cambiar body ni tipos.

Frontend usa únicamente GET. POST no debe exponerse desde el servicio TypeScript.

### 8.3 Errores

Se reutiliza `app.schemas.common.ErrorResponse`:

```json
{
  "error": "analytics_not_published",
  "message": "No analytics snapshot has been published",
  "request_id": "...",
  "details": null
}
```

Mensajes internos de filesystem, paths y tokens no se filtran al cliente; van a logs estructurados con request ID.

## 9. Persistencia e idempotencia

### 9.1 Opción elegida

Archivo JSON propiedad del backend:

```text
/app/data/analytics_summary.json
```

El formato **interno** del archivo es un envelope de almacenamiento, no otro contrato HTTP:

```json
{
  "artifact_hash": "64-char-sha256",
  "storage_version": 1,
  "stored_at": "2026-09-22T18:30:00Z",
  "summary": {"schema_version": "1.0"}
}
```

`summary` contiene el objeto completo de la sección 3. El envelope permite conservar `stored_at` y el hash entre restarts; GET extrae únicamente `summary`. `storage_version` es privado del backend y no sustituye `schema_version`.

Configuración:

```text
ANALYTICS_STORAGE_PATH=/app/data/analytics_summary.json
```

En desarrollo fuera de contenedor puede tener un default relativo resuelto, por ejemplo `backend/data/analytics_summary.json`, pero tests siempre inyectan un `tmp_path`. El servicio no crea un snapshot por defecto.

Docker Compose debe añadir un named volume exclusivo:

```yaml
backend:
  volumes:
    - backend-analytics-data:/app/data

volumes:
  backend-analytics-data:
```

Esto sobrevive restart y recreación normal del contenedor. No sobrevive `docker compose down -v`, lo cual debe documentarse. El directorio debe crearse en la imagen y ser escribible por el usuario efectivo del backend; el Dockerfile actual ejecuta como root, pero Fase 3 debería preferir un usuario no root y hacer `chown` durante build.

### 9.2 Escritura atómica

1. Validar completamente el body.
2. Canonicalizar con claves ordenadas, separadores estables, UTF-8 y newline.
3. Calcular SHA-256 canónico.
4. Bajo un lock de proceso, comparar con el snapshot actual válido.
5. Escribir un temporal único en `/app/data` (mismo filesystem).
6. `flush` + `os.fsync(file_descriptor)` + cerrar.
7. `os.replace(temp, analytics_summary.json)`; POSIX garantiza reemplazo atómico en el mismo filesystem.
8. Abrir/fsync del directorio cuando esté disponible para durabilidad frente a crash.
9. Limpiar temporal en `finally` si falló antes del replace.

GET abre el path definitivo; observará la versión anterior o la nueva, nunca bytes parciales. El runtime Docker actual ejecuta un solo worker Uvicorn, por lo que un `threading.Lock`/lock de servicio basta para serializar POST dentro del proceso. Si en el futuro se usan varios workers/replicas sobre el mismo volumen, se requerirá file lock o almacenamiento transaccional; no se introduce ahora.

### 9.3 Idempotencia

`artifact_hash = SHA256(canonical_validated_summary_bytes)`. Incluye dataset, métricas, configuración de bins y linaje efectivo, a diferencia de `dataset_fingerprint`, que solo identifica snapshots.

- Mismo hash: `unchanged`, sin escritura ni duplicado.
- Hash diferente: replace atómico, `replaced`.
- POST simultáneos: lock + comparación dentro de la sección crítica; last accepted write wins y el archivo nunca queda corrupto.
- No hay tabla de versiones, IDs históricos ni cleanup.

Comparativa:

| Alternativa | Evaluación v1 |
|---|---|
| JSON + named volume | Elegida: un objeto, lectura simple, backup/copias fáciles, sin infraestructura. |
| SQLite | Transacciones útiles, pero innecesarias sin historial ni consultas; añade esquema/migración. |
| PostgreSQL | Ya no existe como dependencia del backend actual y sería sobredimensionado para un snapshot. |
| Redis | Persistencia adicional y semántica de cache innecesarias; explícitamente fuera de alcance. |

## 10. Seguridad mínima

- POST protegido por bearer token compartido en `Authorization`; comparación constante con `secrets.compare_digest`.
- Secret backend: `ANALYTICS_PUBLISH_TOKEN`. Secret del publisher: variable del mismo nombre o `ANALYTICS_PUBLISH_TOKEN`; nunca Git, JSON, logs ni argumentos CLI.
- GET sin autenticación, como lectura pública de agregados.
- El token es obligatorio para habilitar POST. Ausencia en backend produce 503, no modo abierto.
- Mantener el backend en red controlada y usar HTTPS fuera de localhost. La red interna por sí sola no basta porque Compose publica `8000:8000`.
- No implementar OAuth/JWT, usuarios, refresh tokens ni scopes en v1.
- Rotación: actualizar secret en ambos entornos y reiniciar backend; no afecta snapshot persistido.

## 11. Contrato Frontend

### 11.1 Servicio

En `frontend/src/services/salaryApi.ts`, manteniendo el patrón actual de URL relativa y manejo de `body?.message`:

```ts
export async function getAnalyticsSummary(): Promise<AnalyticsSummary>
// GET /api/v1/analytics/summary
```

No necesita variable de entorno: Vite y Nginx ya proxifican `/api` al backend.

Para distinguir 404 no publicado de otros fallos, el helper debe lanzar un error HTTP tipado que preserve al menos `status`, `error` y `message` del `ErrorResponse`; comparar texto libre no es aceptable. Puede generalizarse el manejo de errores de `salaryApi.ts`, sin cambiar el contrato de predicción.

Tipos propuestos:

```ts
export type AnalyticsSchemaVersion = "1.0";

export interface DistributionItem {
  category: string;
  count: number;
  proportion: number;
}

export interface SalaryDistributionBin {
  lower_bound_usd: number;
  upper_bound_usd: number | null;
  count: number;
  proportion: number;
}

export interface TechnologyFrequency {
  technology: string;
  count: number;
  proportion: number;
}

export interface AnalyticsSummary {
  schema_version: AnalyticsSchemaVersion;
  dataset: {
    source: string;
    target_scope: string;
    counts: {
      raw_snapshot_rows: number;
      validated_rows: number;
      modelable_rows: number;
    };
    data_range: { published_min: string; published_max: string };
    salary_midpoint: {
      currency: "USD";
      period: "annual";
      mean_usd: number;
      median_usd: number;
      minimum_usd: number;
      maximum_usd: number;
    };
    salary_midpoint_distribution: SalaryDistributionBin[];
    seniority_distribution: DistributionItem[];
    work_mode_distribution: DistributionItem[];
    top_technologies: TechnologyFrequency[];
  };
  model: {
    algorithm: string;
    evaluation_rows: number;
    mae_average_usd: number;
    r2_salary_min: number;
    r2_salary_max: number;
    predicted_range_coverage: number;
    uncertainty_margin_usd: number;
    uncertainty_nominal_coverage: number;
    uncertainty_test_coverage: number;
  };
  metadata: {
    dataset_fingerprint: string;
    snapshot_count: number;
    git_commit: string | null;
    dvc_revision: string | null;
    params_hash: string | null;
  };
}
```

### 11.2 Estados de Home

Home mantiene estado discriminado o tres estados coordinados:

- `loading`: skeleton/texto “Cargando analítica”; no mostrar números anteriores.
- `success`: render de `AnalyticsSummary` con `Intl.NumberFormat` para enteros, USD y porcentajes.
- `unpublished`: GET 404 con `error=analytics_not_published`; panel vacío explícito “La analítica aún no fue publicada”, sin tratarlo como cero.
- `error`: red, 5xx, JSON inválido u otros 4xx; mensaje con reintento manual, manteniendo navegación disponible.
- Si el componente se desmonta, abortar fetch o ignorar resolución tardía.

Las etiquetas de categorías (`SE` -> Senior, etc.) son presentación y pertenecen al frontend. El frontend no recalcula agregados; solo puede escalar `proportion` a porcentaje y calcular altura relativa de barras para dibujar.

## 12. Mapeo Home actual → analítica real

| Elemento Home actual | Estado actual | Sustitución propuesta | Fuente real |
|---|---|---|---|
| Badge `ML Model v1.0` | Literal ambiguo | Conservar como etiqueta de producto o cambiar a `Analytics schema 1.0`; no inferir versión desplegada | `schema_version`; no Registry |
| Ofertas Ingeridas `1,450+` | Hardcoded; semántica ambigua | Cambiar rótulo a **Vacantes modelables** y valor exacto `modelable_rows`. Como detalle mostrar `validated_rows` y snapshots; no llamar “ofertas únicas” a `raw_snapshot_rows` | preprocess + validation + manifest |
| Tendencia `+12.5% este mes` | Inventada | Eliminar. No existe serie temporal comparable por mes en el contrato | Sin fuente válida |
| Salario Medio `$52,800` | Hardcoded | Mostrar **Punto medio mediano anual** (`median_usd`), más robusto; opcionalmente media en texto secundario | Parquets modelables, targets observados |
| `USD mercado tech` | Literal | Conservar como unidad contextual: `USD/año, salarios reportados` | `currency`, `period`, `target_scope` |
| Roles Clasificados `18` | Inventado; no hay taxonomía de 18 clases | Eliminar/reemplazar tarjeta por **MAE promedio test** o **Vacantes validadas**. No usar número de títulos únicos como “roles clasificados” | metrics o validation |
| R² `94.2%` | Falso | Mostrar **R² test mín.–máx.** con `r2_salary_min` y `r2_salary_max` (p. ej. 57,3%–62,8%), aclarando las dos salidas | metrics.json |
| `Gradient Boosting` | Literal genérico | Mostrar algorithm del snapshot (`lightgbm`) con etiqueta amigable | experiment_manifest.json |
| Distribución salarial: 6 barras CSS | Mock | Conservar área visual; renderizar seis bins reales con count/proportion y límites USD | `salary_midpoint_distribution` |
| Tecnologías y conteos | Mock; incluye skills no canónicas | Conservar lista, cambiar subtítulo a **Frecuencia de menciones en vacantes modelables**; renderizar top 8 reales. React/TypeScript desaparecen con contrato actual | sumas de `skill_*` |
| “con mejor remuneración” | No se calcula filtro salarial | Eliminar esa frase; la frecuencia cubre toda la población modelable | Contrato de skills |
| Tres filas de ejemplo | Mock | Eliminar. Sustituir tarjeta por distribución de seniority/work mode o metadatos de cobertura/rango temporal | distribuciones agregadas / metadata |
| Acciones rápidas | Navegación real, no dato | Conservar sin cambios | Router frontend |

Layout recomendado sin diseñar UI: cuatro tarjetas de resumen (modelables, mediana midpoint, MAE promedio, R² min–max), histograma real, acciones rápidas, top tecnologías y distribución de seniority/work mode. No mostrar filas de ejemplo porque el contrato intencionalmente no publica registros.

## 13. Flujo operacional

### Operación normal

```bash
cd ml
dvc repro analytics
ANALYTICS_PUBLISH_URL=http://localhost:8000/api/v1/analytics/snapshots \
ANALYTICS_PUBLISH_TOKEN='<secret>' \
python -m ml_pipeline publish-analytics
```

```text
snapshots + reports + processed partitions
  -> analytics
  -> dashboard_summary.json
  -> publish-analytics
  -> POST backend
  -> /app/data/analytics_summary.json (named volume)
  -> GET /api/v1/analytics/summary
  -> Home
```

### Caso A — cambia el dataset

`dvc repro` ejecuta las etapas invalidadas hasta evaluate y analytics. Luego el operador publica explícitamente. Un `dvc repro` exitoso no implica publicación.

### Caso B — nada cambia

`dvc repro` hace skip de analytics y conserva el artifact. `publish-analytics` puede ejecutarse independientemente; backend responde `unchanged` si ya tiene el mismo hash.

### Caso C — falta el artifact analytics

`dvc repro analytics` lo reconstruye desde deps existentes. `publish-analytics` solo, en cambio, falla con exit 2 y no intenta reconstruir.

### Caso D — backend recién desplegado/vacío

No es necesario recalcular ML. Si el artifact válido existe localmente, ejecutar `publish-analytics`; el backend crea su copia y Home deja el estado unpublished.

### Caso E — backend reiniciado/recreado

Restart o recreación conservan el named volume y GET sigue funcionando. Tras `down -v`, pérdida/corrupción o migración a un volumen nuevo, volver a publicar el artifact existente.

## 14. Manejo de errores

| Falla | Comportamiento ML/publisher | Backend / Frontend | Recovery |
|---|---|---|---|
| Falta input de analytics | etapa exit 1, no reemplaza output anterior | Sin cambio | `dvc pull`/`dvc repro` upstream y repetir analytics |
| Falta dashboard artifact | publish exit 2, sin HTTP | Sin cambio | `dvc repro analytics` |
| JSON malformado/no finito | publish exit 2 antes de red | POST también devolvería 422 | Regenerar; no editar artifact a mano |
| Invariantes/fingerprint no coinciden | analytics/publish falla | Backend rechaza 422 si llega | Reproducir DAG completo y revisar artifacts mezclados |
| Backend caído/DNS/conexión | publish exit 6 | Snapshot previo sigue sirviéndose si backend está activo en otra instancia | Restaurar servicio y repetir publicación |
| Timeout | publish exit 6; resultado remoto puede ser incierto | Operador puede repetir idempotentemente | Repetir y observar `unchanged`/`created` |
| Token ausente/incorrecto | publish exit 3 si local ausente; exit 4 ante 401 | 401, no escribe | Sincronizar/rotar secret; no mostrarlo en logs |
| Schema desconocido | publish local exit 2 o backend 422/exit 4 | Conserva snapshot anterior | Actualizar consumidor/productor coordinadamente |
| Storage no escribible/lleno | POST 503; temporal limpiado; copia previa intacta | GET sirve copia previa si legible | Corregir volumen/permisos/espacio y repetir |
| Snapshot nunca publicado | N/A | GET 404 `analytics_not_published`; Home muestra empty/unpublished | Publicar artifact existente |
| Snapshot persistido corrupto | Nueva publicación puede reemplazarlo; idempotencia trata actual como inválido | GET 503, nunca body parcial | Revisar volumen y republicar artifact validado |
| Error GET/red frontend | N/A | Home muestra error y opción reintentar | Reintentar sin mocks ni ceros |

La respuesta a errores usa el formato global existente. El snapshot anterior no se elimina antes de una escritura nueva válida.

### Compatibilidad de schema

- Productor ML y backend soportan inicialmente solo `1.0`.
- Backend valida versión antes de persistir. Una versión desconocida recibe 422 y no altera estado.
- GET devuelve la versión persistida, que siempre fue aceptada por esa versión del backend.
- Cambios aditivos opcionales requieren `1.1`; cambios incompatibles requieren `2.0` y despliegue coordinado backend -> ML publisher -> frontend.
- No se diseñan migraciones: para una versión nueva se republica un artifact nuevo. Durante rollout, conservar soporte explícito de la versión anterior solo si se necesita despliegue sin downtime.

## 15. Estrategia de tests

### ML

| Test | Verificación mínima |
|---|---|
| Determinismo | Dos generaciones con mismos fixtures producen bytes idénticos. |
| Schema | Campos/tipos cerrados, versión 1.0, JSON finito. |
| Conteos/poblaciones | raw, validated, modelable y suma de splits correctos; fingerprints concordantes. |
| Salarios | midpoint por fila, media/mediana/extremos y bins con límites inclusivo/exclusivo. |
| Distribuciones | Seniority/work mode suman población y proporciones correctas. |
| Skills | Suma de flags binarios, top N, empate estable; nunca usa feature importance. |
| Inputs faltantes/mezclados | Falla clara y conserva output previo. |
| Publish success | Request, headers y parseo de `created/replaced/unchanged`; exit 0. |
| Publish local invalid | Artifact ausente/JSON/schema inválido; no se invoca HTTP; exit 2. |
| Publish HTTP/transport | 401/422 -> 4, 5xx -> 5, timeout/conexión -> 6; token no aparece en logs. |

### Backend

| Test | Verificación mínima |
|---|---|
| POST válido inicial | Bearer correcto, 201, persisted JSON válido y response schema. |
| POST inválido | 422 por tipos, extras, NaN/invariantes/distribuciones incorrectas y versión desconocida. |
| Auth | Ausente/incorrecto 401; token backend ausente 503; comparación sin leak. |
| Idempotencia | Segundo POST idéntico devuelve unchanged, mismo hash/stored_at y no reescribe. |
| Replace | Payload distinto válido devuelve replaced y GET ve únicamente el nuevo. |
| Atomicidad | Simular error antes de replace: archivo anterior intacto; GET concurrente jamás parsea parcial. |
| GET vacío | 404 ErrorResponse específico. |
| GET válido | 200 exacto, ETag y Cache-Control. |
| Corrupción/I/O | 503 sanitizado, request ID, sin borrar copia. |
| Persistencia/restart | Dos instancias de app con mismo `tmp_path` leen el snapshot existente. En integración Compose, recrear contenedor sin borrar volume. |

### Frontend

El checkout no tiene framework de tests frontend configurado. Fase 4 deberá añadir la mínima infraestructura (por ejemplo Vitest + Testing Library) si se exigen tests automatizados, actualizando package/lock de forma consistente.

- loading sin valores mock;
- success y formato de conteos/USD/proporciones;
- 404 unpublished como empty state diferenciado;
- error de red/5xx y reintento;
- histograma y tecnologías renderizados desde payload;
- eliminación de filas/tendencias inventadas;
- cleanup/abort de request al desmontar.

### E2E

```text
fixture determinista -> analytics -> dashboard_summary.json
-> publisher con token -> POST -> volumen backend
-> GET igual al artifact semántico -> Home sin literales mock
```

Agregar escenarios de publicación repetida, backend vacío y recreación del backend conservando named volume. La prueba E2E no debe meter `publish-analytics` dentro de `dvc repro`.

## 16. Archivos previstos por fase

Las listas son previsiones; cada agente debe volver a inspeccionar el checkout y limitarse a su fase.

### Fase 2 — ML

Modificar probablemente:

- `ml/dvc.yaml`: etapa `analytics` únicamente.
- `ml/params.yaml`: bins y top N.
- `ml/src/ml_pipeline/cli.py`: registrar `analytics` y `publish-analytics`.
- `ml/src/ml_pipeline/settings.py`: configuración de publicación, o un settings específico para no obligar variables en etapas DVC.
- `ml/src/ml_pipeline/common/io.py`: solo si se reutiliza escritura atómica/canónica.
- `ml/src/ml_pipeline/analytics.py`: nuevo cálculo/schema local.
- `ml/src/ml_pipeline/publish_analytics.py`: nuevo cliente HTTP separado.
- `ml/tests/unit/test_analytics.py`, `test_publish_analytics.py`: nuevos.
- `ml/tests/integration/test_pipeline.py`: incluir etapa/artifact si corresponde.
- `ml/requirements.txt` y lock: declarar cliente HTTP/schema si se añade dependencia directa.
- `ml/.env.example`, `ml/README.md`: variables y operación separada.

Generará en runtime/DVC `ml/artifacts/reports/dashboard_summary.json`; no debe crearse en Fase 1. Según la política actual de reportes, revisar si `.gitignore` necesita cambio; no asumirlo sin comprobar cómo se versionan los JSON existentes.

No modificar backend, frontend, Compose raíz ni publicar durante esta fase.

### Fase 3 — Backend

Modificar probablemente:

- `backend/app/api/router.py`.
- `backend/app/api/routes/analytics.py` (nuevo).
- `backend/app/schemas/analytics.py` (nuevo).
- `backend/app/services/analytics_service.py` o repository equivalente (nuevo).
- `backend/app/core/config.py`.
- `backend/app/core/exceptions.py` solo para errores tipados específicos.
- `backend/app/api/routes/__init__.py`, `schemas/__init__.py`, `services/__init__.py` solo si el estilo de imports lo requiere.
- `backend/tests/conftest.py`, `backend/tests/test_analytics.py` (nuevo).
- `backend/.env.example`, `backend/README.md`.
- `backend/Dockerfile` para directorio/permisos y usuario no root si se adopta.
- `docker-compose.yml` para token, storage path y named volume propio.
- `.env.example` raíz para placeholders de configuración (sin secret real).

La excepción a “solo backend” es estrictamente la configuración Docker raíz necesaria para persistencia y secrets, autorizada por la definición de Fase 3. No montar `ml/`.

### Fase 4 — Frontend

Modificar probablemente:

- `frontend/src/services/salaryApi.ts` para tipos y GET, o separar `analyticsApi.ts` si se prefiere cohesión; no duplicar manejador HTTP.
- `frontend/src/pages/Home/HomePage.tsx`.
- `frontend/src/pages/Home/HomePage.module.css`.
- nuevos componentes presentacionales solo si reducen complejidad.
- `frontend/package.json` y lock correspondiente si se añade testing.
- tests nuevos de service/Home.
- `frontend/README.md`.

No modificar backend/ML. El endpoint relativo ya funciona con Vite/Nginx; no se prevé cambio de proxy.

### Validación E2E final

- Puede añadir un script/test E2E o documentación operacional en raíz.
- Puede tocar Compose solo para corregir integración descubierta, no para añadir servicios.
- Debe verificar que el único canal entre ML y backend sea HTTP POST y entre frontend/backend sea HTTP GET.

## 17. Riesgos / decisiones pendientes

| Riesgo/decisión | Resolución recomendada |
|---|---|
| Auditoría atribuyó salarios de validation a población modelable | Corregido en este contrato: salarios dashboard se recalculan en particiones modelables; no copiar `validation.target_statistics`. |
| `generated_at` rompe determinismo | Omitirlo en schema 1.0. Usar rango de datos y linaje; `stored_at` solo como metadata operacional del backend. |
| Extractor de skills por substring | Aceptar semántica actual y etiquetarla; mejora futura separada puede cambiar schema/dataset fingerprint derivado. |
| Outliers modelables hasta USD 2.190.158 | Mostrar histograma con último bin abierto y mediana; no truncar silenciosamente a `train_limits`. Los límites de modelo no son filtros descriptivos. |
| Datos Unicode en categorías (`híbrido`) | JSON UTF-8 y comparaciones exactas; frontend no debe eliminar acentos de claves. |
| Dos lockfiles frontend (`bun.lock` y `package-lock.json`) | Fase 4 debe confirmar package manager canónico antes de añadir dependencias y no actualizar ambos accidentalmente. |
| Backend actual corre como root | Para v1 funciona, pero Fase 3 debería usar usuario no root y permisos explícitos del volumen. |
| Varios workers/replicas futuros | El reemplazo es atómico, pero el lock en memoria no coordina procesos. Mantener un worker actual o introducir file lock antes de escalar; no DB ahora. |
| Cache HTTP y publicación reciente | `max-age=60` implica hasta 60 s de staleness. Adecuado para snapshots manuales; botón reintentar puede usar `cache: no-store` si UX lo necesita. |
| Compatibilidad durante despliegue | Desplegar backend que acepta schema antes del nuevo publisher; frontend después de que GET exista. |
| Secret en Compose | Usar `.env` no versionado/secret del entorno. `${...}` en Compose no debe contener default real. |
| Ejemplo contiene commit/linaje actuales y checkout está dirty | Es solo ejemplo observado. El artifact real toma el linaje reproducible de evaluate; no convierte dirty state en campo porque `git_dirty` no se publica en schema 1.0. |

No hay una decisión pendiente que bloquee las fases. La principal elección que no debe reabrirse durante implementación es la población: agregados descriptivos de salario/skills/distribuciones usan las 54.363 filas modelables con salario reportado, no las 260.158 validadas mixtas.

## 18. Plan de implementación recomendado

### Fase 2 — ML

1. Crear modelos/validadores locales del schema 1.0.
2. Implementar agregaciones puras sobre fixtures y después conectar artifacts reales.
3. Añadir parámetros analytics y etapa DVC con deps exhaustivos.
4. Garantizar serialización determinista y reemplazo seguro.
5. Implementar publisher independiente con exit codes y token seguro.
6. Añadir unit/integration tests y documentación.
7. Verificar `dvc repro analytics` dos veces (run/skip), reconstrucción tras output ausente y que ninguna prueba/etapa haga HTTP salvo tests mock del publisher.

### Fase 3 — Backend

1. Implementar schemas Pydantic cerrados y validadores cruzados.
2. Implementar servicio de snapshot con canonical hash, lock y atomic replace.
3. Implementar auth Bearer para POST y errores dentro de `ErrorResponse`.
4. Registrar router; implementar POST dinámico 201/200 y GET 200/404/503.
5. Añadir settings, examples y tests con directorios temporales.
6. Añadir named volume y permisos en Docker; confirmar persistencia tras recreación.
7. Probar publicación real desde ML solo después de que unit tests pasen.

### Fase 4 — Frontend

1. Definir tipos espejo y `getAnalyticsSummary()` usando URL relativa.
2. Modelar loading/success/unpublished/error sin valores fallback inventados.
3. Mapear tarjetas y agregados según la tabla de la sección 12.
4. Eliminar tendencia, roles clasificados, tecnologías y filas mock.
5. Renderizar histograma/distribuciones sin recalcular estadísticas.
6. Añadir tests y verificar build/lint.

### Validación E2E final

1. Partir de backend con volumen vacío y comprobar GET 404/Home empty.
2. Ejecutar `dvc repro analytics`; validar artifact contra schema.
3. Publicar con token; comprobar 201 y GET semánticamente idéntico.
4. Repetir publicación; comprobar 200 unchanged y mismo ETag.
5. Abrir Home y contrastar cada valor con el artifact.
6. Recrear backend sin borrar volume; comprobar GET/Home persistentes.
7. Cambiar fixture/dataset controlado, reproducir y publicar; comprobar replace sin lectura parcial.
8. Confirmar que DVC no realizó HTTP y que backend/frontend no accedieron a rutas de `ml/`.

HOME_ANALYTICS_IMPLEMENTATION_PLAN_STATUS=COMPLETE
