# Home Analytics — Phase 2 ML Report

Fecha: 2026-09-22

## 1. Resumen

Se implementó exclusivamente la parte ML de Home Analytics:

- contrato local estricto `AnalyticsSummary` schema `1.0`;
- comando reproducible `python -m ml_pipeline analytics`;
- etapa DVC `analytics` posterior a `evaluate`;
- artifact `ml/artifacts/reports/dashboard_summary.json`;
- escritura determinista y reemplazo atómico;
- comando administrativo separado `python -m ml_pipeline publish-analytics`;
- cliente HTTP con validación local, bearer token y exit codes contractuales;
- tests unitarios e integración sin publicaciones reales;
- documentación y configuración mínima del módulo ML.

`analytics` no realiza HTTP, no consulta MLflow y no ejecuta DVC como subprocess. `publish-analytics` no calcula métricas, no ejecuta DVC y no consulta MLflow.

## 2. Archivos modificados

Archivos existentes modificados:

- `ml/.env.example`
- `ml/README.md`
- `ml/dvc.lock`
- `ml/dvc.yaml`
- `ml/params.yaml`
- `ml/requirements.txt`
- `ml/requirements.lock.txt`
- `ml/src/ml_pipeline/cli.py`
- `ml/src/ml_pipeline/settings.py`
- `ml/tests/integration/test_pipeline.py`

Archivos nuevos:

- `ml/artifacts/reports/dashboard_summary.json`
- `ml/src/ml_pipeline/analytics.py`
- `ml/src/ml_pipeline/analytics_schema.py`
- `ml/src/ml_pipeline/publish_analytics.py`
- `ml/tests/unit/test_analytics.py`
- `ml/tests/unit/test_publish_analytics.py`
- `HOME_ANALYTICS_PHASE2_ML_REPORT.md`

No se modificaron backend, frontend, model provider, Docker Compose, documentación raíz preexistente ni el plan contractual. El cambio preexistente en `frontend/src/components/navigation/Sidebar/Sidebar.tsx` y los documentos raíz ya presentes se conservaron intactos.

## 3. Contrato implementado

El artifact implementa `schema_version: "1.0"` con objetos cerrados y tipos estrictos para:

- dataset: fuente, scope, conteos, rango temporal, salarios midpoint, histograma, seniority, work mode y tecnologías;
- model: métricas de evaluación reproducidas por el pipeline;
- metadata: fingerprint, snapshots y linaje.

Se rechazan campos extra, versiones desconocidas, strings numéricos, `NaN`, infinitos, timestamps no UTC, hashes inválidos y violaciones cruzadas. Se validan bins contiguos, sumas poblacionales, proporciones a 6 decimales, orden estable, skills canónicas, conteos de splits y fingerprints concordantes.

La metadata usa exactamente:

```text
dataset_fingerprint
snapshot_count
git_commit
dvc_yaml_hash
params_hash
```

No existe `dvc_revision` en el contrato publicado. El checkout actual guarda el SHA-256 de `dvc.yaml` bajo el nombre histórico `experiment_manifest.lineage.dvc_revision`; el generador lo mapea a `dvc_yaml_hash` sin recalcular linaje ni realizar lecturas no declaradas. También acepta la clave corregida `dvc_yaml_hash` si un manifest futuro ya la produce.

No se incluye `generated_at`. La serialización usa UTF-8, claves ordenadas, indentación estable, newline final y `allow_nan=False`.

## 4. Nueva etapa DVC

Definición efectiva:

```yaml
analytics:
  cmd: python -m ml_pipeline analytics
  deps:
    - artifacts/reports/data_manifest.json
    - artifacts/reports/experiment_manifest.json
    - artifacts/reports/metrics.json
    - artifacts/reports/preprocess.json
    - artifacts/reports/validation.json
    - data/processed/test.parquet
    - data/processed/train.parquet
    - data/processed/validation.parquet
    - src/ml_pipeline/analytics.py
    - src/ml_pipeline/analytics_schema.py
    - src/ml_pipeline/cli.py
    - src/ml_pipeline/common/io.py
    - src/ml_pipeline/common/logging.py
    - src/ml_pipeline/data/collect.py
    - src/ml_pipeline/features.py
    - src/ml_pipeline/settings.py
  params:
    - analytics
  outs:
    - artifacts/reports/dashboard_summary.json:
        cache: false
```

Parámetros:

```yaml
analytics:
  salary_bins_usd: [0, 50000, 100000, 150000, 200000, 250000]
  top_technologies_limit: 8
```

`dvc.lock` fue actualizado exclusivamente mediante `dvc repro analytics`; no se editó manualmente. La salida DOT de `dvc dag` confirmó explícitamente `evaluate -> analytics`, además de las dependencias descriptivas desde collect/validate/preprocess.

## 5. Estadísticas generadas

Artifact final: `ml/artifacts/reports/dashboard_summary.json`.

### Poblaciones y rango

| Métrica | Valor real |
|---|---:|
| Registros brutos en snapshots (`raw_snapshot_rows`) | 333.773 |
| Filas validadas/deduplicadas (`validated_rows`) | 260.158 |
| Filas modelables con scope `reportado` (`modelable_rows`) | 54.363 |
| Publicación mínima | 2025-01-01T00:13:20Z |
| Publicación máxima | 2026-09-01T05:37:04Z |
| Snapshots | 5 |

`raw_snapshot_rows` es suma de filas leídas en todos los cortes y puede contar la misma vacante en múltiples snapshots; no representa vacantes únicas.

### Salary midpoint observado

Población: unión train + validation + test, sin predicciones y sin truncar por `train_limits`.

| Métrica USD/año | Valor |
|---|---:|
| Media | 167.538,99 |
| Mediana | 163.000,00 |
| Mínimo | 412,00 |
| Máximo | 2.190.158,00 |

Histograma `[lower, upper)`:

| Rango USD | Count | Proportion |
|---|---:|---:|
| 0–49.999 | 994 | 0,018284 |
| 50.000–99.999 | 6.125 | 0,112669 |
| 100.000–149.999 | 14.686 | 0,270147 |
| 150.000–199.999 | 17.903 | 0,329323 |
| 200.000–249.999 | 9.652 | 0,177547 |
| >= 250.000 | 5.003 | 0,092030 |

### Seniority

| Categoría | Count | Proportion |
|---|---:|---:|
| SE | 34.214 | 0,629362 |
| MI | 15.792 | 0,290492 |
| EN | 2.968 | 0,054596 |
| EX | 1.342 | 0,024686 |
| desconocido | 47 | 0,000865 |

### Work mode

| Categoría | Count | Proportion |
|---|---:|---:|
| presencial | 41.934 | 0,771370 |
| remoto | 10.702 | 0,196862 |
| híbrido | 902 | 0,016592 |
| remoto_sin_detalle | 784 | 0,014422 |
| remoto_global | 41 | 0,000754 |

### Top technologies

Los valores provienen exclusivamente de sumar las columnas binarias `skill_*` de las particiones procesadas. No se leyó `feature_importance.json`.

| Tecnología canónica | Count | Proportion |
|---|---:|---:|
| python | 35.936 | 0,661038 |
| sql | 23.991 | 0,441311 |
| machine_learning | 20.737 | 0,381454 |
| aws | 16.706 | 0,307305 |
| spark | 11.876 | 0,218457 |
| azure | 11.601 | 0,213399 |
| kubernetes | 8.818 | 0,162206 |
| pytorch | 7.763 | 0,142799 |

Estas cifras significan **menciones detectadas por el extractor canónico** actual. No representan una ontología exhaustiva de tecnologías. Una vacante puede sumar en varias tecnologías, por lo que las proporciones no tienen que sumar 1.

### Model evaluation metrics

| Métrica | Valor real |
|---|---:|
| Algorithm | lightgbm |
| Evaluation rows | 8.155 |
| MAE promedio USD | 27.081,22 |
| R² salary min | 0,572696 |
| R² salary max | 0,628212 |
| Predicted range coverage | 0,113918 |
| Uncertainty margin USD | 50.927,29 |
| Nominal uncertainty coverage | 0,800000 |
| Test uncertainty coverage | 0,789332 |

Estas son métricas de evaluación asociadas al snapshot analítico y reproducidas por el pipeline local. **No prueban que el modelo evaluado sea el `champion`, una versión desplegada ni el modelo actualmente servido por inference.** No se consultó MLflow ni Model Registry.

## 6. Publisher

Comando:

```bash
python -m ml_pipeline publish-analytics
```

Variables:

```text
ANALYTICS_PUBLISH_URL
ANALYTICS_PUBLISH_TOKEN
ANALYTICS_PUBLISH_TIMEOUT_SECONDS=10
```

URL y token son obligatorios únicamente al publicar. El flujo valida primero `dashboard_summary.json` con el mismo `AnalyticsSummary` usado por el generador y luego envía el objeto sin wrapper a:

```http
POST /api/v1/analytics/snapshots
Content-Type: application/json
Accept: application/json
Authorization: Bearer <token>
```

Acepta HTTP 201/200 y valida `artifact_hash`, `schema_version`, `status` (`created`, `replaced`, `unchanged`) y `stored_at` UTC.

Exit codes:

| Exit | Caso |
|---:|---|
| 0 | created/replaced/unchanged |
| 2 | artifact ausente, JSON inválido o schema incompatible |
| 3 | configuración local ausente/inválida |
| 4 | HTTP 4xx |
| 5 | HTTP 5xx o respuesta success incompatible |
| 6 | timeout/DNS/connect/TLS/transporte |

No hay retries automáticos. No se realizó ninguna publicación HTTP real en esta fase; todos los escenarios HTTP se probaron con `httpx.MockTransport`.

## 7. Tests

Comandos y resultados relevantes:

```text
pytest -q ml/tests/unit/test_analytics.py ml/tests/unit/test_publish_analytics.py ml/tests/unit/test_settings.py
26 passed

pytest -q ml/tests/unit ml/tests/contract
108 passed, 2 skipped, 34 warnings

pytest -q ml/tests/unit/test_analytics.py ml/tests/unit/test_publish_analytics.py ml/tests/integration/test_pipeline.py
27 passed, 28 warnings

ruff check <analytics.py, analytics_schema.py, publish_analytics.py y tests nuevos>
All checks passed

git diff --check
sin errores
```

Los dos contract tests omitidos ya estaban marcados por el proyecto como pertenecientes a fases posteriores. Los warnings observados corresponden a deprecación de `TargetEncoder` y columnas sin observaciones en el fixture de integración existente; no son fallos de analytics.

Cobertura nueva incluye determinismo, conteos, splits, salary midpoint, bins, unknowns, ordering, skills múltiples/top N, schema estricto, NaN/infinito/extras/versiones, fingerprints, escritura atómica, respuestas publisher 200/201, todos los exit codes, request exacto y no exposición de token.

## 8. DVC reproducibility

Validaciones manuales realizadas sobre el checkout actual:

1. Primera reproducción exitosa: upstream hizo skip y `analytics` generó el artifact; DVC actualizó el lock normalmente.
2. Segunda reproducción sin cambios: `Stage 'analytics' didn't change, skipping` y `Data and pipelines are up to date`.
3. Reconstrucción: se movió únicamente `dashboard_summary.json` a un respaldo temporal, `dvc repro analytics` detectó el output ausente y lo regeneró.
4. Los bytes del respaldo y del archivo reconstruido fueron idénticos.

El DAG final contiene:

```text
collect -> validate -> preprocess -> qualify -> train -> evaluate -> analytics
```

y dependencias directas adicionales desde los artifacts descriptivos necesarios.

## 9. Determinismo

Confirmado por test unitario y por reconstrucción DVC real. Para el checkout actual:

```text
SHA-256 dashboard_summary.json:
e8a77e9bb8869d0794a568fe6bf627d4e4381cb357e6b6b6df624a15c0fd2046
```

No se usan reloj, UUID, random, hostname, PID ni timestamp de ejecución. El temporal se crea en el mismo directorio, se escribe por completo, hace `flush` + `fsync` y solo entonces usa `os.replace`. Un test simula fallo del replace y confirma que el output previo permanece intacto.

## 10. Seguridad

- El token se obtiene solo del entorno y el dataclass lo excluye de `repr`.
- El publisher nunca imprime headers, token ni payload completo.
- Errores de transporte usan mensajes genéricos seguros.
- Si el backend reflejara accidentalmente el token en su mensaje de error, el cliente lo reemplaza por `<redacted>`.
- Tests verifican que el token no aparezca en excepción, stdout, stderr ni logs.
- `ml/.env.example` contiene únicamente un placeholder vacío; no se commiteó ningún secret.
- No se ejecutó el publisher contra red real.

## 11. Cambios de dependencias

Se añadieron dependencias directas, sin upgrades generales:

```text
httpx>=0.28,<0.29       -> lock 0.28.1
pydantic>=2.13,<3       -> lock 2.13.5
```

`httpx` implementa el cliente explícito y permite tests con transporte simulado. `pydantic` define un único contrato estricto compartido por generator y publisher. Aunque Pydantic ya llegaba transitivamente mediante MLflow, ahora se declara de forma directa porque el código propio depende de él.

## 12. Issues encontrados

- La ejecución DVC dentro del sandbox restringido no pudo abrir su base local de estado. Se ejecutaron `dvc dag`/`dvc repro` con acceso autorizado al estado/cache de DVC; la etapa en sí solo leyó inputs locales y escribió el artifact esperado.
- El lint completo `ruff check ml/src ml/tests` reporta 77 incidencias preexistentes en módulos/tests no relacionados (imports sin usar/orden, simplificaciones y catches amplios). No se modificaron incidentalmente. Los tres módulos nuevos y sus dos tests unitarios pasan Ruff sin incidencias.
- El entorno virtual no tenía `httpx` ni Ruff instalados aunque la pila traía Pydantic transitivamente. `httpx` quedó declarado/pinneado; Ruff se instaló solo como herramienta local de validación y no se añadió como dependencia runtime.

No se encontró ningún bug upstream bloqueante. Los artifacts actuales coinciden con los valores de referencia del plan.

## 13. Preparación para Fase 3

El backend debe implementar exactamente:

```http
POST /api/v1/analytics/snapshots
Authorization: Bearer <ANALYTICS_PUBLISH_TOKEN>
Content-Type: application/json
Accept: application/json
```

Request body: objeto `AnalyticsSummary` schema `1.0` sin wrapper, con `metadata.dvc_yaml_hash` y sin `dvc_revision`/`generated_at`.

Success response para HTTP 201 o 200:

```json
{
  "artifact_hash": "<sha256 lowercase de 64 caracteres>",
  "schema_version": "1.0",
  "status": "created",
  "stored_at": "2026-09-22T18:30:00Z"
}
```

`status` debe ser `created`, `replaced` o `unchanged`; `stored_at` debe ser RFC3339 UTC. Errores 4xx y 5xx deben usar un mensaje seguro. El backend no debe asumir que `model` describe al champion desplegado: describe la evaluación reproducible asociada al snapshot recibido.

El artifact real ya está listo para una publicación inicial cuando Fase 3 implemente el endpoint. No hace falta recalcular ML para ese bootstrap.

HOME_ANALYTICS_PHASE2_ML_STATUS=COMPLETE
