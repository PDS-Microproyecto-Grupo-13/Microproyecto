# Home Analytics — Phase 3 Backend Report

Fecha: 2026-09-22

## 1. Resumen

Se implementó exclusivamente la capa backend de Home Analytics:

- contrato Pydantic estricto `AnalyticsSummary` schema `1.0` compatible con el
  artifact efectivo de Fase 2;
- publicación administrativa autenticada mediante
  `POST /api/v1/analytics/snapshots`;
- lectura pública mediante `GET /api/v1/analytics/summary`;
- persistencia atómica de la última publicación en storage propio del backend;
- hash canónico e idempotencia `created` / `replaced` / `unchanged`;
- named volume Docker persistente;
- tests unitarios, de API, storage, seguridad e idempotencia;
- publicación HTTP real desde el publisher de Fase 2 y validación de
  persistencia tras recrear el backend.

El backend no accede a ningún path de `ml/`, no lee CSV/Parquet, no ejecuta DVC,
no recalcula estadísticas y no consulta MLflow para Home Analytics.

Archivos modificados:

- `.env.example`
- `backend/.env.example`
- `backend/README.md`
- `backend/app/api/router.py`
- `backend/app/core/config.py`
- `backend/app/core/exceptions.py`
- `backend/tests/conftest.py`
- `backend/tests/test_predictions.py`
- `docker-compose.yml`

Archivos nuevos:

- `backend/app/api/routes/analytics.py`
- `backend/app/schemas/analytics.py`
- `backend/app/services/analytics_service.py`
- `backend/tests/test_analytics.py`
- `HOME_ANALYTICS_PHASE3_BACKEND_REPORT.md`

El ajuste en `test_predictions.py` convierte dos overrides de dependencias de
test en async para evitar el bloqueo del threadpool observado con las versiones
locales de FastAPI/AnyIO; no cambia comportamiento productivo.

## 2. Endpoints implementados

### POST `/api/v1/analytics/snapshots`

Recibe `AnalyticsSummary` sin wrapper y requiere:

```http
Authorization: Bearer <ANALYTICS_PUBLISH_TOKEN>
Content-Type: application/json
```

Resultados:

| HTTP | `status` | Significado |
|---:|---|---|
| 201 | `created` | Primera publicación |
| 200 | `replaced` | Snapshot válido diferente |
| 200 | `unchanged` | Mismo contenido canónico; no se reescribe |

Respuesta:

```json
{
  "artifact_hash": "<sha256 canónico>",
  "schema_version": "1.0",
  "status": "created",
  "stored_at": "2026-09-22T12:26:10.952033Z"
}
```

### GET `/api/v1/analytics/summary`

- snapshot válido: `200` y body exacto `AnalyticsSummary`;
- primer arranque sin snapshot: `404 analytics_not_published`, comportamiento
  esperado para el bootstrap;
- storage corrupto/no legible: `503 analytics_storage_unavailable`;
- todas las respuestas GET usan `Cache-Control: no-store`;
- no se implementaron ETag, `max-age`, 304 ni fallbacks mock.

## 3. Schema y validaciones

El contrato acepta exclusivamente `schema_version: "1.0"` y objetos cerrados.
Mantiene exactamente `metadata.dvc_yaml_hash`; no acepta `dvc_revision` ni
`generated_at`.

Se validan tipos estrictos, números finitos, rangos, hashes lowercase, timestamps
RFC3339 UTC, orden temporal, límites salariales, bins contiguos desde cero,
único último bin abierto, sumas poblacionales, proporciones redondeadas a seis
decimales, categorías únicas y ordenadas, y technologies canónicas únicas y
ordenadas. La lista de technologies refleja las claves `SKILLS` efectivas de
Fase 2.

El objeto `model` representa métricas de evaluación asociadas al snapshot
analítico. No significa ni prueba que el modelo sea el `champion`, una versión
desplegada o el modelo servido actualmente.

Los 422 reutilizan `app.schemas.common.ErrorResponse`. Se corrigió el handler
global para serializar de forma segura el `ctx` de errores Pydantic complejos,
evitando que una invariante cruzada produjera accidentalmente 500.

## 4. Persistencia e idempotencia

Configuración por defecto:

```text
ANALYTICS_STORAGE_PATH=/app/data/analytics_summary.json
```

Envelope interno:

```json
{
  "artifact_hash": "<sha256 canónico>",
  "storage_version": 1,
  "stored_at": "<RFC3339 UTC>",
  "summary": {"schema_version": "1.0"}
}
```

La escritura crea un temporal en el mismo directorio, escribe el contenido
completo, ejecuta `flush` y `fsync`, cierra el archivo y usa `os.replace`. Si la
escritura/reemplazo falla, el snapshot anterior permanece intacto. La lectura
revalida tanto el envelope como el summary y comprueba nuevamente su hash.

El hash se calcula con SHA-256 sobre JSON validado, UTF-8, claves ordenadas,
separadores compactos, Unicode sin escape forzado y sin NaN. Por ello el hash
backend es un hash canónico y no necesariamente el SHA-256 físico del archivo
pretty-printed de ML. En la integración:

```text
SHA físico artifact ML: e8a77e9bb8869d0794a568fe6bf627d4e4381cb357e6b6b6df624a15c0fd2046
Hash canónico backend:  363e4e613e45755e809465ac94bb6d603fa3f9da05a52ebf6a587ae1c600b037
```

Una publicación `unchanged` conserva el mismo `stored_at` y no reescribe el
archivo.

## 5. Autenticación

`ANALYTICS_PUBLISH_TOKEN` es obligatorio únicamente para POST. El token recibido
se compara con `secrets.compare_digest` sobre bytes UTF-8.

- ausente o incorrecto: `401 analytics_unauthorized`;
- backend sin token configurado: `503 analytics_publish_unavailable`;
- el token se representa como `SecretStr` en settings;
- headers, tokens y payloads completos no se registran;
- tests confirman que un token incorrecto no aparece en logs ni respuestas.

GET es público y read-only.

## 6. Configuración Docker

Docker Compose inyecta:

```text
ANALYTICS_PUBLISH_TOKEN
ANALYTICS_STORAGE_PATH=/app/data/analytics_summary.json
```

Y monta exclusivamente en backend:

```yaml
backend:
  volumes:
    - backend-analytics-data:/app/data

volumes:
  backend-analytics-data:
    name: mlops-backend-analytics-data
```

No existe ningún volumen compartido con `ml/`. `docker compose config` validó la
configuración efectiva. La imagen actual ejecuta como root y pudo escribir el
volumen sin cambios de Dockerfile; ejecutar como usuario no privilegiado queda
como hardening futuro, tal como permite el alcance.

## 7. Tests

Resultados:

```text
pytest -q tests/test_analytics.py
17 passed

MODEL_NAME=salary_predict_model pytest -q
39 passed

ruff check <archivos Phase 3>
All checks passed

ruff format --check <archivos Phase 3>
9 files already formatted

git diff --check
sin errores

docker compose config
exit 0

docker compose build backend
exit 0
```

La variable `MODEL_NAME` se fijó al ejecutar la suite porque el archivo local
`backend/.env` del checkout contiene `salary-predictor` y sobreescribe el default
que dos tests históricos esperan. No se modificó ese archivo local.

La cobertura nueva incluye POST inicial/idéntico/diferente, auth ausente e
incorrecta, token backend no configurado, extras, versión, tipos e invariantes
inválidas, GET vacío/válido/corrupto, fallo de replace conservando el snapshot,
recreación de service, hash canónico, no reescritura y no exposición de tokens.

El lint completo conserva incidencias preexistentes fuera de Fase 3: orden de
imports en `app/clients/inference_client.py` y formato en
`app/api/routes/predictions.py` / `tests/test_clients.py`. No se corrigieron
incidentalmente.

## 8. Publicación real desde Fase 2

No se ejecutó `dvc repro`; se utilizó directamente
`ml/artifacts/reports/dashboard_summary.json` sin modificarlo.

Flujo real ejecutado contra `localhost:8000`:

```text
GET antes de publicar -> 404 analytics_not_published
publish-analytics #1 -> HTTP 201, status=created
publish-analytics #2 -> HTTP 200, status=unchanged
```

Ambas publicaciones devolvieron el mismo hash canónico y el segundo POST conservó
el `stored_at` del primero. El publisher terminó con exit code 0 en ambos casos.
No se ejecutaron entrenamiento, track, register-candidate, promote ni operaciones
MLflow.

## 9. Resultado de GET

Después de publicar:

```text
GET /api/v1/analytics/summary -> 200
Cache-Control: no-store
```

Una comparación JSON profunda confirmó igualdad semántica exacta entre el body
GET y `dashboard_summary.json`, incluidos `metadata.dvc_yaml_hash`, 54.363 filas
modelables, 8.155 filas de evaluación y las ocho technologies publicadas.

## 10. Persistencia tras recreación

Se validó en dos niveles:

1. Se reinició el proceso backend conservando el mismo storage; GET continuó en
   200 y el body siguió siendo semánticamente idéntico al artifact.
2. Se publicó el artifact real en un contenedor con
   `mlops-backend-analytics-data`, se eliminó el contenedor y se creó otro con el
   mismo named volume; GET devolvió 200 con el snapshot completo.

Los contenedores temporales de verificación fueron eliminados. El named volume
permanece y conserva el snapshot, por lo que sobrevive recreaciones normales y
`docker compose down/up`; `docker compose down -v` sí lo elimina, según contrato.

## 11. Issues encontrados

- El sandbox restringido bloqueó conexiones incluso a loopback entre procesos.
  La integración HTTP local solicitada se repitió con autorización fuera del
  sandbox y terminó correctamente.
- Dos overrides síncronos históricos del cliente de inferencia bloquearon el
  threadpool con la combinación local FastAPI/AnyIO. Se cambiaron únicamente en
  tests a overrides async equivalentes para poder completar la suite.
- El `.env` local redefine `MODEL_NAME`, por lo que la suite histórica necesita
  fijar explícitamente el valor contractual durante la ejecución.
- La imagen backend continúa ejecutando como root. Es funcional con el volumen
  y no se amplió el alcance para refactorizar permisos/usuario.

No se encontraron problemas de contrato, atomicidad o integración bloqueantes.

## 12. Contrato listo para Fase 4

Frontend debe consumir únicamente:

```http
GET /api/v1/analytics/summary
Accept: application/json
```

- `200`: renderizar el `AnalyticsSummary` schema `1.0` recibido;
- `404` con `error=analytics_not_published`: mostrar estado vacío explícito;
- `503` con `error=analytics_storage_unavailable`: mostrar error temporal;
- otros errores/transporte: usar el patrón HTTP global del frontend;
- manejar loading sin mocks ni fallbacks;
- respetar que `model` son métricas de evaluación del snapshot, no del champion;
- no conocer DVC, MLflow, artifacts ni el mecanismo POST de publicación.

El backend está listo para Fase 4 con `Cache-Control: no-store`, snapshot real
persistido y contrato GET estable.

HOME_ANALYTICS_PHASE3_BACKEND_STATUS=COMPLETE
