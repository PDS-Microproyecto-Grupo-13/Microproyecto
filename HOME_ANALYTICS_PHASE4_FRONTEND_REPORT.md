# Home Analytics — Phase 4 Frontend Report

Fecha: 2026-09-22

## 1. Archivos modificados

Archivos existentes modificados:

- `frontend/README.md`
- `frontend/src/pages/Home/HomePage.tsx`
- `frontend/src/pages/Home/HomePage.module.css`
- `frontend/src/services/salaryApi.ts`

Archivos nuevos:

- `frontend/src/services/analyticsApi.ts`
- `frontend/src/services/http.ts`
- `HOME_ANALYTICS_PHASE4_FRONTEND_REPORT.md`

No se modificaron backend, ML, model provider, Docker Compose, configuración raíz,
DEPLOYMENT ni reportes anteriores.

## 2. Service y tipos implementados

Se agregó:

```text
getAnalyticsSummary({ signal? })
-> GET /api/v1/analytics/summary
-> Promise<AnalyticsSummary>
```

El request envía `Accept: application/json` y admite `AbortSignal`. No existe
ninguna función POST de analytics.

`analyticsApi.ts` representa exactamente el contrato efectivo schema `1.0`:

- dataset, conteos, rango temporal, salary midpoint e histograma;
- distribuciones de seniority y work mode;
- tecnologías con count/proportion;
- métricas de evaluación del modelo;
- metadata con `dataset_fingerprint`, `snapshot_count`, `git_commit`,
  `dvc_yaml_hash` y `params_hash`.

No se introdujeron `dvc_revision`, `generated_at` ni campos inventados.

Se creó un helper HTTP común con `ApiError`, que conserva `status`, código
estable `error` y `request_id` del backend. `salaryApi.ts` reutiliza este helper
sin cambiar su request/response funcional, por lo que Home puede distinguir
`404 analytics_not_published` de 503, red y otros errores sin comparar mensajes
libres.

## 3. Estados de Home

Home implementa explícitamente:

- **loading**: panel "Cargando analítica", sin cifras ni fallbacks;
- **success**: dashboard completo construido desde `AnalyticsSummary`;
- **unpublished**: `404 analytics_not_published` muestra
  "Analítica aún no publicada" y botón de reintento;
- **error**: 503 storage, otros HTTP o transporte muestran un error y botón de
  reintento.

Cada reintento ejecuta un GET nuevo. No existe cache propio, localStorage ni
persistencia en frontend. El cleanup del efecto ejecuta `AbortController.abort()`
y mantiene además un guard `active` para ignorar resultados tardíos después del
unmount.

## 4. Mapeo de mocks a datos reales

| Elemento anterior | Sustitución real | Valor del snapshot validado |
|---|---|---:|
| Ofertas ingeridas `1,450+` | Vacantes modelables (`modelable_rows`) | 54.363 |
| Salario medio `$52,800` | Punto medio mediano anual (`median_usd`) | US$ 163.000 |
| Roles clasificados `18` | MAE promedio test (`mae_average_usd`) | US$ 27.081 |
| R² `94.2%` | Rango R² de evaluación | 57,3% – 62,8% |
| Barras placeholder | `salary_midpoint_distribution` | 6 bins reales |
| Tecnologías/conteos mock | `top_technologies` | 8 menciones canónicas |
| Tres ofertas inventadas | `seniority_distribution` y rango temporal | 5 categorías reales |

La primera tarjeta presenta además `validated_rows` y `snapshot_count`. No
rotula `raw_snapshot_rows` como vacantes únicas. La segunda muestra la media solo
como contexto secundario. MAE y R² se etiquetan explícitamente como evaluación
del snapshot, no como accuracy ni métricas del champion.

## 5. Elementos mock eliminados

Se eliminaron de Home:

- `1,450+` y "Ofertas Ingeridas";
- `+12.5% este mes` y toda tendencia temporal falsa;
- `$52,800`;
- "Roles Clasificados" y el conteo 18;
- `R² 94.2%`;
- histograma/alturas placeholder;
- React, TypeScript y todos los conteos mock de tecnologías;
- las ofertas ficticias Senior MLOps Engineer, Full Stack Developer y Data
  Scientist, junto con sus salarios inventados;
- el badge ambiguo "ML Model v1.0".

El badge ahora es `Analytics v1.0`, derivado de `schema_version` únicamente
después de recibir el summary.

## 6. Formateo y presentación

La UI usa `Intl.NumberFormat` para enteros, USD y porcentajes, y
`Intl.DateTimeFormat` UTC para el rango de publicación. Con el snapshot actual:

```text
modelable_rows       -> 54.363
median_usd           -> US$ 163.000
mae_average_usd      -> US$ 27.081
R² min – max         -> 57,3% – 62,8%
```

El histograma utiliza exclusivamente bins y counts recibidos; frontend calcula
solo la altura relativa al bin máximo y labels de presentación. Tecnologías se
renderizan desde el payload con nombre legible, count y proportion. La etiqueta
aclara "Menciones detectadas en vacantes modelables", no feature importance ni
ontología exhaustiva.

La tabla ficticia se reemplazó por seniority agregado con los mappings de
presentación `SE`, `MI`, `EN`, `EX` y `desconocido`, más el rango temporal real.

## 7. Tests, build y lint

El proyecto no tenía framework ni scripts de tests frontend. Añadir Vitest,
Testing Library y sus dependencias solo para esta fase habría ampliado de forma
desproporcionada el tooling; no se añadieron dependencias.

Validaciones ejecutadas:

```text
npm ci
added 32 packages

npm run lint
Oxlint: exit 0, sin warnings

npm run build
TypeScript tsc -b: exit 0
Vite production build: exit 0
1835 módulos transformados

git diff --check
exit 0
```

El build también confirmó que los strings mock antiguos no aparecen en el bundle
JavaScript de Home.

## 8. Validación contra backend real

Se levantaron backend y Vite localmente y se consultó analytics a través del
proxy real del frontend:

```text
GET http://127.0.0.1:5173/                         -> 200
GET http://127.0.0.1:5173/api/v1/analytics/summary -> 200
```

La respuesta efectiva coincidió con el snapshot publicado y se comprobaron
explícitamente:

```text
schema_version=1.0
modelable_rows=54363
evaluation_rows=8155
dvc_yaml_hash=642fe55786ad977f3bf0a1462384c90c5269b3f472adccbae36901d60c5d461b
```

El navegador integrado no estaba disponible en esta sesión, por lo que no fue
posible capturar una inspección visual automatizada. No se sustituyó por un
navegador externo no autorizado. La compilación React/TypeScript, el proxy Vite,
el contrato real 200 y las fuentes renderizadas sí quedaron validados.

## 9. Comportamiento 404 unpublished

Se reinició un backend temporal con storage vacío, sin destruir el snapshot real,
y se consultó mediante el mismo proxy Vite:

```text
GET /api/v1/analytics/summary
-> 404
-> error=analytics_not_published
```

Home traduce exclusivamente esa combinación tipada al estado
"Analítica aún no publicada". No muestra ceros, cards antiguas ni valores mock.
El backend vacío puede arrancar antes del bootstrap; tras publicar y reintentar o
recargar, Home consulta nuevamente el GET y pasa a success.

## 10. Issues encontrados

- No existía infraestructura de tests frontend; se optó por lint, type-check,
  build e integración real en lugar de incorporar un stack de testing nuevo.
- El navegador integrado no estaba disponible para validación visual en esta
  sesión.
- El primer intento de integración consultó Vite antes de que Uvicorn terminara
  de iniciar; el flujo final añadió readiness explícito y validó 200/404.

No se encontró ninguna insuficiencia en el contrato backend ni un bloqueo para
Home.

## 11. Fronteras arquitectónicas

- Frontend consume únicamente `GET /api/v1/analytics/summary` para analytics.
- Frontend no conoce ni invoca el POST administrativo.
- Frontend no conoce ni maneja el token de publicación.
- Home Analytics no consulta DVC, MLflow, artifacts ni rutas de `ml/`.
- No existe request directo desde frontend hacia MLflow o el filesystem.
- `model` se presenta como evaluación asociada al snapshot, nunca como champion,
  producción, versión desplegada o modelo servido.
- No existen fallbacks hardcoded para loading, unpublished o error.
- El flujo de predicción existente continúa usando su endpoint backend
  `/api/v1/predictions`.

HOME_ANALYTICS_PHASE4_FRONTEND_STATUS=COMPLETE
