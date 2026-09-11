# MODEL_PROVIDER — REPORTE SUBETAPA 1: Robustecimiento de Promoción y Asignación de Champion

**Fecha:** 2026-09-11  
**Repositorio:** `PDS-Microproyecto-Grupo-13/Microproyecto`  
**Tracking URI:** `http://localhost:5000`  
**Modelo:** `salary-predictor`  
**Versión Promovida:** `1`  
**Alias Asignado:** `champion`  

---

## 1. Resumen Ejecutivo

En esta Subetapa 1 se robusteció la gobernanza de promoción de modelos en [model_provider/scripts/promote_model.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/scripts/promote_model.py) para garantizar que ninguna versión pueda recibir un alias crítico (como `champion`) sin cumplir estrictamente con las precondiciones de calidad y elegibilidad establecidas en las Fases 3 a 7.

Tras verificar las salvaguardas mediante una suite de pruebas unitarias al 100% (27 tests aprobados en [model_provider/tests](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/tests)), se procedió a la promoción formal de `salary-predictor:1` al alias `champion` en el servidor de MLflow activo. Se verificó el comportamiento idempotente del CLI y la ausencia de efectos secundarios sobre el contenedor de inferencia o el resto del monorepo.

---

## 2. Robustecimiento de `promote_model.py`

Se rediseñó la función central `promote_model_version` en [model_provider/scripts/promote_model.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/scripts/promote_model.py) para implementar las 4 precondiciones obligatorias de gobernanza:

### Precondiciones Validadas
1. **Existencia del Modelo Registrado:**
   - Se consulta `client.get_registered_model(name=model_name)`.
   - Si no existe, se captura `MlflowException`, se emite evento estructurado `registered_model_not_found` y se aborta lanzando `ValueError`.
2. **Existencia de la Versión Solicitada:**
   - Se consulta `client.get_model_version(name=model_name, version=str(version))`.
   - Si la versión no existe, se captura `MlflowException`, se emite `model_version_not_found` y se aborta lanzando `ValueError`.
3. **Estado de Registro `READY`:**
   - Se valida `model_version.status.upper() == "READY"`.
   - Si el estado es preliminar o fallido (`PENDING_REGISTRATION`, `FAILED_REGISTRATION`), se emite `model_version_not_ready` y se aborta con `ValueError`.
4. **Tag Obligatorio `eligible == "true"`:**
   - Se valida `model_version.tags.get("eligible") == "true"`.
   - Si el tag no está presente o su valor es diferente a `"true"`, se emite `model_version_not_eligible` y se aborta con `ValueError`.

### Comportamiento Idempotente y Seguro
- **Idempotencia:** Antes de mutar el registro, se consulta el alias mediante `client.get_model_version_by_alias(name=model_name, alias=alias)`. Si el alias ya apunta a la versión solicitada (`prev_version == target_version_str`), el script no invoca `set_registered_model_alias` de forma innecesaria, emite el evento estructurado con `result="already_assigned"` y retorna limpiamente.
- **Atomicidad y No Afectación ante Fallos:** Ante cualquier fallo en las validaciones, `client.set_registered_model_alias` **nunca** es ejecutado, impidiendo la corrupción de punteros o la pérdida de un alias previo.
- **Desuso de Stages Legados:** El flujo opera exclusivamente con la API moderna de Aliases de MLflow (`set_registered_model_alias`), sin manipular los stages obsoletos (`Staging`, `Production`, `Archived`).
- **Logging Estructurado Enriquecido:** Todos los eventos registran contexto operacional consistente con campos como:
  - `timestamp`, `level`, `event`
  - `model`, `alias`, `from_version`, `to_version`, `result`

---

## 3. Pruebas Unitarias y Cobertura

Se actualizaron los fixtures en [model_provider/tests/conftest.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/tests/conftest.py) para incluir `tags={"eligible": "true"}` y se extendió la suite en [model_provider/tests/test_promote_model.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/tests/test_promote_model.py) cubriendo los siguientes escenarios:

| Test Case | Propósito / Validación |
| :--- | :--- |
| `test_promote_model_version_success_with_previous_alias` | Reasignación correcta de alias desde versión 1 a versión 2 |
| `test_promote_model_version_initial_promotion_no_prev_alias` | Asignación inicial de alias cuando no existía versión previa |
| `test_promote_model_version_idempotent_already_assigned` | Idempotencia: no muta cuando el alias ya está asignado a la misma versión |
| `test_promote_model_version_registered_model_not_found` | Rechazo con `ValueError` cuando el modelo no existe en el Registry |
| `test_promote_model_version_target_version_not_found` | Rechazo con `ValueError` cuando la versión no existe |
| `test_promote_model_version_rejects_status_not_ready` | Rechazo cuando `status != "READY"` (ej. `FAILED_REGISTRATION`) |
| `test_promote_model_version_rejects_missing_eligible_tag` | Rechazo cuando el tag `eligible` no existe en la versión |
| `test_promote_model_version_rejects_eligible_false` | Rechazo explícito cuando `eligible == "false"` |
| `test_promote_model_version_set_alias_failure` | Manejo de excepción de MLflow al asignar el alias (`RuntimeError`) |

### Resultado de Ejecución de Pruebas
Comando: `ml/.venv/bin/pytest model_provider/tests -v`
```text
============================= test session starts ==============================
platform linux -- Python 3.11.2, pytest-8.4.2, pluggy-1.6.0
rootdir: /home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider
configfile: pyproject.toml
collected 27 items

model_provider/tests/test_healthcheck.py::test_check_health_success PASSED [  3%]
model_provider/tests/test_healthcheck.py::test_check_health_fallback_to_ping PASSED [  7%]
model_provider/tests/test_healthcheck.py::test_check_health_all_endpoints_fail PASSED [ 11%]
model_provider/tests/test_model_info.py::test_format_timestamp PASSED    [ 14%]
model_provider/tests/test_model_info.py::test_inspect_model_general_success PASSED [ 18%]
model_provider/tests/test_model_info.py::test_inspect_model_by_alias_success PASSED [ 22%]
model_provider/tests/test_model_info.py::test_inspect_model_by_version_success PASSED [ 25%]
model_provider/tests/test_model_info.py::test_inspect_model_not_found_exits PASSED [ 29%]
model_provider/tests/test_model_info.py::test_inspect_model_alias_not_found_exits PASSED [ 33%]
model_provider/tests/test_promote_model.py::test_promote_model_version_success_with_previous_alias PASSED [ 37%]
model_provider/tests/test_promote_model.py::test_promote_model_version_initial_promotion_no_prev_alias PASSED [ 40%]
model_provider/tests/test_promote_model.py::test_promote_model_version_idempotent_already_assigned PASSED [ 44%]
model_provider/tests/test_promote_model.py::test_promote_model_version_registered_model_not_found PASSED [ 48%]
model_provider/tests/test_promote_model.py::test_promote_model_version_target_version_not_found PASSED [ 51%]
model_provider/tests/test_promote_model.py::test_promote_model_version_rejects_status_not_ready PASSED [ 55%]
model_provider/tests/test_promote_model.py::test_promote_model_version_rejects_missing_eligible_tag PASSED [ 59%]
model_provider/tests/test_promote_model.py::test_promote_model_version_rejects_eligible_false PASSED [ 62%]
model_provider/tests/test_promote_model.py::test_promote_model_version_set_alias_failure PASSED [ 66%]
model_provider/tests/test_start.py::test_load_config_from_env_valid PASSED [ 70%]
model_provider/tests/test_start.py::test_load_config_from_env_missing_model_name PASSED [ 74%]
model_provider/tests/test_start.py::test_load_config_from_env_invalid_port PASSED [ 77%]
model_provider/tests/test_start.py::test_wait_for_tracking_server_success PASSED [ 81%]
model_provider/tests/test_start.py::test_wait_for_tracking_server_failure PASSED [ 85%]
model_provider/tests/test_start.py::test_resolve_model_info_success PASSED [ 88%]
model_provider/tests/test_start.py::test_resolve_model_info_missing_registered_model PASSED [ 92%]
model_provider/tests/test_start.py::test_resolve_model_info_missing_alias PASSED [ 96%]
model_provider/tests/test_start.py::test_build_serve_command PASSED      [100%]

============================== 27 passed in 3.83s ==============================
```

---

## 4. Promoción Real en MLflow y Verificación de Idempotencia

### 4.1. Estado previo de MLflow Model Registry
```text
Registered Model: salary-predictor
Aliases:          {}
Version 1:        Status: READY, Tags: {'candidate': 'true', 'eligible': 'true', ...}
```

### 4.2. Ejecución de la Promoción
```bash
ml/.venv/bin/python model_provider/scripts/promote_model.py \
  --model salary-predictor \
  --version 1 \
  --alias champion \
  --tracking-uri http://localhost:5000
```
Salida en stdout:
```text
2026-09-11T06:17:41Z [INFO] event=model_promotion_started model=salary-predictor version=1 alias=champion tracking_uri=http://localhost:5000
2026-09-11T06:17:41Z [INFO] event=previous_alias model=salary-predictor alias=champion version=none
2026-09-11T06:17:41Z [INFO] event=model_promoted model=salary-predictor alias=champion from_version=none to_version=1 result=promoted
```

### 4.3. Verificación de Idempotencia (Re-ejecución)
Al invocar nuevamente el comando idéntico:
```text
2026-09-11T06:17:46Z [INFO] event=model_promotion_started model=salary-predictor version=1 alias=champion tracking_uri=http://localhost:5000
2026-09-11T06:17:46Z [INFO] event=previous_alias model=salary-predictor alias=champion version=1
2026-09-11T06:17:46Z [INFO] event=model_promoted model=salary-predictor alias=champion from_version=1 to_version=1 result=already_assigned
```
El log confirmó `result=already_assigned` y no se realizaron mutaciones redundantes en MLflow.

### 4.4. Verificación de Salvaguarda ante Entradas Inválidas
Se probó promover una versión inexistente (`version 99`):
```bash
ml/.venv/bin/python model_provider/scripts/promote_model.py \
  --model salary-predictor \
  --version 99 \
  --alias champion \
  --tracking-uri http://localhost:5000
```
Salida:
```text
2026-09-11T06:17:49Z [INFO] event=model_promotion_started model=salary-predictor version=99 alias=champion tracking_uri=http://localhost:5000
2026-09-11T06:17:50Z [ERROR] event=model_version_not_found model=salary-predictor version=99 error=...
2026-09-11T06:17:50Z [ERROR] event=model_promotion_failed error=Model version '99' for registered model 'salary-predictor' does not exist
```
El script terminó con exit code 1 y el alias `champion` permaneció intacto apuntando a la versión `1`.

---

## 5. Estado Confirmado en MLflow tras la Promoción

Inspección realizada con [model_provider/scripts/model_info.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/scripts/model_info.py):

```bash
ml/.venv/bin/python model_provider/scripts/model_info.py --model salary-predictor --tracking-uri http://localhost:5000
```
```text
=================================================================
 Registered Model: salary-predictor
 Description:      No description
 Created:          2026-09-11 00:03:26 UTC
 Last Updated:     2026-09-11 00:03:26 UTC
 Aliases:          {'champion': '1'}
=================================================================

Registered Model Versions:
  • Version 1 (Status: READY) [Aliases: champion]
    - Run ID:   db1fd1c17c8e499d976c8446fe17e3ec
    - Source:   models:/m-3cc347ba70af4dc782ba5aa2a9b1c5b8
    - Created:  2026-09-11 00:03:26 UTC
```

Detalles de la versión promovida:
- **Model Name:** `salary-predictor`
- **Version:** `1`
- **Alias:** `champion` -> `1`
- **Status:** `READY`
- **Tags:**
  - `candidate`: `true`
  - `eligible`: `true`
  - `algorithm`: `lightgbm`
  - `dataset_fingerprint`: `541f47bf1bb49a6a0e5947d49dade61c444e74cd5350cfbb68a185506cd957ed`
  - `primary_metric`: `mae_promedio`
  - `primary_metric_value`: `27081.221469`
  - `run_id`: `db1fd1c17c8e499d976c8446fe17e3ec`

---

## 6. Desacoplamiento Operacional: Promoción vs. Serving

Se confirmó expresamente que:
1. **Promover NO es Desplegar:** La promoción en MLflow es exclusivamente una operación de metadatos en la base de datos de backend de MLflow Tracking.
2. **Sin Afectación a Contenedores en Ejecución:** No se reinició el contenedor `mlops-inference`, ni se modificaron `docker-compose.yml`, `Dockerfile` o variables de entorno del sistema de serving.
3. **Dualidad CLI vs. UI de MLflow:**
   - **CLI (`promote_model.py`):** Aplica programáticamente las salvaguardas de gobernanza (`status == READY` y `eligible == true`).
   - **MLflow UI:** Permite a un operador humano mover alias manualmente en la interfaz web sin ejecutar validaciones de código locales, aunque el estado resultante en la base de datos de MLflow es idéntico (`{'champion': '1'}`).

---

## 7. Alcance y Restricciones Cumplidas

- **ml/ intacto:** No se alteraron pipelines, modelos `model.joblib`, ni artefactos de tracking.
- **backend/ y frontend/ intactos:** No se realizaron modificaciones en clientes HTTP ni esquemas de datos.
- **Serving intacto:** No se alteró la configuración del servicio de inferencia ni se inició deployment en esta etapa.

---

MODEL_PROVIDER_STAGE_1_STATUS=READY_FOR_STAGE_2
