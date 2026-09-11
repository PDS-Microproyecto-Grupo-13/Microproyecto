# FASE 7 — MLflow Model Registry del Modelo Salarial Candidato

## 1. Resumen Ejecutivo

En la **Fase 7**, se ha completado el ciclo de registro formal del modelo salarial candidato en el **MLflow Model Registry** del servidor activo (`http://localhost:5000`), ejecutando la CLI `python -m ml_pipeline register-candidate`:

$$\text{artifacts/work/model/model.joblib} \longrightarrow \text{MLflow Run (Tracking: Fase 6)} \longrightarrow \text{Model Registry (salary-predictor:v1)}$$

### Principios y Reglas de Gobernanza Cumplidas:
1. **Inviolabilidad del Modelo y del Run de Tracking**:
   - Se tomó **exactamente** el modelo empaquetado en el Run de la Fase 6 (`run_id`: `db1fd1c17c8e499d976c8446fe17e3ec`, `model_uri`: `models:/m-3cc347ba70af4dc782ba5aa2a9b1c5b8`).
   - No se reentrenó, no se recalibró y no se volvieron a evaluar datos sobre particiones.
   - No se generaron nuevos Tracking Runs innecesarios.
2. **Validación Exhaustiva de Precondiciones**:
   - Se validó la existencia e integridad de `artifacts/reports/candidate.json` y `artifacts/reports/tracking.json`.
   - Se verificó que ambos reportes certificaran `eligible == true`.
   - Se comprobó la coherencia exacta del `dataset_fingerprint` (`541f47bf1bb49a6a0e5947d49dade61c444e74cd5350cfbb68a185506cd957ed`).
   - Se verificó la existencia del Run en MLflow y la capacidad de resolución y carga del PyFunc fuente.
3. **Idempotencia y Prevención de Duplicados**:
   - Se implementó detección proactiva de versiones registradas por `run_id` y `model_uri`. Re-ejecuciones de `register-candidate` reutilizan la versión existente sin crear incrementos accidentales (`version 2`).
4. **Delimitación Estricta de Gobernanza (Sin Alias ni Promociones)**:
   - La versión registrada permanece estrictamente como **candidata**.
   - **No se asignaron alias** (`aliases: []`, el alias `champion` está estrictamente prohibido en esta fase).
   - **No se modificaron etapas** (`current_stage: "None"`).
   - No se alteró código en `model_provider/`, `backend/` ni `frontend/`.
5. **Independencia de DVC**:
   - `register-candidate` permanece desacoplado de DVC, manteniendo el pipeline reproducible DVC (`collect -> validate -> preprocess -> qualify -> train -> evaluate`) completamente intacto y limpio (`dvc status: Data and pipelines are up to date`).

---

## 2. Metadatos de Registro y Lineage

La invocación de `python -m ml_pipeline register-candidate` registró formalmente la versión 1 del modelo en MLflow:

| Atributo | Valor Registrado | Observación |
| :--- | :--- | :--- |
| **Model Name** | `salary-predictor` | Definido en `settings.mlflow_model_name` (`ml/.env`) |
| **Model Version** | `1` | Primera versión formal del predictor salarial |
| **Source Run ID** | `db1fd1c17c8e499d976c8446fe17e3ec` | Run originado en Fase 6 |
| **Source Model URI** | `models:/m-3cc347ba70af4dc782ba5aa2a9b1c5b8` | URI PyFunc original del tracking run |
| **Exact Registry URI** | `models:/salary-predictor/1` | URI canónica de la versión registrada |
| **Current Stage** | `None` | Sin transiciones de ciclo de vida legadas |
| **Aliases** | `[]` *(vacío)* | **Ningún alias asignado** (prohibido `champion`) |
| **Status** | `READY` | Modelo disponible y resuelto para inferencia |

### 2.1. Tags Estructurados Aplicados al Model Version
Se aplicaron explícitamente los siguientes tags a la versión registrada:

```json
{
  "candidate": "true",
  "eligible": "true",
  "algorithm": "lightgbm",
  "dataset_fingerprint": "541f47bf1bb49a6a0e5947d49dade61c444e74cd5350cfbb68a185506cd957ed",
  "run_id": "db1fd1c17c8e499d976c8446fe17e3ec",
  "primary_metric": "mae_promedio",
  "primary_metric_value": "27081.221469"
}
```

---

## 3. Verificación de Paridad Numérica e Invariantes

Como parte obligatoria del proceso de registro en [`ml/src/ml_pipeline/tracking/registry.py`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml/src/ml_pipeline/tracking/registry.py), el modelo fue cargado directamente desde su URI canónica de registro (`models:/salary-predictor/1`) y contrastado con el modelo fuente de tracking sobre el `input_example` oficial.

### 3.1. Paridad Numérica Directa

$$\Delta = |\hat{y}_{\text{source}} - \hat{y}_{\text{registry}}|$$

| Target Evaluado | Tolerancia Relativa (`rtol`) | Tolerancia Absoluta (`atol`) | Diferencia Máxima Observada | Estado de Paridad |
| :--- | :---: | :---: | :---: | :---: |
| `salary_min_usd` | $10^{-5}$ | $10^{-5}$ | $0.000000$ USD | **APROBADO (IDÉNTICO)** |
| `salary_max_usd` | $10^{-5}$ | $10^{-5}$ | $0.000000$ USD | **APROBADO (IDÉNTICO)** |
| `salary_midpoint_usd` | $10^{-5}$ | $10^{-5}$ | $0.000000$ USD | **APROBADO (IDÉNTICO)** |

### 3.2. Chequeo de Invariantes Operacionales
- **Valores Finitos**: Todas las predicciones son números reales finitos; sin presencia de valores `NaN`, `+Inf` ni `-Inf`.
- **Valores Positivos**: $100\%$ de las predicciones satisfacen $\hat{y} > 0$.
- **Monotonicidad y Orden de Rango**: $100\%$ de las filas cumplen $\hat{y}_{\min} \le \hat{y}_{\max}$.
- **Punto Medio Exacto**: $\hat{y}_{\text{midpoint}} = \frac{\hat{y}_{\min} + \hat{y}_{\max}}{2}$ con tolerancia numérica absoluta $< 10^{-5}$.
- **Límites Operativos de Entrenamiento**: Todas las predicciones se encuentran acotadas dentro del intervalo canónico $[\text{floor}, \text{ceiling}] = [\$10,935.57, \$720,000.00]$.

---

## 4. Idempotencia y Comportamiento de Re-ejecución (Deduplicación)

Para garantizar la estabilidad en entornos CI/CD y evitar la proliferación descontrolada de versiones huérfanas en el registro ante fallos o reintentos, el pipeline implementa un chequeo previo en `register_candidate`:

```python
versions = client.search_model_versions(f"name = '{model_name}'")
for v in versions:
    if v.run_id == run_id or v.source == model_uri:
        existing_version = str(v.version)
        LOGGER.info("register-candidate | existing version found | model=%s version=%s run_id=%s", model_name, existing_version, run_id)
        break
```

### Verificación Práctica de Idempotencia:
Se ejecutó intencionalmente el comando por segunda vez consecutiva en el entorno real:
```text
2026-09-10 20:03:46,235 | INFO | ml_pipeline.tracking.registry | register-candidate | existing version found | model=salary-predictor version=1 run_id=db1fd1c17c8e499d976c8446fe17e3ec
2026-09-10 20:03:47,189 | INFO | ml_pipeline.tracking.registry | register-candidate | verifying exact registry model | uri=models:/salary-predictor/1
2026-09-10 20:03:48,723 | INFO | ml_pipeline.tracking.registry | register-candidate | result=success | model=salary-predictor version=1 run_id=db1fd1c17c8e499d976c8446fe17e3ec exact_uri=models:/salary-predictor/1
```
- **Resultado**: Reutilizó la versión `1`, re-verificó los tags y la paridad numérica, y actualizó el artefacto local sin crear la versión `2`.
- **Conteo total de versiones en el registro**: Exactamente **1 versión**.

---

## 5. Reporte Local de Registro (`artifacts/reports/registration.json`)

El reporte formal persistido localmente contiene:

```json
{
  "dataset_fingerprint": "541f47bf1bb49a6a0e5947d49dade61c444e74cd5350cfbb68a185506cd957ed",
  "eligible": true,
  "exact_registry_uri": "models:/salary-predictor/1",
  "model_name": "salary-predictor",
  "model_uri": "models:/m-3cc347ba70af4dc782ba5aa2a9b1c5b8",
  "primary_metric": "mae_promedio",
  "primary_metric_value": 27081.221469,
  "run_id": "db1fd1c17c8e499d976c8446fe17e3ec",
  "source": "models:/m-3cc347ba70af4dc782ba5aa2a9b1c5b8",
  "version": "1"
}
```

---

## 6. Resultados de Pruebas Unitarias y de Integración

Se añadieron y actualizaron pruebas específicas para cubrir todos los aspectos del registro en Model Registry.

### 6.1. Pruebas Unitarias (`tests/unit/test_registry.py`)
11 pruebas unitarias cubriendo:
- `test_registration_rejects_missing_candidate_report`: Rechazo si falta `candidate.json`.
- `test_registration_rejects_missing_tracking_report`: Rechazo si falta `tracking.json`.
- `test_registration_rejects_ineligible_candidate`: Rechazo si `candidate.json -> eligible == false`.
- `test_registration_rejects_ineligible_tracking`: Rechazo si `tracking.json -> eligible == false`.
- `test_registration_rejects_missing_run_id_or_model_uri`: Rechazo si faltan punteros críticos de MLflow.
- `test_registration_rejects_fingerprint_mismatch`: Rechazo si hay divergencia de huella entre reportes.
- `test_registration_rejects_nonexistent_run`: Rechazo si el Run ID no existe en MLflow.
- `test_registration_rejects_unresolvable_model`: Rechazo si el modelo no puede cargarse.
- `test_registration_success_end_to_end_and_tags_and_reports`: Registro exitoso, verificación de tags, confirmación de ausencia de alias champion y persistencia de `registration.json`.
- `test_registration_deduplication_prevents_duplicate_versions`: Garantía de idempotencia sin creación de versiones duplicadas.
- `test_registration_parity_failure_raises`: Detección de fallos en caso de corrupción de predicciones o violación de invariantes.

### 6.2. Prueba de Integración (`tests/integration/test_tracking_registry.py`)
Se reactivó y adaptó la prueba de integración completa sobre el ciclo salarial LightGBM:
- Ejecuta las 6 etapas del pipeline sobre datos sintéticos multi-snapshot Foorilla.
- Ejecuta `track` registrando el Run en un backend SQLite aislado.
- Ejecuta `register_candidate` registrando la versión en el Model Registry temporal.
- Valida la paridad numérica completa, tags e idempotencia.

### 6.3. Ejecución Total de la Suite Pytest
```text
pytest tests/ -v
=========================== short test summary info ============================
SKIPPED [2] tests/contract/test_model_contract.py:16: Model contract belongs to post-data migration phases
=========== 84 passed, 2 skipped, 112 warnings in 148.52s (0:02:28) ============
```
- **Total Aprobado**: **84 pruebas exitosas** ($100\%$ de las pruebas ejecutadas).
- **Pruebas Omitidas**: 2 pruebas de contrato reservadas para fases post-migración.
- **Fallos**: **0**.

---

## 7. Estado del Grafo DVC y Repositorio

1. **Estado de DVC (`dvc status`)**:
   ```text
   Data and pipelines are up to date.
   ```
   El pipeline DVC permanece 100% limpio y no fue afectado por las operaciones de tracking ni de registro.
2. **Archivos Modificados en Git**:
   - `ml/src/ml_pipeline/tracking/registry.py`: Lógica de precondiciones, deduplicación, tags y verificación de paridad.
   - `ml/tests/unit/test_registry.py`: Suite exhaustiva de 11 pruebas unitarias.
   - `ml/tests/integration/test_tracking_registry.py`: Prueba de integración extremo a extremo reactivada para el pipeline salarial.
   - `ml/README.md`: Documentación de la Fase 7 y actualización de la guía de pruebas.
   - `ml/artifacts/reports/registration.json`: Artefacto de reporte local de la versión candidata.

---

## 8. Gaps Remanentes y Preparación para Fase 8

1. **Desacoplamiento de la variable `regions`**:
   - El modelo PyFunc y `prepare_features()` requieren estrictamente la columna cruda `regions`.
   - El contrato del backend actual no provee `regions`.
   - Este gap se encuentra delimitado y controlado para ser resuelto en la capa de serving / mediador en las fases subsiguientes, sin comprometer la reproducibilidad del pipeline de datos ni el artefacto del modelo.
2. **Promoción a Champion**:
   - La versión `1` registrada en `salary-predictor` cumple todos los requisitos técnicos, de elegibilidad y de paridad.
   - La asignación del alias `champion` y la integración con `model_provider/` se llevará a cabo en la Fase 8 según las reglas de gobernanza del proyecto.

---

```text
PHASE_7_STATUS=READY_FOR_PHASE_8
```
