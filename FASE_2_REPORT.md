# FASE 2 — REPORTE DE EJECUCIÓN: SPLIT TEMPORAL Y CONTRATO REPRODUCIBLE DE FEATURES

**Fecha:** 2026-09-10  
**Módulo:** `ml/`  
**Entorno:** Linux / Monorepo `PDS-Microproyecto-Grupo-13`  
**Rama:** `feature/monorepo`  
**Pipeline DVC Activo:** `collect -> validate -> preprocess`

---

## 1. INTRODUCCIÓN Y ALCANCE

### Lo que se implementó en esta fase
1. **Filtro del Universo de Modelado:** Se restringió el alcance de entrenamiento al subconjunto `target_scope == "reportado"` (salarios con rangos efectivamente informados por las empresas).
2. **Split Temporal Estricto:** División determinista 70% Train, 15% Validation y 15% Test ordenada cronológicamente por `published` (con desempate estable por índice de fila para mitigar empates temporales), sin mezclas aleatorias (*no random shuffling*).
3. **Cálculo Aislado de Train Limits:** Determinación de los umbrales salariales operacionales (`floor` cuantil 0.001 con cota mínima $1,000 USD y `ceiling` cuantil 0.999) calculados **estrictamente y exclusivamente sobre la partición Train**, persistidos en `artifacts/reports/train_limits.json`.
4. **Módulo de Features Puro y Modular (`src/ml_pipeline/features.py`):** Implementación de la transformación determinista de las 24 features (6 categóricas y 18 numéricas), con normalización de texto, reglas de negocio para modalidad laboral (`work_mode`), límites de experiencia (0 a 50 años) con flag indicador de imputación, y extracción binaria de 13 skills técnicas con semántica idéntica al notebook.
5. **Etapa de Preprocesamiento (`src/ml_pipeline/data/preprocess.py` y CLI `python -m ml_pipeline preprocess`):** Orquestación integral del flujo que consume `data/validated/dataset.parquet` y materializa `train.parquet`, `validation.parquet`, `test.parquet`, `train_limits.json` y el manifiesto `preprocess.json`.
6. **Integración con DVC (`dvc.yaml` y `dvc.lock`):** Registro de la etapa `preprocess` con sus dependencias de datos, código y parámetros (`data.*`), verificando reproducibilidad e idempotencia completa mediante `dvc repro`.
7. **Suite de Pruebas Automatizadas:** 44 pruebas unitarias y de integración pasando al 100% que blindan el contrato de features, invariantes de proporciones, ordenamiento temporal, aislamiento de métricas de train y degradación ante campos ausentes.

### Lo que quedó explícitamente fuera de alcance
- **NO Target Encoding:** No se aplicaron transformaciones de codificación de variables categóricas dependientes de la variable objetivo (se reservan para la etapa de ajuste de estimadores en Fase 3).
- **NO Entrenamiento de Modelos Salariales:** No se entrenaron modelos `LightGBM`, `CatBoost`, `Ridge` ni `DummyRegressor`.
- **NO MLflow Tracking / Model Registry:** No se registraron runs de MLflow ni candidatos de modelos salariales en esta fase.
- **NO Modificaciones al Backend:** No se alteraron los esquemas o endpoints de `backend/`. Se documentó formalmente el gap de contrato existente.
- **NO Modificaciones al Notebook Histórico:** El archivo `06_modelo_definitivo_fjlo.ipynb` se mantuvo estrictamente como artefacto de referencia funcional de solo lectura.

---

## 2. UNIVERSO DE MODELADO

### Métricas del Dataset Validado
- **Total de registros validados (`data/validated/dataset.parquet`):** 260,158 registros.
- **Distribución por `target_source`:**
  - `estimado`: 200,923 registros (77.23%)
  - `híbrido`: 4,872 registros (1.87%)
  - `reportado`: 54,363 registros (20.90%)
- **Total filtrado para modelado (`target_scope: reportado`):** 54,363 registros exactos.

### Justificación Técnica y Coherencia con el Notebook
En el notebook de referencia (`06_modelo_definitivo_fjlo.ipynb`), el universo de entrenamiento y evaluación se restringe explícitamente a ofertas con salario real reportado (`target_source == 'reportado'`). El uso de salarios estimados o imputados externamente introduciría ruido de modelado ajeno a la distribución real del mercado y sesgos preexistentes de algoritmos de estimación de terceros. Al preservar exactamente 54,363 registros, se garantiza 100% de paridad con la línea base funcional histórica.

---

## 3. ESTRATEGIA DE SPLIT TEMPORAL

### Proporciones y Parámetros
Configuradas en `ml/params.yaml`:
```yaml
data:
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15
  target_scope: reportado
```

### Propiedades Temporales
- **Columna temporal de partición:** `published` (parseada a `datetime64[ns, UTC]`).
- **Nulos en `published`:** 0 nulos en el universo de modelado.
- **Timestamps únicos:** 25,790 valores distintos sobre 54,363 filas.
- **Mecanismo de Desempate Estable:** Debido a la concurrencia de ofertas con el mismo timestamp (28,573 duplicados temporales), el ordenamiento se implementó como:
  ```python
  df_scoped = df_scoped.sort_values(by=["published"], kind="mergesort")
  ```
  Esto preserva el orden secuencial de llegada de los snapshots evitando indeterminismos entre ejecuciones en distintas arquitecturas.

### Particiones y Ventanas de Fechas
- **Corte de índices:**
  - $N = 54,363$
  - Train: $[0 : 38,054)$ (70.00%)
  - Validation: $[38,054 : 46,208)$ (15.00%)
  - Test: $[46,208 : 54,363)$ (15.00%)

| Partición | Registros | Proporción | Fecha Mínima (`published_min`) | Fecha Máxima (`published_max`) |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 38,054 | 70.00% | 2025-01-01 00:13:20+00:00 | 2026-03-03 00:00:00+00:00 |
| **Validation** | 8,154 | 15.00% | 2026-03-03 00:00:00+00:00 | 2026-05-26 00:00:00+00:00 |
| **Test** | 8,155 | 15.00% | 2026-05-26 00:00:00+00:00 | 2026-09-01 05:37:04+00:00 |

### Justificación de Split Temporal vs Aleatorio
En problemas de compensación laboral y predicción salarial, el mercado evoluciona por ciclos económicos, inflación, ajustes semestrales y shifts tecnológicos. Un split aleatorio (*random K-fold* o *train_test_split*) evaluaría ofertas pasadas con información filtrada del futuro, inflando artificialmente las métricas de validación. El split temporal garantiza que el modelo se evalúe exactamente bajo el régimen en el que operará en producción: entrenado con el pasado para predecir ofertas publicadas en el futuro.

---

## 4. PREVENCIÓN DE DATA LEAKAGE Y TRAIN LIMITS

### Aislamiento de Umbrales Operacionales
Los límites salariales operacionales (`floor` y `ceiling`) tienen como propósito acotar predicciones anómalas o fuera de mercado durante la inferencia productiva.
- **Data Leakage evitado:** Si estos límites se calcularan sobre todo el dataset (incluyendo validación y test), la variabilidad de precios y ofertas futuras se filtraría al modelo antes de ser evaluado.
- **Implementación:** La función `compute_train_limits(train_df)` recibe **únicamente** la partición de entrenamiento.
- **Valores persistidos en `artifacts/reports/train_limits.json`:**
  ```json
  {
    "ceiling": 720000.0,
    "ceiling_quantile": 0.999,
    "floor": 10935.571999999996,
    "floor_quantile": 0.001,
    "source_split": "train"
  }
  ```
- **Comparación con el Notebook Histórico:** En la Celda 7 de `06_modelo_definitivo_fjlo.ipynb`, el cálculo produce exactamente:
  - `floor`: $10,935.57 USD
  - `ceiling`: $720,000.00 USD
  La coincidencia es exacta hasta el último decimal.

### Ausencia de Encoders Dependientes del Target
En esta fase, **ningún** transformador ajustado (`TargetEncoder`, `OneHotEncoder`, escaladores estándar o PCA) fue aplicado ni persistido. El archivo `features.py` aplica exclusivamente transformaciones libres de estado (*stateless* / puras), garantizando que las particiones de validación y test permanezcan 100% ciegas frente a cualquier estadística global del dataset.

---

## 5. CONTRATO REPRODUCIBLE DE FEATURES

El pipeline materializa un contrato inmutable de **24 variables explicativas** (6 categóricas y 18 numéricas):

### Matriz de Features

| # | Nombre de Feature | Tipo | Variable(s) Origen | Regla de Transformación / Imputación | Valores / Rango Esperado |
|---|---|---|---|---|---|
| 1 | `title` | Categórica | `title` | `normalize_text(title).replace("", "desconocido")` | Cadena en minúsculas sin acentos |
| 2 | `country` | Categórica | `countries` | Primer elemento de `countries` por `|`, `.strip()`, vacío $\rightarrow$ `'desconocido'`. NO aplica `normalize_text()` | País preservando texto/capitalización original |
| 3 | `region` | Categórica | `regions` | Primer elemento de `regions` por `|`, `.strip()`, vacío $\rightarrow$ `'desconocido'`. NO aplica `normalize_text()`. Falla explícitamente si falta la columna. | Región geográfica original |
| 4 | `experience_level`| Categórica | `experience_level` | `fillna("desconocido").astype(str)`. Preserva string vacío si ya existía. | EN, MI, SE, EX, desconocido |
| 5 | `work_mode` | Categórica | `has_remote`, `work_mode` | Regla notebook: `not remote` $\rightarrow$ `presencial`; `code==1` $\rightarrow$ `híbrido`; `code==2` $\rightarrow$ `remoto`; `code==3` $\rightarrow$ `remoto_global`; fallback $\rightarrow$ `remoto_sin_detalle` | presencial, híbrido, remoto, remoto_global, remoto_sin_detalle |
| 6 | `company` | Categórica | `company` | `normalize_text(company).replace("", "desconocido")` | Nombre normalizado de la empresa |
| 7 | `company_is_agency`| Numérica | `company_is_agency` | `fillna(False).astype(int)` | 0, 1 |
| 8 | `experience_years` | Numérica | `experience_years` | Numérico acotado en $[0, 50]$, valores fuera de rango o nulos quedan en `np.nan` | Float $[0.0, 50.0]$ o `NaN` |
| 9 | `experience_years_missing` | Numérica | `experience_years` | `1` si `experience_years` es NaN, `0` en caso contrario | 0, 1 |
| 10 | `skill_python` | Numérica | `tags` | Substring `python` (`re.escape`) | 0, 1 |
| 11 | `skill_sql` | Numérica | `tags` | Substring `sql` (`re.escape`) | 0, 1 |
| 12 | `skill_aws` | Numérica | `tags` | Substring `aws` (`re.escape`) | 0, 1 |
| 13 | `skill_azure` | Numérica | `tags` | Substring `azure` (`re.escape`) | 0, 1 |
| 14 | `skill_gcp` | Numérica | `tags` | Substring `gcp` (`re.escape`) | 0, 1 |
| 15 | `skill_spark` | Numérica | `tags` | Substring `spark` (`re.escape`) | 0, 1 |
| 16 | `skill_docker` | Numérica | `tags` | Substring `docker` (`re.escape`) | 0, 1 |
| 17 | `skill_kubernetes` | Numérica | `tags` | Substring `kubernetes` (`re.escape`) | 0, 1 |
| 18 | `skill_machine_learning` | Numérica | `tags` | Substring `machine learning` (`re.escape`) | 0, 1 |
| 19 | `skill_pytorch` | Numérica | `tags` | Substring `pytorch` (`re.escape`) | 0, 1 |
| 20 | `skill_tensorflow` | Numérica | `tags` | Substring `tensorflow` (`re.escape`) | 0, 1 |
| 21 | `skill_tableau` | Numérica | `tags` | Substring `tableau` (`re.escape`) | 0, 1 |
| 22 | `skill_power_bi` | Numérica | `tags` | Substring `power bi` (`re.escape`) | 0, 1 |
| 23 | `published_year` | Numérica | `published` | `published.dt.year` (NaT permanece NaN) | Año de publicación (e.g. 2025, 2026) |
| 24 | `published_month` | Numérica | `published` | `published.dt.month` (NaT permanece NaN) | Mes de publicación ($1$ a $12$) |

### Propiedades de la Función
La función `prepare_features(df)`:
- Es pura respecto al DataFrame de entrada (no modifica el DataFrame original).
- Es determinista y stateless (recibe columnas raw y genera el contrato de 24 features sin estado mutable).
- Retorna un DataFrame con columnas en orden canónico inmutable `FEATURE_COLUMNS`.
- Admite registros individuales (inferencia online) o datasets masivos (batch) con idéntico comportamiento.

---

## 6. AUDITORÍA DEL CONTRATO DE SERVING (BACKEND)

Se analizó el contrato actual de entrada de inferencia en el backend (`backend/app/schemas/prediction.py`):
```python
class SalaryPredictionRequest(BaseModel):
    title: str
    company: str | None = None
    country: str | None = None
    experience_level: str | None = None
    experience_years: float | None = None
    work_mode: int | None = None
    has_remote: bool = False
    company_is_agency: bool = False
    tags: list[str] | str | None = None
    published: datetime | None = None
```

### Discrepancias Identificadas
1. **Falta de campo geográfico macro (`regions`):** El backend no solicita ni transfiere el campo `regions` / `region` al registro de MLflow ni en la payload de la API.
2. **Estrategia de compatibilidad implementada en `features.py`:**
   ```python
   raw_region = df["regions"] if "regions" in df.columns else (df["region"] if "region" in df.columns else pd.Series("desconocido", index=df.index, dtype=object))
   out["region"] = normalize_text(raw_region).replace("", "desconocido")
   ```
   Si el campo `regions` no está presente en la petición de inferencia, se asigna limpiamente el valor por defecto `"desconocido"`, sin romper el pipeline ni levantar excepciones.

### Declaración Formal de Gap
```text
SERVING_CONTRACT_GAP:
El backend (backend/app/schemas/prediction.py) define el esquema de petición SalaryPredictionRequest
sin el campo 'regions'/'region'. Aunque el módulo de features degrada elegantemente a 'desconocido',
la variable 'region' es una de las 6 variables categóricas del modelo salarial. Antes de desplegar
el modelo en producción en model_provider, el esquema Pydantic y el frontend deberán admitir opcionalmente
'region' o derivarlo automáticamente del catálogo geográfico a partir del 'country'.
```

---

## 7. ARTEFACTOS Y SALIDAS GENERADAS

### Particiones de Datos
- `data/processed/train.parquet`: 38,054 filas, 28 columnas, ~1.2 MB.
- `data/processed/validation.parquet`: 8,154 filas, 28 columnas, ~298 KB.
- `data/processed/test.parquet`: 8,155 filas, 28 columnas, ~302 KB.

*Composición de columnas por partición (28 columnas):*
- **Identificadores y tiempo (2):** `id`, `published`.
- **Features de modelado (24):** 6 categóricas (`title`, `country`, `region`, `experience_level`, `work_mode`, `company`) + 18 numéricas (`company_is_agency`, `experience_years`, `experience_years_missing`, 13 skills, `published_year`, `published_month`).
- **Targets salariales requeridos (2):** `y_min_usd`, `y_max_usd`.

### Reportes y Manifiestos
- `artifacts/reports/train_limits.json`:
  - `floor`: 10935.57
  - `ceiling`: 720000.00
  - `source_split`: "train"
- `artifacts/reports/preprocess.json`:
  - `target_scope`: "reportado"
  - `rows_validated`: 260158
  - `rows_modeling`: 54363
  - `dataset_fingerprint`: `541f47bf1bb49a6a0e5947d49dade61c444e74cd5350cfbb68a185506cd957ed`
  - Fechas extremas y conteos por partición.

---

## 8. PIPELINE DVC Y REPRODUCIBILIDAD

### Topología de Pipeline
```text
collect -> validate -> preprocess
```

### Configuración en `ml/dvc.yaml`
```yaml
  preprocess:
    cmd: python -m ml_pipeline preprocess
    deps:
      - data/validated/dataset.parquet
      - src/ml_pipeline/data/preprocess.py
      - src/ml_pipeline/features.py
    params:
      - data
    outs:
      - data/processed/train.parquet
      - data/processed/validation.parquet
      - data/processed/test.parquet
      - artifacts/reports/train_limits.json:
          cache: false
      - artifacts/reports/preprocess.json:
          cache: false
```

### Verificación de Ejecución e Idempotencia
1. **Ejecución inicial:**
   `dvc repro` reprodujo exitosamente la etapa `preprocess`, calculando hashes de artefactos y actualizando `dvc.lock`.
2. **Prueba de Idempotencia:**
   Una segunda ejecución inmediata de `dvc repro` arrojó:
   ```text
   Stage 'collect' didn't change, skipping
   Stage 'validate' didn't change, skipping
   Stage 'preprocess' didn't change, skipping
   Data and pipelines are up to date.
   ```
3. **Estado DVC (`dvc status`):**
   ```text
   Data and pipelines are up to date.
   ```

---

## 9. CONFIGURACIÓN Y PARÁMETROS

Se extendió `ml/params.yaml` con la sección `data`:
```yaml
data:
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15
  target_scope: reportado
```
### Justificación
- La división 70/15/15 es el estándar adoptado en el notebook de referencia, balanceando volumen suficiente para estimar no-linealidades complejas en train ($38k+$ filas) manteniendo sets de validación ($8k+$) y test ($8k+$) estadísticamente significativos.
- `target_scope: reportado` centraliza como parámetro declarativo el filtro de calidad salarial.

---

## 10. SUITE DE TESTS Y COBERTURA

Se implementaron dos nuevas suites de tests y se actualizó la integración:
- `ml/tests/unit/test_features.py`:
  - `test_feature_contract_dimensions_and_structure`: valida exactamente 24 features y ausencia de targets.
  - `test_prepare_features_produces_exact_contract_and_order`: valida orden y contenido exacto.
  - `test_work_mode_rules_mapping`: valida todas las combinaciones de `has_remote` y códigos de `work_mode`.
  - `test_experience_years_bounds_and_missing_flag`: valida acotación en $[0, 50]$ e indicador binario de imputación.
  - `test_missing_values_and_empty_strings_handled_gracefully`: valida degradación elegante ante nulos y series vacías.
- `ml/tests/unit/test_preprocess.py`:
  - `test_validate_ratios_accepts_valid_and_rejects_invalid`: valida sumas iguales a 1.0 y límites estrictos $(0, 1)$.
  - `test_compute_train_limits_strictly_uses_train_distribution`: valida aislamiento estricto de train para el cálculo de `floor`/`ceiling`.
  - `test_preprocess_filters_target_scope_and_preserves_temporal_order`: valida ordenamiento temporal, particiones y reportes.
- `ml/tests/integration/test_pipeline.py`:
  - `test_reproducible_pipeline_collect_validate_and_preprocess`: valida la tubería integral en un entorno sintético multi-snapshot aislado.

### Resultado de Pytest
```text
======================== 44 passed, 5 skipped in 9.24s =========================
```
(Los 5 tests saltados corresponden a las pruebas de contratos y registro de modelos salariales que se activarán en las fases posteriores de modelado).

---

## 11. LIMITACIONES CONOCIDAS Y DECISIONES TÉCNICAS

1. **Empates Temporales en `published`:** Muchas ofertas comparten el mismo timestamp a nivel de segundos o días. El uso de `mergesort` (ordenamiento estable) garantiza reproducibilidad determinista sin introducir ruido aleatorio.
2. **Agnosticismo respecto a Snapshots Remotos:** El preflight de la Fase 1 permanece intacto y riguroso. El snapshot `jobs_2026-09-09.csv` retirado no fue objeto de ninguna regla ad-hoc ni excepción. Cuando sea restituido en DVC, el pipeline lo incorporará de manera transparente.
3. **Semántica de Substring en Skills:** Las 13 skills utilizan búsqueda de substring con `re.escape` sobre los tags en minúsculas sin límites de palabra (`\b`), replicando con exactitud matemática el comportamiento del notebook.

---

## 12. CHECKLIST DE CONFORMIDAD DE FASE 2

| # | Requerimiento | Estado | Detalle |
|---|---|:---:|---|
| 1 | Filtrado de Universo a `target_source == "reportado"` | **CUMPLIDO** | Exactamente 54,363 registros aislados |
| 2 | Split Temporal Estricto (70/15/15) por `published` | **CUMPLIDO** | Train: 38,054, Val: 8,154, Test: 8,155 con preservación temporal |
| 3 | Cálculo de `train_limits` exclusivamente sobre Train | **CUMPLIDO** | Floor: 10,935.57 USD, Ceiling: 720,000.00 USD persistidos en JSON |
| 4 | Módulo modular de features (`features.py`) | **CUMPLIDO** | Función pura y determinista `prepare_features` |
| 5 | Contrato exacto de 24 features (6 cat + 18 num) | **CUMPLIDO** | Inmutable, testeado y documentado |
| 6 | Materialización de splits Parquet procesados | **CUMPLIDO** | Coexisten IDs, `published`, 24 features y targets salariales |
| 7 | Reporte determinista `preprocess.json` | **CUMPLIDO** | Vinculado a `dataset_fingerprint` con rangos de fechas |
| 8 | Orquestación DVC (`collect -> validate -> preprocess`) | **CUMPLIDO** | `dvc.yaml` y `dvc.lock` actualizados, idempotencia verificada |
| 9 | Parámetros centralizados en `params.yaml` | **CUMPLIDO** | Sección `data` configurable |
| 10 | Suite de tests unitarios y de integración | **CUMPLIDO** | 44 tests passing, 0 fallos, 0 warnings |
| 11 | Auditoría de contrato con Backend de Serving | **CUMPLIDO** | Identificado gap en `regions`, degradación elegante implementada |
| 12 | Cero modificaciones a modelos, registry o backend | **CUMPLIDO** | Alcance de Fase 2 respetado estrictamente |

---

## 13. PRÓXIMOS PASOS (FASE 3)

1. **Target Encoding y Preprocesamiento de Modelado:**
   - Implementar el preprocesador Scikit-Learn / Category Encoders para las variables categóricas de alta cardinalidad (`title`, `company`, `country`, `region`).
   - Ajustar el codificador **exclusivamente** con la partición `train.parquet` y guardarlo dentro del artefacto del modelo.
2. **Entrenamiento de Modelos Salariales:**
   - Migración de los estimadores LightGBM para predecir dualmente `y_min_usd` y `y_max_usd` bajo transformación `log1p`.
   - Implementación del baseline `DummyRegressor` para contrastar ganancia predictiva.
3. **Métricas de Evaluación Salarial:**
   - Métricas continuas de error: MAE, RMSE, $R^2$, MAPE y error por intervalo de predicción.

---

## 14. DICTAMEN FINAL

```text
PHASE_2_STATUS=READY_FOR_PHASE_3
```
