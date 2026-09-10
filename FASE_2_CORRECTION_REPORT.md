# FASE 2 — REPORTE DE CORRECCIÓN: PARIDAD EXACTA DE FEATURES CON EL NOTEBOOK

**Fecha:** 2026-09-10  
**Módulo:** `ml/`  
**Referencia Funcional:** `ml/notebooks/06_modelo_definitivo_fjlo.ipynb`  
**Estado:** Corrección completada — Bloqueado para decisión de ordenamiento de split

---

## 1. ARCHIVOS MODIFICADOS

1. [`ml/src/ml_pipeline/features.py`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml/src/ml_pipeline/features.py)
2. [`ml/tests/unit/test_features.py`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml/tests/unit/test_features.py)
3. [`ml/tests/integration/test_pipeline.py`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml/tests/integration/test_pipeline.py)
4. [`ml/dvc.lock`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml/dvc.lock)
5. [`ml/artifacts/reports/preprocess.json`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml/artifacts/reports/preprocess.json)
6. [`ml/artifacts/reports/train_limits.json`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml/artifacts/reports/train_limits.json)
7. [`FASE_2_REPORT.md`](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/FASE_2_REPORT.md)

---

## 2. CORRECCIONES REALIZADAS EN `prepare_features`

Se revisó minuciosamente la celda de definición en el notebook `06_modelo_definitivo_fjlo.ipynb` (líneas 552-572) y se aplicaron las siguientes reglas exactas en `features.py`:

- **`country`:**
  - Consume exclusivamente la columna `countries`.
  - Toma el primer elemento separado por `|` (`str.split('|').str[0].str.strip()`).
  - Asigna `'desconocido'` en caso de valor nulo o cadena vacía tras strip.
  - **Se eliminó** la llamada a `normalize_text()`, preservando mayúsculas y acentos originales (e.g. `"México|USA"` $\rightarrow$ `"México"`, no `"mexico"`).

- **`region`:**
  - Consume exclusivamente la columna `regions`.
  - Misma lógica que `country` (primer elemento de `|`, strip, vacío $\rightarrow$ `'desconocido'`).
  - **Se eliminó** el fallback tolerante `regions -> region -> desconocido`.
  - **Fallo explícito:** Si la columna `regions` no existe en el DataFrame, se lanza inmediatamente un `KeyError("Required column 'regions' is missing from DataFrame")`.

- **`experience_level`:**
  - Implementación exacta del notebook: `df["experience_level"].fillna("desconocido").astype(str)`.
  - **Se eliminó** `.replace("", "desconocido")`. Las cadenas vacías preexistentes permanecen como cadenas vacías `""`.

- **`title` y `company`:**
  - Se mantiene la normalización de texto estándar `normalize_text().replace("", "desconocido")`, idéntica a la función `norm()` del notebook.

- **`company_is_agency`:**
  - Implementación exacta del notebook: `df["company_is_agency"].fillna(False).astype(int)`.

- **`skills` (13 skills):**
  - Implementación exacta de la semántica del notebook:
    ```python
    tags = df["tags"].fillna("").astype(str).str.lower()
    for name, token in SKILLS.items():
        out[f"skill_{name}"] = tags.str.contains(re.escape(token), regex=True).astype(int)
    ```
  - Se descartó cualquier uso de word boundaries (`\b`), regex flexibles (`machine\s*learning`), tokenizaciones o fuzzy matching.
  - Substrings válidos (e.g. `"pythonista"`) activan `skill_python` tal como ocurre en el notebook.
  - Variantes con guión (`"machine-learning"`) o sin espacio (`"powerbi"`) no activan el token, preservando paridad estricta.

- **`published`:**
  - Implementación exacta: `pub.dt.year` y `pub.dt.month`.
  - **Se eliminó** `.fillna(0).astype(int)`. Los valores nulos o `NaT` producen `NaN` (`float64`), no `0`.

- **Corrección en Documentación:**
  - Se corrigieron todas las menciones erróneas de "12 skills" a "13 skills".
  - Se eliminó la afirmación de que `prepare_features` es idempotente ($f(f(x)) = f(x)$), reemplazándola por la descripción precisa: función **pura** respecto al DataFrame de entrada, **determinista** y **stateless**.

---

## 3. SUITE DE TESTS Y RESULTADO PYTEST

Se incorporaron pruebas específicas en `tests/unit/test_features.py`:
- `test_parity_exact_rules`: Comprueba las 13 skills, 24 features (6 categóricas + 18 numéricas), preservación de capitalización/acentos en `country` (`"México"`), preservación de cadena vacía en `experience_level`, substring matching en skills (`"pythonista"`, `"machine learning"` vs `"machine-learning"`), y NaT produciendo NaN en fechas.
- `test_missing_regions_column_raises_key_error`: Comprueba que la ausencia de la columna `regions` lanza `KeyError`.
- Actualización de `tests/integration/test_pipeline.py` para incluir `regions` en los datos sintéticos.

### Resultado de Pytest
```text
=================== 46 passed, 5 skipped in 8.65s ===================
```
- **46 pasados**, **0 fallidos**, **0 errores**.
- 5 tests omitidos intencionalmente correspondientes a fases futuras de modelado salarial.

---

## 4. CONTRATO FINAL DE 24 FEATURES

El contrato reproducido mantiene exactamente **24 features** en orden canónico inmutable:

### 6 Variables Categóricas
1. `title`
2. `country`
3. `region`
4. `experience_level`
5. `work_mode`
6. `company`

### 18 Variables Numéricas
7. `company_is_agency`
8. `experience_years`
9. `experience_years_missing`
10. `skill_python`
11. `skill_sql`
12. `skill_aws`
13. `skill_azure`
14. `skill_gcp`
15. `skill_spark`
16. `skill_docker`
17. `skill_kubernetes`
18. `skill_machine_learning`
19. `skill_pytorch`
20. `skill_tensorflow`
21. `skill_tableau`
22. `skill_power_bi`
23. `published_year`
24. `published_month`

---

## 5. CONFIRMACIÓN DE 13 SKILLS

Se confirma la existencia y orden exacto de las **13 skills técnicas**:
1. `python` (token: `'python'`)
2. `sql` (token: `'sql'`)
3. `aws` (token: `'aws'`)
4. `azure` (token: `'azure'`)
5. `gcp` (token: `'gcp'`)
6. `spark` (token: `'spark'`)
7. `docker` (token: `'docker'`)
8. `kubernetes` (token: `'kubernetes'`)
9. `machine_learning` (token: `'machine learning'`)
10. `pytorch` (token: `'pytorch'`)
11. `tensorflow` (token: `'tensorflow'`)
12. `tableau` (token: `'tableau'`)
13. `power_bi` (token: `'power bi'`)

---

## 6. CONTEOS Y TRAIN LIMITS

Tras la ejecución de `python -m ml_pipeline preprocess` y `dvc repro`:
- **Universo validado:** 260,158 filas
- **Universo de modelado (`target_scope: reportado`):** 54,363 filas
- **Partición Train (70%):** 38,054 filas
- **Partición Validation (15%):** 8,154 filas
- **Partición Test (15%):** 8,155 filas
- **Train Limits (`artifacts/reports/train_limits.json`):**
  - `floor`: **10,935.571999999996** USD
  - `ceiling`: **720,000.0** USD
  - `source_split`: `"train"`

Los conteos y límites se conservan idénticos a los valores de referencia.

---

## 7. DIAGNÓSTICO DE MEMBRESÍA DE SPLITS TEMPORALES

Se realizó el diagnóstico comparativo sobre `data/validated/dataset.parquet` con el universo `target_source == 'reportado'` ($N = 54,363$):

- **Método A (Notebook):**
  `df.sort_values("published", na_position="first")`
  *(utiliza el algoritmo por defecto de pandas: quicksort, el cual es inestable ante claves repetidas)*.

- **Método B (Pipeline Actual):**
  `df.sort_values("published", na_position="first", kind="mergesort")`
  *(utiliza ordenamiento estable, preservando el orden relativo de inserción original de los registros ante claves repetidas)*.

### Resultados de la Comparación

| Métrica | Valor |
| :--- | :--- |
| **Total de filas en universo de modelado** | 54,363 |
| **IDs diferentes en Train** | **26** |
| **IDs diferentes en Validation** | **30** |
| **IDs diferentes en Test** | **4** |
| **Total de filas que cambian de partición** | **30** |
| **Train Limits idénticos entre Método A y Método B** | **Sí** (`floor`: 10935.57, `ceiling`: 720000.0) |

### Análisis de la Causa Raíz
Existen **25,790** timestamps únicos para **54,363** filas, resultando en **28,573 duplicados temporales**.
En las fronteras de corte exactas:
1. **Frontera Train / Validation (índice 38,054):** Ocurre en `2026-03-03 00:00:00+00:00`, donde existen **172 ofertas publicadas exactamente en ese instante**. `quicksort` y `mergesort` ordenan de modo distinto ese bloque de 172 ofertas empatadas, provocando que 26 registros crucen la frontera entre Train y Validation.
2. **Frontera Validation / Test (índice 46,208):** Ocurre en `2026-05-26 00:00:00+00:00`, donde existen **43 ofertas publicadas exactamente en ese instante**. La diferencia de desempate provoca que 4 registros crucen la frontera entre Validation y Test.

---

## 8. DIVERGENCIAS RESTANTES Y DECISIÓN PENDIENTE

1. **Divergencia de Política de Ordenamiento Temporal:**
   - Si se adopta **Método A (`quicksort`)**, se iguala la partición exacta que obtuvo el notebook en esa sesión específica de Python/pandas, pero se asume un ordenamiento dependiente de la implementación interna de quicksort (que no garantiza estabilidad ante empates).
   - Si se mantiene **Método B (`mergesort`)**, se garantiza reproducibilidad estricta y determinista independiente de la plataforma, pero 30 filas difieren respecto a la asignación de partición del notebook histórico.
   - Conforme a la regla de la fase: *no se cambió silenciosamente la política* y se mantiene `mergesort` a la espera de la decisión del equipo.

2. **Gap en Contrato de Serving:**
   - El backend (`backend/app/schemas/prediction.py`) no incluye el campo `regions`. Dado que `prepare_features` ahora exige la columna `regions` sin fallback, el backend requerirá actualización antes de habilitar inferencia con este modelo.

---

## 9. DICTAMEN FINAL

Dado que los dos métodos de ordenamiento asignan IDs diferentes entre splits (30 filas afectadas):

```text
PHASE_2_CORRECTION_STATUS=BLOCKED_FOR_SPLIT_DECISION
```
