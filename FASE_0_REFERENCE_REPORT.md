# FASE_0_REFERENCE_REPORT: Alineación y Definición de la Fuente de Datos Foorilla

## A. Fuente Foorilla Canónica Detectada

- **Ruta Canónica Identificada**: `ml/data/raw/foorilla`
- **Ruta Absoluta**: `/home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/ml/data/raw/foorilla`
- **Variable Canónica**: `FOORILLA_SOURCE_DIR = ml/data/raw/foorilla`
- **Rol Arquitectónico**: Fuente RAW inmutable. Los archivos y snapshots en este directorio operan como datos fuente de solo lectura y bajo ningún concepto serán alterados ni sobreescritos por etapas del pipeline (`collect`, `validate`, `preprocess`, etc.).

---

## B. Inventario Actual de `jobs_*.csv`

Inspección realizada directamente sobre el checkout real del sistema de archivos y el catálogo DVC:

| Archivo | Fecha Inferida | Tamaño (Bytes) | Tamaño Formateado | Filas (con header) | Materializado Localmente | Archivo `.dvc` Asociado | MD5 Hash (DVC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `jobs_2026-08-16.csv` | 2026-08-16 | 395,375,574 | ~377.1 MB | 326,381 | **SÍ** | `jobs_2026-08-16.csv.dvc` | `0332f61b942bded5029a2336be39e314` |
| `jobs_2026-08-20.csv` | 2026-08-20 | 3,628,504 | ~3.46 MB | 2,804 | **SÍ** | `jobs_2026-08-20.csv.dvc` | `819cd4843a5294275547d84f33b3eb4e` |
| `jobs_2026-08-24.csv` | 2026-08-24 | 1,909,773 | ~1.82 MB | 1,473 | **SÍ** | `jobs_2026-08-24.csv.dvc` | `f86bfad34a84d9142947b58f66c25f7c` |
| `jobs_2026-08-28.csv` | 2026-08-28 | 2,634,831 | ~2.51 MB | 2,010 | **SÍ** | `jobs_2026-08-28.csv.dvc` | `a964add8a906eb4aaba3baf929aa1c87` |
| `jobs_2026-09-01.csv` | 2026-09-01 | 1,536,288 | ~1.46 MB | 1,186 | **SÍ** | `jobs_2026-09-01.csv.dvc` | `37979f94603a6779d8ff962db26a666a` |
| `jobs_2026-09-09.csv` | 2026-09-09 | 6,388,075 | ~6.09 MB | N/A (en cache/remoto) | **NO** | `jobs_2026-09-09.csv.dvc` | `13beb94708d4efeffd7037825a99caaf` |

**Total de snapshots detectados**: 6 snapshots (5 materializados localmente, 1 en `.dvc` pendiente de pull).  
**Volumen total materializado**: 405,084,970 bytes (~386.3 MB), 333,854 registros brutos.

Archivos complementarios en la carpeta:
- Para cada corte existen archivos de metadatos API `jobs_*_manifest.json` y sus correspondientes `.dvc` (`jobs_*_manifest.json.dvc`).
- Un archivo `ml/data/raw/foorilla/.gitignore` que mantiene fuera de git los ficheros CSV e incluye solo los punteros `.dvc`.

---

## C. Estado DVC de los Snapshots

1. **Estructura DVC**:
   - El root DVC se encuentra en `ml/.dvc`.
   - Los snapshots grandes están versionados individualmente mediante punteros `.dvc` (`data/raw/foorilla/jobs_*.csv.dvc`).
2. **Remotos configurados en `ml/.dvc/config`**:
   - `aws-remote`: `s3://amzn-s3-mlops-foorilla`
   - `gdrive-remote`: `gdrive://1CNmcTbMiSGg3tfZ0HXeoK9a6lJsvO8vz` (remoto por defecto).
3. **Estado de Cache**:
   - Los 5 primeros cortes (`2026-08-16` a `2026-09-01`) se encuentran materializados en el disco local y coinciden con sus punteros DVC.
   - El corte `jobs_2026-09-09.csv` se encuentra registrado en Git a través de su archivo `jobs_2026-09-09.csv.dvc`, pero actualmente reporta estado `not in cache` / no materializado en el entorno local.

---

## D. Comportamiento de Ingesta Observado en el Notebook

Del análisis funcional de las celdas 3, 5 y 7 de `ml/notebooks/06_modelo_definitivo_fjlo.ipynb`:

1. **Descubrimiento y Carga**:
   - Búsqueda determinista: `files = sorted(DATA_DIR.glob('jobs_*.csv'))`.
   - Lectura con `pd.read_csv(path, low_memory=False)`.
   - Enriquecimiento de procedencia: cada registro recibe `_priority` (orden numérico del archivo, donde los más recientes tienen mayor índice) y `_source_file` (`path.name`).
   - Concatenación unificada: `raw = pd.concat(parts, ignore_index=True)`.
2. **Deduplicación Etapa 1 (por ID)**:
   - Parseo de fecha: `raw['published'] = pd.to_datetime(raw['published'], errors='coerce', utc=True)`.
   - Completitud: `raw['_complete'] = raw.notna().sum(axis=1)`.
   - Ordenamiento y descarte: `raw.sort_values(['id', '_priority', '_complete', 'published']).drop_duplicates('id', keep='last')`.
3. **Deduplicación Etapa 2 (por Firma / Republicaciones)**:
   - Función de normalización de cadenas `norm(values)` (minúsculas, normalización NFKD, eliminación de acentos, filtrado alfanumérico).
   - URL sin anclas: `urldefrag(apply_url)[0].rstrip('/').lower()`.
   - Firma: `_signature = url + '|' + norm(company) + '|' + norm(title) + '|' + norm(location)`.
   - Información salarial: conteo de campos presentes en `['salary_min', 'salary_max', 'salary_min_usd', 'salary_max_usd']`.
   - Ordenamiento y descarte: `sort_values(['_signature', 'company_is_agency', '_salary_info', '_complete', 'published']).drop_duplicates('_signature', keep='last')`.
4. **Definición y Coherencia de Targets**:
   - Variables objetivo: `y_min_usd` y `y_max_usd` (salario mínimo y máximo anual en USD).
   - Clasificación de procedencia (`target_source`):
     - `'reportado'`: tanto `salary_min` como `salary_max` originales están presentes.
     - `'híbrido'`: al menos uno está presente.
     - `'estimado'`: inferido/estimado externamente por la plataforma.
   - Filtros de higiene:
     - `complete`: `y_min_usd.notna() & y_max_usd.notna()`.
     - `positive`: `y_min_usd > 0 & y_max_usd > 0`.
     - `ordered`: `y_min_usd <= y_max_usd`.
     - `inlier`: filtrado IQR (± 3 IQR) sobre el punto medio en escala logarítmica: `log1p((y_min_usd + y_max_usd) / 2)`.
5. **Partición de Datos**:
   - Selección estricta de alcance: solo registros con `target_source == 'reportado'`.
   - Partición estrictamente temporal ordenada por `published` (sin aleatoriedad ni shuffle): 70% entrenamiento, 15% validación, 15% prueba.
   - Cálculo de límites operativos en entrenamiento: `floor = max(1000, quantile(0.001))` y `ceiling = quantile(0.999)`.
6. **Conteos Históricos de Referencia del Notebook**:
   - Filas integradas: 333,773
   - IDs únicos: 333,422
   - Vacantes deduplicadas: 283,551
   - Muestra modelado (inliers coherentes): 260,158
   - Rangos reportados (dataset para split): 54,363
     - Entrenamiento (70%): 38,054
     - Validación (15%): 8,154
     - Prueba (15%): 8,155
   - Límites operativos aprendidos en entrenamiento: `floor=10,935.57 USD`, `ceiling=720,000.00 USD`.

---

## E. Definición Aprobada de Entrada

**"Todos los archivos que cumplan `jobs_*.csv` dentro de `ml/data/raw/foorilla/` forman automáticamente el dataset de entrada del pipeline."**

- **Sin listas estáticas**: No se mantendrán listas fijas de archivos en `params.yaml`.
- **Evolución continua**: Si en el futuro se añade un nuevo snapshot (ej. `jobs_2026-09-15.csv`), se modifica uno existente o se retira un snapshot, el pipeline lo incorporará automáticamente en la siguiente ejecución.
- **Tolerancia a discrepancias numéricas**: Las cifras del notebook son una referencia histórica del corte con el que fue ejecutado. El nuevo pipeline operará con la totalidad de snapshots materializados disponibles, por lo que la paridad exigida es algorítmica y procedural (reglas de deduplicación, limpieza, filtrado temporal, límites), no de conteos absolutos idénticos.

---

## F. Estrategia DVC Recomendada para Dependencias Dinámicas

### Diagnóstico de Alternativas
1. **Dependencia por archivo individual en `dvc.yaml`**: Inviable y rechazada, ya que obligaría a editar manualmente `dvc.yaml` con cada nuevo snapshot, violando el requerimiento de detección automática.
2. **Dependencia de la carpeta completa `data/raw/foorilla`**:
   - Configurar en la etapa `collect` de `dvc.yaml`:
     ```yaml
     collect:
       cmd: python -m ml_pipeline collect
       deps:
         - data/raw/foorilla
         - src/ml_pipeline/data/collect.py
     ```
   - **Mecanismo**: DVC calcula recursivamente el árbol de estados y hashes de los contenidos del directorio `data/raw/foorilla`. Si se añade, modifica o elimina cualquier snapshot `.csv` (o cualquier `.dvc`), el hash de la dependencia de directorio cambia inmediatamente, invalidando el cache de `collect` y forzando su reejecución en el próximo `dvc repro`.
   - **Eficiencia**: DVC utiliza una base de datos local de estados (`state`) basada en inode/mtime/tamaño, por lo que no recalcula hashes de 400 MB en cada invocación de `dvc status` salvo que el archivo realmente haya cambiado.

### Recomendación para la Fase 1
- Declarar `data/raw/foorilla` como dependencia directa en la etapa `collect`.
- En `collect.py`, implementar el descubrimiento determinista `sorted(Path("data/raw/foorilla").glob("jobs_*.csv"))`.
- Incorporar una verificación explícita en `collect.py` que compruebe si existen archivos `.dvc` cuyos `.csv` correspondientes no estén materializados localmente (como `jobs_2026-09-09.csv`), emitiendo una advertencia clara en logs y registrándolo en el manifiesto.

---

## G. Contrato Propuesto de `data_manifest.json`

Ubicación del artefacto: `artifacts/reports/data_manifest.json`  
Generado por: Etapa `collect`  
Propósito: Trazabilidad inmutable de los snapshots exactos que componen el dataset de cada corrida.

```json
{
  "source": "foorilla",
  "snapshot_count": 5,
  "snapshots": [
    {
      "name": "jobs_2026-08-16.csv",
      "date": "2026-08-16",
      "size_bytes": 395375574,
      "fingerprint": "0332f61b942bded5029a2336be39e314",
      "rows_raw": 326380,
      "dvc_tracked": true
    },
    {
      "name": "jobs_2026-08-20.csv",
      "date": "2026-08-20",
      "size_bytes": 3628504,
      "fingerprint": "819cd4843a5294275547d84f33b3eb4e",
      "rows_raw": 2803,
      "dvc_tracked": true
    },
    {
      "name": "jobs_2026-08-24.csv",
      "date": "2026-08-24",
      "size_bytes": 1909773,
      "fingerprint": "f86bfad34a84d9142947b58f66c25f7c",
      "rows_raw": 1472,
      "dvc_tracked": true
    },
    {
      "name": "jobs_2026-08-28.csv",
      "date": "2026-08-28",
      "size_bytes": 2634831,
      "fingerprint": "a964add8a906eb4aaba3baf929aa1c87",
      "rows_raw": 2009,
      "dvc_tracked": true
    },
    {
      "name": "jobs_2026-09-01.csv",
      "date": "2026-09-01",
      "size_bytes": 1536288,
      "fingerprint": "37979f94603a6779d8ff962db26a666a",
      "rows_raw": 1185,
      "dvc_tracked": true
    }
  ],
  "dataset_fingerprint": "a3f5b8c9d1e2f3... (SHA-256 consolidado de snapshots y orden)",
  "unmaterialized_dvc_snapshots": [
    {
      "name": "jobs_2026-09-09.csv",
      "dvc_md5": "13beb94708d4efeffd7037825a99caaf",
      "size_bytes": 6388075
    }
  ]
}
```

**Principios de Determinismo**:
1. Sin timestamps no deterministas (ej. no incluir `created_at` ni `generated_at`).
2. Lista `snapshots` ordenada estrictamente de forma alfabética por `name`.
3. `dataset_fingerprint` estable: hash SHA-256 calculado sobre la concatenación ordenada de nombres y huellas de los snapshots participantes.
4. Integración con linaje: `collect_lineage()` en `tracking/lineage.py` consumirá directamente `manifest["dataset_fingerprint"]`, reemplazando el sha256 hardcodeado de `data/raw/dataset.csv`.

---

## H. Restos Identificados del Pipeline `breast_cancer`

| Elemento Encontrado | Archivos / Componentes | Clasificación | Acción en la Migración |
| :--- | :--- | :--- | :--- |
| `load_breast_cancer` | `src/ml_pipeline/data/collect.py`, `src/ml_pipeline/data/validate.py` | **Debe eliminarse** | Reemplazar en Fase 1 por ingesta de Foorilla y validación de esquema de vacantes. |
| `data/raw/dataset.csv` | `dvc.yaml`, `collect.py`, `validate.py`, `lineage.py`, `.gitignore` | **Debe eliminarse** | Suplantar por la entrada de carpeta `data/raw/foorilla` y salida `data/interim/foorilla_consolidated.*`. |
| `data/validated/dataset.csv` | `dvc.yaml`, `validate.py`, `preprocess.py` | **Debe adaptarse** | Adaptar al dataset validado de Foorilla (`data/validated/dataset.*`). |
| `data/processed/train.csv`, `test.csv` | `dvc.yaml`, `preprocess.py`, `train.py`, `evaluate.py`, `mlflow_tracker.py` | **Debe adaptarse** | Adaptar a split temporal de 3 particiones: `train.*`, `validation.*`, `test.*` + `train_limits.json`. |
| `stratify=frame["target"]` | `src/ml_pipeline/data/preprocess.py` | **Debe eliminarse** | Eliminar en Fase 2; el split salarial es temporal por fecha de publicación, no estratificado aleatorio. |
| Columna `target` (0/1 binario) | `validate.py`, `preprocess.py`, `train.py`, `evaluate.py`, tests | **Debe eliminarse** | Reemplazar por variables continuas `y_min_usd` y `y_max_usd`. |
| `LogisticRegression`, `RandomForestClassifier` | `src/ml_pipeline/modeling/factory.py`, `params.yaml` | **Mantener temporalmente / Adaptar** | Mantener en Fase 0 y 1; en fases de modelado salarial migrar a regresores (`Ridge`, `LGBMRegressor`, etc.). |
| Métricas `accuracy`, `precision`, `recall`, `f1` | `src/ml_pipeline/modeling/evaluate.py`, `candidate.py`, `params.yaml` | **Mantener temporalmente / Adaptar** | Mantener temporalmente; en fase de evaluación salarial migrar a `mae`, `rmse`, `mape`, `r2`, cobertura. |
| `minimum_score: 0.80` sobre F1 | `params.yaml`, `candidate.py` | **Mantener temporalmente / Adaptar** | Adaptar a umbrales de error salarial o comparación con `DummyRegressor`. |
| Tests de `expected_columns()` de sklearn | `tests/unit/test_validation.py`, `tests/contract/test_model_contract.py` | **Debe adaptarse** | Actualizar progresivamente a partir de la Fase 1 conforme se migren los contratos de datos. |

---

## I. Grafo DVC Objetivo

```
                  ┌──────────────────────────────┐
                  │    data/raw/foorilla/        │ (Inmutable, raw)
                  │    (jobs_*.csv)              │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    collect    │
                         └───────┬───────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────────┐
│ data/interim/foorilla_        │ │ artifacts/reports/                │
│ consolidated.parquet          │ │ data_manifest.json                │
└───────────────┬───────────────┘ └───────────────────────────────────┘
                │
                ▼
        ┌───────────────┐
        │   validate    │
        └───────┬───────┘
                │
                ├─────────────────────────────────┐
                ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────────┐
│ data/validated/dataset.parquet│ │ artifacts/reports/validation.json │
└───────────────┬───────────────┘ └───────────────────────────────────┘
                │
                ▼
        ┌───────────────┐
        │  preprocess   │ (Split temporal 70/15/15)
        └───────┬───────┘
                │
                ├──────────────────────┬──────────────────────┬──────────────────────┐
                ▼                      ▼                      ▼                      ▼
┌─────────────────────────┐ ┌────────────────────┐ ┌────────────────────┐ ┌──────────────────────┐
│ data/processed/train.*  │ │ data/processed/    │ │ data/processed/    │ │ artifacts/reports/   │
│                         │ │ validation.*       │ │ test.*             │ │ train_limits.json    │
└───────────────┬─────────┘ └──────────┬─────────┘ └──────────┬─────────┘ └──────────────────────┘
                │                      │                      │
                └──────────────┬───────┘                      │
                               ▼                              │
                       ┌───────────────┐                      │
                       │    qualify    │                      │
                       └───────┬───────┘                      │
                               │                              │
                ┌──────────────┴───────────────┐              │
                ▼                              ▼              │
┌───────────────────────────────┐ ┌─────────────────────────┐ │
│ artifacts/reports/            │ │ calibration/            │ │
│ qualification.json            │ │ uncertainty artifacts   │ │
└───────────────────────────────┘ └─────────────────────────┘ │
                               │                              │
                               ▼                              │
                       ┌───────────────┐                      │
                       │     train     │                      │
                       └───────┬───────┘                      │
                               │                              │
                               ▼                              │
                ┌───────────────────────────────┐             │
                │ artifacts/work/model/         │             │
                │ model.joblib (o bundle)       │             │
                └──────────────┬────────────────┘             │
                               │                              │
                               └──────────────┬───────────────┘
                                              ▼
                                      ┌───────────────┐
                                      │   evaluate    │
                                      └───────┬───────┘
                                              │
                ┌─────────────────────────────┼─────────────────────────────┐
                ▼                             ▼                             ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐ ┌───────────────────────────────┐
│ artifacts/reports/            │ │ artifacts/reports/            │ │ artifacts/reports/            │
│ metrics.json                  │ │ candidate.json                │ │ experiment_manifest.json      │
└───────────────────────────────┘ └───────────────────────────────┘ └───────────────────────────────┘

========================= FUERA DE DVC (Track & Serve) =========================
               ┌───────────────────────────────┐
               │ ml_pipeline track             │ (MLflow Tracking Run)
               └───────────────┬───────────────┘
                               ▼
               ┌───────────────────────────────┐
               │ ml_pipeline register-candidate│ (MLflow Model Registry)
               └───────────────┬───────────────┘
                               ▼
               ┌───────────────────────────────┐
               │ model_provider promotion      │ (Promoción Alias Champion)
               └───────────────────────────────┘
```

---

## J. Cambios Concretos que Deberá Realizar la Fase 1

La Fase 1 se centrará exclusivamente en la infraestructura de datos (`collect`, `validate`, manifiesto y esquema):

1. **Reimplementación de `collect.py`**:
   - Lectura dinámica de `jobs_*.csv` en `data/raw/foorilla`.
   - Adición de columnas `_priority` y `_source_file`.
   - Concatenación determinista.
   - Deduplicación por `id` (conservando el más completo y reciente).
   - Generación de firma `_signature` (`url|norm(company)|norm(title)|norm(location)`) y deduplicación de republicaciones.
   - Tipificación de salarios y etiquetado `target_source` (`reportado`, `híbrido`, `estimado`).
   - Filtrado de coherencia: completos, positivos, ordenados y recorte IQR sobre `log1p(midpoint)`.
   - Generación del archivo consolidado en `data/interim/foorilla_consolidated.parquet` (o CSV según convención de almacenamiento).
   - Generación del artefacto `artifacts/reports/data_manifest.json` con metadatos deterministas.
2. **Reimplementación de `validate.py`**:
   - Reemplazar la validación de 30 columnas numéricas de sklearn por la validación del esquema tabular de Foorilla (28 columnas de entrada + columnas procesadas).
   - Comprobación de tipos, integridad de claves y presencia de rangos válidos.
   - Emisión del reporte `artifacts/reports/validation.json`.
   - Emisión del dataset validado en `data/validated/dataset.parquet` (o CSV).
3. **Actualización de `dvc.yaml` (etapas iniciales)**:
   - Configurar `collect` con dependencia sobre `data/raw/foorilla` y salidas en `data/interim/` y `artifacts/reports/data_manifest.json`.
   - Conectar `validate` con la salida de `collect`.
4. **Adaptación de Tests de Datos**:
   - Actualizar `tests/unit/test_validation.py` con fixtures basadas en el esquema de Foorilla.
   - Añadir tests unitarios para la lógica de deduplicación y generación de `data_manifest.json`.

---

## K. Riesgos o Bloqueos Detectados

1. **Snapshot pendiente de materialización local (`jobs_2026-09-09.csv`)**:
   - El archivo `jobs_2026-09-09.csv.dvc` está registrado en Git, pero el `.csv` físico no está presente en disco (`not in cache`).
   - **Impacto**: Si `collect` se ejecuta hoy en local, solo procesará los 5 cortes materializados. Si se requiere incluir este corte, será necesario ejecutar `dvc pull` sobre ese archivo (lo cual requiere acceso al bucket S3 o Google Drive configurado).
   - **Mitigación**: `collect.py` detectará automáticamente todos los `.csv` físicamente presentes y registrará en `data_manifest.json` tanto los snapshots procesados como los punteros `.dvc` no materializados. No bloquea el inicio de la Fase 1.
2. **Uso de memoria con el snapshot inicial (`jobs_2026-08-16.csv`)**:
   - El archivo supera los 395 MB (~326k filas).
   - **Impacto**: Aunque cabe perfectamente en memoria RAM (~1-2 GB durante el merge y cálculo de firmas), se debe emplear `low_memory=False` y tipos eficientes en pandas para evitar degradación de rendimiento.
3. **Selección de formato de serialización (`.parquet` vs `.csv`)**:
   - Los datos intermedios en CSV plano de 326k filas ocupan ~400 MB y pierden tipos nativos. Formato Parquet reduce el tamaño a ~30 MB y preserva tipos estrictos. Se recomienda formalizar el uso de Parquet en `data/interim/` y `data/validated/`.
4. **Incompatibilidad temporal de etapas downstream**:
   - Las etapas posteriores (`preprocess`, `train`, `evaluate`) aún esperan el formato del dataset antiguo. Se debe evitar ejecutar `dvc repro` completo hasta que se adapten secuencialmente en las Fases 2, 3 y 4.

**Conclusión de Riesgos**: No existen bloqueos técnicos bloqueantes que impidan proceder con la Fase 1.

---

## L. Condiciones de Aprobación de la Fase 0

- [x] **Fuente Foorilla canónica identificada**: `ml/data/raw/foorilla` establecida como única fuente RAW inmutable.
- [x] **Gestión DVC documentada**: Identificados los archivos `.dvc`, los remotos configurados y el estado del cache local.
- [x] **Regla de ingesta aprobada**: Todos los `jobs_*.csv` presentes y futuros forman automáticamente el dataset de entrada sin listas manuales en `params.yaml`.
- [x] **Estrategia DVC definida**: Dependencia sobre el directorio `data/raw/foorilla` en la etapa `collect`.
- [x] **Contrato de manifiesto establecido**: Especificación determinista de `artifacts/reports/data_manifest.json` sin timestamps inestables.
- [x] **Catálogo de deuda del pipeline de juguete**: Identificados y clasificados todos los restos de `breast_cancer`, `LogisticRegression`, `f1`, etc.
- [x] **Nuevo grafo objetivo formalizado**: Flujo DVC de 6 etapas (`collect` → `validate` → `preprocess` → `qualify` → `train` → `evaluate`) y tracking externo.
- [x] **Cero modificaciones no autorizadas**:
  - No se entrenó ningún modelo.
  - No se modificaron tests.
  - No se crearon runs en MLflow ni se alteró el Model Registry.
  - No se tocaron `model_provider` ni `backend`.
  - El repositorio permanece en un estado limpio, consistente y no parcialmente migrado.

---

PHASE_0_STATUS=READY_FOR_PHASE_1
