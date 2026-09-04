# Baseline LightGBM para rangos salariales

Familia de modelos para el experimento de rangos salariales, junto a
`ml/catboost_baseline`. Reutiliza sin modificar `data.py`, `features.py` y `evaluation.py`de
ese paquete — este paquete solo agrega el código de entrenamiento específico
de LightGBM (`model.py`), su propio punto de entrada de línea de comandos
(`run.py`), y un `evaluation_extras.py`, con
una métrica adicional (de generalización por empresa)


## Preparación

Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\Activate.ps1 en Windows
python -m pip install -r requirements.lock.txt
python -m pip install -r ml/lightgbm_baseline/requirements.txt
```

Recupere los mismos cortes CSV que usa el baseline de CatBoost:

```bash
dvc pull data/raw/foorilla/jobs_2026-08-16.csv.dvc \
  data/raw/foorilla/jobs_2026-08-20.csv.dvc \
  data/raw/foorilla/jobs_2026-08-24.csv.dvc \
  data/raw/foorilla/jobs_2026-08-28.csv.dvc
```

## Prueba rápida

Verifica el flujo del código; sus métricas no son representativas (datos
parciales, pocas rondas de entrenamiento):

```bash
python -m ml.lightgbm_baseline.run \
  --max-rows-per-file 10000 \
  --iterations 20 \
  --no-mlflow \
  --run-name smoke-test
```

## Entrenamiento base

```bash
python -m ml.lightgbm_baseline.run
```

La configuración está en `params.yaml`. Los artefactos locales (ignorados por
Git) incluyen:

- Dos modelos nativos de LightGBM (`.txt`, mediante `Booster.save_model`).
- Métricas de validación, prueba y del baseline de mediana ingenua.
- Predicciones sobre el conjunto de prueba.
- Métricas por procedencia del objetivo, experiencia, país y familia de cargo.
- Métricas de generalización por empresa (vista/no vista), opcional —
  ver la sección siguiente.
- Importancia de variables (basada en ganancia/`gain`).
- Un manifiesto con el commit de git, punteros de DVC, versiones de paquetes
  y la configuración utilizada.

### Resultado de referencia (2 de septiembre de 2026)

Se entrenó sobre 39.862 vacantes con salario reportado, se validó con 7.573 y se
evaluó una sola vez sobre 6.928 observaciones posteriores en el tiempo.

| Variante | Variables | MAE medio validación | MAE medio prueba | Mejora vs. mediana | R²val (mín,máx) | R²prb (mín,máx)
| --- | ---: | ---: | ---: | ---: |---: |---: |
| Con empresa | 54 | USD 27.233 | USD 30.231 | 44,88 % | 0.62, 0.67 | 0.51, 0.57
| Sin empresa | 53 | USD 36.104 | USD 38.432 | 29,92 % | 0.41, 0.45 | 0.32, 0.38
| Mediana de entrenamiento | — | — | USD 54.846 | — |— |— |

**Los valores obtenidos con el modelo catboost base (31/08/26):**

Se entrenó sobre 39.707 vacantes con salario reportado, se validó con 6.931 y se
evaluó una sola vez sobre 7.549 observaciones posteriores en el tiempo.

| Variante | Variables | MAE medio validación | MAE medio prueba | Mejora vs. mediana |
| --- | ---: | ---: | ---: | ---: |
| Con empresa | 54 | USD 28.654 | USD 31.825 | 41,8 % |
| Sin empresa | 53 | USD 36.901 | USD 38.520 | 29,6 % |
| Mediana de entrenamiento | — | — | USD 54.722 | — |

La corrida
con empresa obtuvo `R² = 0,468` para el mínimo y `R² = 0,545` para el máximo; no
debe interpretarse todavía como modelo final.

## Generalización por empresa (vista/no vista)

Adición que separa el conjunto de prueba según si la empresa del aviso aparece 
en los datos de entrenamiento — esto se calcula sobre la columna `company` cruda, sin
importar si el modelo la usó como variable. En la variante sin empresa el
modelo no tiene acceso a esa columna, pero el corte vista/no vista se sigue
aplicando igual, ya que solo depende de los datos, no del feature set., y 
compara ambos subconjuntos contra un dummy de mediana calculado por separado para 
cada uno. Vive en `ml.lightgbm_baseline.evaluation` (`company_generalization_metrics`,
`company_generalization_dummy_metrics`, `company_generalization_summary`) y
se ejecuta automáticamente después de las métricas por segmento en `run.py`,
escribiendo `company_generalization_metrics.csv` y
`company_generalization_summary.json` en el directorio de artefactos de cada
corrida (registrados en MLflow igual que el resto).

Es **opcional**, controlado desde `params.yaml`:

```yaml
evaluation:
  company_generalization:
    enabled: true
    minimum_size: 30
```

o desde la línea de comandos, sin tocar el archivo:

```bash
python -m ml.lightgbm_baseline.run --no-company-generalization
```

### Resultado de referencia (2 de septiembre de 2026)

| Variante | Segmento | Filas | MAE medio | Mejora vs. mediana del segmento |
| --- | --- | ---: | ---: | ---: |
| Con empresa | Vista | 6.152 | USD 28.959 | 45,9 % |
| Con empresa | No vista | 1.421 | USD 35.736 | 40,8 % |
| Sin empresa | Vista | 6.152 | USD 39.043 | 27,1 % |
| Sin empresa | No vista | 1.421 | USD 35.792 | 40,8 % |

18,76% del conjunto de prueba corresponde a empresas no vistas en
entrenamiento. El beneficio de incluir `company` se concentra casi por
completo en el segmento de empresas ya vistas (~10.000 USD de mejora en MAE);
en empresas no vistas, con y sin empresa quedan prácticamente empatados
(35.736 vs. 35.792 USD) — evidencia directa de que la identidad de empresa no
está generalizando a empleadores nuevos, justo la población más relevante
para el enfoque de equidad salarial remota LatAm del proyecto.

### Variantes para comparar contra la corrida de CatBoost

```bash
# Experimento ampliado (reportado + híbrido + estimado, métricas por procedencia)
python -m ml.lightgbm_baseline.run --experiment expanded

# Dos objetivos directos en vez de log(mínimo) + log(amplitud)
python -m ml.lightgbm_baseline.run --target-strategy direct

# Sin identidad de empresa, para comparar la generalización a empleadores no vistos
python -m ml.lightgbm_baseline.run --exclude-company --run-name lightgbm-sin-empresa
```

La selección del modelo debe basarse en las métricas de **validación**; el
conjunto de prueba se reserva estrictamente para el reporte final, la misma
regla que sigue el baseline de CatBoost.

## Notebook interactivo

`notebooks/lightgbm_local.ipynb` (raíz del repositorio) es la versión local 
para exploración interactiva: carga, split, entrenamiento, métricas, generalización 
por empresa, registro en MLflow local, y una matriz compacta de variantes.

## MLflow

Por defecto, la corrida se registra en un almacén SQLite local:

```text
ml/lightgbm_baseline/.mlflow/mlflow.db
```

```bash
mlflow server \
  --backend-store-uri sqlite:///ml/lightgbm_baseline/.mlflow/mlflow.db \
  --host 127.0.0.1 --port 5000
```

También puede apuntarse al servidor de tracking compartido en EC2 sin cambiar
el código:

```bash
export MLFLOW_TRACKING_URI="http://<ec2-host>:5000"
python -m ml.lightgbm_baseline.run
```

## Pruebas

```bash
python -m pytest ml/lightgbm_baseline/tests/test_model.py
```

`test_model.py` cubre únicamente las piezas específicas de LightGBM (manejo de
tipos de dato categóricos, el envoltorio de entrenamiento, la forma de la
importancia de variables) — la lógica compartida de datos/variables/evaluación
ya está cubierta por `test_pipeline.py` del paquete de CatBoost.