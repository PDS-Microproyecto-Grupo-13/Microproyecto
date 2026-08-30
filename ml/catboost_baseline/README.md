# Propuesta base: CatBoost para rangos salariales

Este módulo convierte las decisiones del EDA en un ejercicio reproducible para
comparar un primer modelo CatBoost contra un predictor ingenuo de medianas. No es
todavía el modelo seleccionado para producción.

## Decisiones del ejercicio

- Integra todos los cortes `jobs_*.csv` recuperados por DVC.
- Conserva la versión más reciente y completa de cada `id`.
- Deduplica republicaciones por URL, empresa, cargo y ubicación normalizados.
- Construye `Y1 = salary_min_usd` y `Y2 = salary_max_usd`.
- Descarta objetivos faltantes, no positivos, desordenados y extremos lejanos con
  la misma regla del EDA.
- Usa por defecto únicamente rangos completamente reportados.
- Fija una división temporal 70/15/15 calculada sobre la muestra completa.
- Convierte `topics` y `tags` en indicadores binarios y entrega a CatBoost las
  demás categorías como texto.
- Entrena el mínimo y la amplitud en escala `log1p`; al reconstruir siempre se
  cumple `Y2 >= Y1`.
- Compara CatBoost contra la mediana de entrenamiento y registra resultados por
  segmento.

## Preparación

Desde la raíz del repositorio:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
python -m pip install -r ml\catboost_baseline\requirements.txt
```

Recupere los cuatro CSV versionados:

```powershell
dvc pull data\raw\foorilla\jobs_2026-08-16.csv.dvc `
  data\raw\foorilla\jobs_2026-08-20.csv.dvc `
  data\raw\foorilla\jobs_2026-08-24.csv.dvc `
  data\raw\foorilla\jobs_2026-08-28.csv.dvc
```

## Prueba rápida

Esta ejecución verifica el código, pero sus métricas no son válidas para comparar
modelos porque solo toma una fracción inicial de cada archivo:

```powershell
.venv\Scripts\python.exe -m ml.catboost_baseline.run `
  --max-rows-per-file 10000 `
  --iterations 20 `
  --no-mlflow `
  --run-name smoke-test
```

## Entrenamiento base

```powershell
.venv\Scripts\python.exe -m ml.catboost_baseline.run
```

La configuración está en `params.yaml`. Los artefactos locales, ignorados por
Git, incluyen:

- Dos modelos CatBoost nativos (`.cbm`).
- Métricas de validación, prueba y baseline ingenuo.
- Predicciones sobre prueba.
- Métricas por procedencia, experiencia, país y familia de cargo.
- Importancia de variables.
- Manifiesto con commit, punteros DVC, versiones y parámetros.

### Resultado de referencia (30 de agosto de 2026)

Se entrenó sobre 39.707 vacantes con salario reportado, se validó con 6.931 y se
evaluó una sola vez sobre 7.549 observaciones posteriores en el tiempo.

| Variante | Variables | MAE medio validación | MAE medio prueba | Mejora vs. mediana |
| --- | ---: | ---: | ---: | ---: |
| Con empresa | 54 | USD 28.654 | USD 31.825 | 41,8 % |
| Sin empresa | 53 | USD 36.901 | USD 38.520 | 29,6 % |
| Mediana de entrenamiento | — | — | USD 54.722 | — |

La identidad de la empresa aporta bastante precisión, pero puede limitar la
generalización a empleadores nuevos. Por eso conviene conservar ambas corridas
como comparación y decidir el caso de uso antes de seleccionar una. La corrida
con empresa obtuvo `R² = 0,468` para el mínimo y `R² = 0,545` para el máximo; no
debe interpretarse todavía como modelo final.

## MLflow local

Por defecto, la corrida se registra en una base SQLite local:

```text
ml/catboost_baseline/.mlflow/mlflow.db
```

Para abrir la interfaz después del entrenamiento:

```powershell
.venv\Scripts\mlflow.exe server `
  --backend-store-uri sqlite:///ml/catboost_baseline/.mlflow/mlflow.db `
  --host 127.0.0.1 `
  --port 5000
```

También puede apuntarse a otro servidor sin cambiar el código:

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
.venv\Scripts\python.exe -m ml.catboost_baseline.run
```

## Variantes para discutir con el equipo

Experimento ampliado:

```powershell
.venv\Scripts\python.exe -m ml.catboost_baseline.run --experiment expanded
```

Dos objetivos directos, corrigiendo cruces solo al final:

```powershell
.venv\Scripts\python.exe -m ml.catboost_baseline.run --target-strategy direct
```

Sin identidad de empresa, para medir dependencia de empleadores conocidos:

```powershell
.venv\Scripts\python.exe -m ml.catboost_baseline.run --exclude-company --run-name catboost-sin-empresa
```

La selección final debe usar validación. El conjunto de prueba no debe emplearse
para ajustar variables, hiperparámetros o pesos de un ensamble.
