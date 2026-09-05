from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/03_experimentos_modelos.ipynb"

def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())

def source(text):
    return nbf.v4.new_code_cell(text.strip())

cells = [
markdown("""
# Comparación local de modelos para predicción de rangos salariales

En este notebook comparamos varias alternativas de regresión para predecir el límite inferior y superior del salario anual de una vacante. Nuestro propósito es elegir, con métricas reproducibles, la familia de modelo que pasará como insumo al modelado definitivo y que después ejecutaremos en EC2 con MLflow.

No usamos la prueba final para escoger el modelo: comparamos primero en validación temporal y reservamos el periodo más reciente para evaluarlo una única vez en el notebook 4.
"""),
markdown("""
## 1. Alcance, criterios y librerías

Trabajamos exclusivamente con rangos salariales reportados por las vacantes; no mezclamos los estimados por Foorilla en este experimento principal. Seleccionamos por menor MAE promedio de los dos límites y complementamos la decisión con RMSE, R², MAPE, error de amplitud, cobertura e incoherencia entre límites.
"""),
source("""
from pathlib import Path
from urllib.parse import urldefrag
import json, os, re, sys, warnings
import joblib, mlflow, mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
SEED = 42
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
DATA_DIR, ARTIFACT_DIR = ROOT / 'data', ROOT / 'artifacts'
EXPERIMENT_DIR = ARTIFACT_DIR / 'comparacion_modelos'
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE', 'true')
TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', (ROOT / 'mlruns').resolve().as_uri())
LOG_MODELS_TO_MLFLOW = os.getenv('LOG_MODELS_TO_MLFLOW', 'false').lower() == 'true'
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment('salarios-vacantes-comparacion-local')
print({'python': sys.version.split()[0], 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
       'mlflow': mlflow.__version__, 'lightgbm': __import__('lightgbm').__version__,
       'xgboost': __import__('xgboost').__version__, 'catboost': __import__('catboost').__version__,
       'semilla': SEED, 'tracking_uri': TRACKING_URI,
       'registrar_modelos_mlflow': LOG_MODELS_TO_MLFLOW})
"""),
markdown("""
## 2. Integración y limpieza de la muestra

Conservo la lógica del EDA: priorizo el corte más reciente por identificador, retiro republicaciones por URL, empresa, cargo y ubicación, y conservo rangos positivos y ordenados. También retiro extremos mediante la cerca exterior de Tukey sobre el punto medio logarítmico.
"""),
source("""
files = sorted(DATA_DIR.glob('jobs_*.csv'))
assert files, 'No se encontraron archivos jobs_*.csv en data/'
parts = []
for priority, path in enumerate(files):
    frame = pd.read_csv(path, low_memory=False)
    frame['_priority'] = priority
    parts.append(frame)
raw = pd.concat(parts, ignore_index=True)
raw['published'] = pd.to_datetime(raw['published'], errors='coerce', utc=True)
raw['_complete'] = raw.notna().sum(axis=1)
by_id = raw.sort_values(['id', '_priority', '_complete', 'published']).drop_duplicates('id', keep='last').copy()

def norm(values):
    return (values.fillna('').astype(str).str.lower().str.normalize('NFKD')
            .str.encode('ascii', errors='ignore').str.decode('ascii')
            .str.replace(r'[^a-z0-9]+', ' ', regex=True).str.strip())

url = by_id.apply_url.fillna('').astype(str).map(lambda value: urldefrag(value)[0].rstrip('/').lower())
by_id['_signature'] = url + '|' + norm(by_id.company) + '|' + norm(by_id.title) + '|' + norm(by_id.location)
by_id['_salary_info'] = by_id[['salary_min', 'salary_max', 'salary_min_usd', 'salary_max_usd']].notna().sum(axis=1)
dedup = by_id.sort_values(['_signature', '_salary_info', '_complete', 'published']).drop_duplicates('_signature', keep='last').copy()
for side in ['min', 'max']:
    dedup[f'y_{side}_usd'] = pd.to_numeric(dedup[f'salary_{side}_usd'], errors='coerce')
    dedup[f'{side}_reported'] = pd.to_numeric(dedup[f'salary_{side}'], errors='coerce').notna()
dedup['target_source'] = np.select([dedup.min_reported & dedup.max_reported,
                                    dedup.min_reported | dedup.max_reported],
                                   ['reportado', 'híbrido'], default='estimado')
valid = (dedup.y_min_usd.notna() & dedup.y_max_usd.notna() & (dedup.y_min_usd > 0) &
         (dedup.y_max_usd > 0) & (dedup.y_min_usd <= dedup.y_max_usd))
model_df = dedup.loc[valid].copy()
log_mid = np.log1p((model_df.y_min_usd + model_df.y_max_usd) / 2)
q1, q3 = log_mid.quantile([.25, .75])
model_df = model_df.loc[log_mid.between(q1 - 3 * (q3-q1), q3 + 3 * (q3-q1))].copy()
display(pd.Series({'filas_integradas': len(raw), 'identificadores_unicos': len(by_id),
                   'vacantes_deduplicadas': len(dedup), 'muestra_modelado': len(model_df),
                   'rangos_reportados': int(model_df.target_source.eq('reportado').sum())}))
"""),
markdown("""
## 3. Tipos de datos, variables y validación temporal

Tomamos como base la revisión de campos realizada en `02_eda.ipynb`. Las variables categóricas de texto requieren codificación, las numéricas requieren imputación y las etiquetas múltiples se convierten en indicadores binarios. Solo empleamos características disponibles al publicar: cargo, país, región, experiencia, modalidad, empresa, agencia, años, fecha y habilidades en `tags`. Excluimos salarios, conversiones, URL, identificadores y `expired` para evitar fuga de información. Usamos 70% para entrenamiento, 15% para validación y 15% para prueba final, respetando el orden temporal.
"""),
source("""
SKILLS = {'python':'python', 'sql':'sql', 'aws':'aws', 'azure':'azure', 'gcp':'gcp', 'spark':'spark',
          'docker':'docker', 'kubernetes':'kubernetes', 'machine_learning':'machine learning',
          'pytorch':'pytorch', 'tensorflow':'tensorflow', 'tableau':'tableau', 'power_bi':'power bi'}
def prepare_features(df):
    out = pd.DataFrame(index=df.index)
    out['title'] = norm(df.title).replace('', 'desconocido')
    out['country'] = df.countries.fillna('desconocido').astype(str).str.split('|').str[0].str.strip().replace('', 'desconocido')
    out['region'] = df.regions.fillna('desconocido').astype(str).str.split('|').str[0].str.strip().replace('', 'desconocido')
    out['experience_level'] = df.experience_level.fillna('desconocido').astype(str)
    remote = df.has_remote.fillna(False).astype(bool)
    work = pd.to_numeric(df.work_mode, errors='coerce')
    out['work_mode'] = np.select([~remote, work.eq(1), work.eq(2), work.eq(3)],
                                 ['presencial', 'híbrido', 'remoto', 'remoto_global'], default='remoto_sin_detalle')
    out['company'] = norm(df.company).replace('', 'desconocido')
    out['company_is_agency'] = df.company_is_agency.fillna(False).astype(int)
    out['experience_years'] = pd.to_numeric(df.experience_years, errors='coerce').where(lambda x: x.between(0, 50))
    out['experience_years_missing'] = out.experience_years.isna().astype(int)
    tags = df.tags.fillna('').astype(str).str.lower()
    for name, token in SKILLS.items(): out[f'skill_{name}'] = tags.str.contains(re.escape(token), regex=True).astype(int)
    out['published_year'], out['published_month'] = df.published.dt.year, df.published.dt.month
    return out

X = prepare_features(model_df)
TARGETS = ['y_min_usd', 'y_max_usd']
assert not [column for column in X if 'salary' in column.lower()], 'Hay fuga salarial en las variables'
ordered = model_df.loc[model_df.target_source.eq('reportado')].sort_values('published', na_position='first').index
n = len(ordered); cut_train, cut_val = int(.70*n), int(.85*n)
train_idx, val_idx, test_idx = ordered[:cut_train], ordered[cut_train:cut_val], ordered[cut_val:]
display(pd.DataFrame({'particion':['entrenamiento', 'validacion', 'prueba_final'],
                      'n':[len(train_idx), len(val_idx), len(test_idx)],
                      'inicio':[model_df.loc[x].published.min() for x in [train_idx, val_idx, test_idx]],
                      'fin':[model_df.loc[x].published.max() for x in [train_idx, val_idx, test_idx]]}))
feature_contract = pd.DataFrame([
    {'campo_origen':'title','tipo_origen':'texto','variables_modelo':'title','transformacion':'normalización + TargetEncoder'},
    {'campo_origen':'countries / regions','tipo_origen':'texto multivalor','variables_modelo':'country, region','transformacion':'primera ubicación + TargetEncoder'},
    {'campo_origen':'experience_level','tipo_origen':'categórica','variables_modelo':'experience_level','transformacion':'imputación + TargetEncoder'},
    {'campo_origen':'experience_years','tipo_origen':'numérica','variables_modelo':'experience_years, experience_years_missing','transformacion':'rango 0-50 + mediana + bandera'},
    {'campo_origen':'has_remote / work_mode','tipo_origen':'booleano + código','variables_modelo':'work_mode','transformacion':'modalidad homologada + TargetEncoder'},
    {'campo_origen':'company / company_is_agency','tipo_origen':'texto + booleano','variables_modelo':'company, company_is_agency','transformacion':'TargetEncoder + binaria'},
    {'campo_origen':'tags','tipo_origen':'texto multietiqueta','variables_modelo':'skill_*','transformacion':'vocabulario controlado binario'},
    {'campo_origen':'published','tipo_origen':'fecha','variables_modelo':'published_year, published_month','transformacion':'componentes numéricos'}])
display(feature_contract)
"""),
markdown("""
## 4. Preparación común y función de evaluación

Para Ridge y los modelos de árboles convertimos las variables categóricas mediante `TargetEncoder` con ajuste cruzado e imputamos las variables numéricas con la mediana. Las categorías nuevas reciben la media global aprendida en entrenamiento. CatBoost conserva las categorías como texto porque incorpora un tratamiento nativo para este tipo de variables. En todos los casos entrenamos un modelo para Y1 y otro para Y2 y ordenamos el resultado para evitar rangos invertidos.

Comparamos una línea base, un modelo lineal, cuatro familias incluidas en scikit-learn y tres implementaciones especializadas de boosting: LightGBM, XGBoost y CatBoost. Para que el experimento sea amplio pero reproducible, declaramos primero el rango de cada hiperparámetro y después tomamos una muestra determinística de configuraciones. El número de combinaciones por modelo puede ampliarse con `N_ITER_SEARCH` sin cambiar la lógica del notebook.
"""),
source("""
cat_cols = ['title', 'country', 'region', 'experience_level', 'work_mode', 'company']
num_cols = [column for column in X.columns if column not in cat_cols]
preprocessor = ColumnTransformer([
    ('categoricas', Pipeline([('imputar', SimpleImputer(strategy='most_frequent')),
                              ('codificar', TargetEncoder(target_type='continuous', smooth='auto', cv=5,
                                                          shuffle=True, random_state=SEED))]), cat_cols),
    ('numericas', SimpleImputer(strategy='median'), num_cols)], verbose_feature_names_out=False)

N_ITER_SEARCH = int(os.getenv('N_ITER_SEARCH', '5'))
MODEL_SPECS = {
    'baseline_mediana': {
        'family':'baseline', 'preprocessing':'TargetEncoder', 'scale':False,
        'fixed':{'strategy':'median'}, 'search':{}},
    'ridge': {
        'family':'lineal', 'preprocessing':'TargetEncoder + estandarización', 'scale':True,
        'fixed':{}, 'search':{'alpha':[0.1, 1.0, 10.0, 100.0]}},
    'hist_gradient_boosting': {
        'family':'boosting sklearn', 'preprocessing':'TargetEncoder', 'scale':False,
        'fixed':{'loss':'absolute_error'},
        'search':{'learning_rate':[0.03, 0.05, 0.08], 'max_iter':[180, 260, 360],
                  'max_leaf_nodes':[15, 31, 63], 'min_samples_leaf':[10, 20, 40],
                  'l2_regularization':[0.5, 1.0, 3.0]}},
    'gradient_boosting': {
        'family':'boosting sklearn', 'preprocessing':'TargetEncoder', 'scale':False,
        'fixed':{'loss':'huber'},
        'search':{'learning_rate':[0.03, 0.05, 0.08], 'n_estimators':[140, 220, 320],
                  'max_depth':[2, 3, 4], 'min_samples_leaf':[5, 10, 20],
                  'subsample':[0.8, 1.0]}},
    'random_forest': {
        'family':'bagging', 'preprocessing':'TargetEncoder', 'scale':False,
        'fixed':{},
        'search':{'n_estimators':[180, 280, 400], 'max_depth':[18, 24, None],
                  'min_samples_leaf':[2, 4, 8], 'max_features':[0.65, 0.8, 1.0]}},
    'extra_trees': {
        'family':'bagging', 'preprocessing':'TargetEncoder', 'scale':False,
        'fixed':{},
        'search':{'n_estimators':[220, 320, 440], 'max_depth':[20, 28, None],
                  'min_samples_leaf':[2, 4, 8], 'max_features':[0.7, 0.85, 1.0]}},
    'lightgbm': {
        'family':'boosting especializado', 'preprocessing':'TargetEncoder', 'scale':False,
        'fixed':{'objective':'regression_l1', 'verbosity':-1, 'subsample_freq':1},
        'search':{'n_estimators':[300, 500, 700], 'learning_rate':[0.025, 0.04, 0.06],
                  'num_leaves':[31, 63, 95], 'min_child_samples':[20, 50, 100],
                  'subsample':[0.8, 1.0], 'colsample_bytree':[0.8, 1.0],
                  'reg_lambda':[1.0, 3.0, 8.0]}},
    'xgboost': {
        'family':'boosting especializado', 'preprocessing':'TargetEncoder', 'scale':False,
        'fixed':{'objective':'reg:squarederror', 'tree_method':'hist'},
        'search':{'n_estimators':[300, 500, 700], 'learning_rate':[0.025, 0.04, 0.06],
                  'max_depth':[5, 7, 9], 'min_child_weight':[3, 7, 12],
                  'subsample':[0.8, 1.0], 'colsample_bytree':[0.8, 1.0],
                  'reg_lambda':[1.0, 3.0, 8.0]}},
    'catboost': {
        'family':'boosting especializado', 'preprocessing':'categóricas nativas', 'scale':False,
        'fixed':{'loss_function':'MAE', 'verbose':False, 'allow_writing_files':False},
        'search':{'iterations':[350, 550, 750], 'learning_rate':[0.025, 0.04, 0.06],
                  'depth':[6, 8, 10], 'l2_leaf_reg':[3.0, 6.0, 10.0],
                  'random_strength':[0.5, 1.0, 2.0]}}
}

def readable(values):
    return json.dumps(values, ensure_ascii=False, default=str)

catalog_rows=[];search_summary=[]
for name,spec in MODEL_SPECS.items():
    for parameter,value in spec['fixed'].items():
        catalog_rows.append({'modelo':name,'familia':spec['family'],'preprocesamiento':spec['preprocessing'],
                             'tipo':'fijo','hiperparámetro':parameter,'valores_considerados':readable(value)})
    for parameter,values in spec['search'].items():
        catalog_rows.append({'modelo':name,'familia':spec['family'],'preprocesamiento':spec['preprocessing'],
                             'tipo':'búsqueda','hiperparámetro':parameter,'valores_considerados':readable(values)})
    combinations=int(np.prod([len(values) for values in spec['search'].values()])) if spec['search'] else 1
    search_summary.append({'modelo':name,'parámetros_fijos':len(spec['fixed']),
                           'hiperparámetros_ajustados':len(spec['search']),
                           'combinaciones_posibles':combinations,
                           'configuraciones_evaluadas':min(N_ITER_SEARCH,combinations)})
hyperparameter_catalog=pd.DataFrame(catalog_rows)
search_summary=pd.DataFrame(search_summary)
display(hyperparameter_catalog.style.set_properties(subset=['valores_considerados'],**{'text-align':'left'}))
display(search_summary)
print(f'Conclusión del espacio de búsqueda: evaluaremos hasta {N_ITER_SEARCH} configuraciones por modelo '
      f'mediante muestreo aleatorio reproducible con semilla {SEED}.')
numeric_check=clone(preprocessor).fit_transform(X.loc[train_idx].head(5000),
                                                np.log1p(model_df.loc[train_idx].head(5000).y_min_usd))
display(pd.Series({'filas_verificadas':numeric_check.shape[0],'columnas_numericas':numeric_check.shape[1],
                   'tipo_resultante':str(numeric_check.dtype),'valores_no_finitos':int((~np.isfinite(numeric_check)).sum())}))

def metricas(y_true, raw_prediction):
    prediction = np.sort(np.asarray(raw_prediction), axis=1)
    result = {}
    for position, label in enumerate(['min', 'max']):
        result[f'mae_{label}'] = mean_absolute_error(y_true[:, position], prediction[:, position])
        result[f'rmse_{label}'] = mean_squared_error(y_true[:, position], prediction[:, position]) ** .5
        result[f'mape_{label}'] = mean_absolute_percentage_error(y_true[:, position], prediction[:, position])
        result[f'r2_{label}'] = r2_score(y_true[:, position], prediction[:, position])
    result['mae_promedio'] = (result['mae_min'] + result['mae_max']) / 2
    result['mae_amplitud'] = mean_absolute_error(y_true[:,1]-y_true[:,0], prediction[:,1]-prediction[:,0])
    result['cobertura_intervalo'] = float(np.mean((y_true[:,0] >= prediction[:,0]) & (y_true[:,1] <= prediction[:,1])))
    result['incoherencia_raw'] = float(np.mean(raw_prediction[:,0] > raw_prediction[:,1]))
    return result, prediction

def build_estimator(name, params):
    if name == 'baseline_mediana': return DummyRegressor(**params)
    if name == 'ridge': return Ridge(**params)
    if name == 'hist_gradient_boosting': return HistGradientBoostingRegressor(**params, random_state=SEED)
    if name == 'gradient_boosting': return GradientBoostingRegressor(**params, random_state=SEED)
    if name == 'random_forest': return RandomForestRegressor(**params, n_jobs=-1, random_state=SEED)
    if name == 'extra_trees': return ExtraTreesRegressor(**params, n_jobs=-1, random_state=SEED)
    if name == 'lightgbm': return LGBMRegressor(**params, n_jobs=-1, random_state=SEED)
    if name == 'xgboost': return XGBRegressor(**params, n_jobs=-1, random_state=SEED)
    if name == 'catboost': return CatBoostRegressor(**params, cat_features=cat_cols,
                                                     thread_count=-1, random_seed=SEED)
    raise KeyError(name)

def parameter_candidates(name):
    spec = MODEL_SPECS[name]
    if not spec['search']:
        return [dict(spec['fixed'])]
    total = int(np.prod([len(values) for values in spec['search'].values()]))
    sampled = list(ParameterSampler(spec['search'], n_iter=min(N_ITER_SEARCH, total), random_state=SEED))
    return [{**spec['fixed'], **candidate} for candidate in sampled]

planned_configurations=[]
for model_name in MODEL_SPECS:
    for number,params in enumerate(parameter_candidates(model_name),start=1):
        planned_configurations.append({'modelo':model_name,
            'configuracion':f'{model_name}__{number:02d}','hiperparametros':readable(params)})
planned_configurations=pd.DataFrame(planned_configurations)
display(planned_configurations.style.set_properties(subset=['hiperparametros'],**{'text-align':'left'}))
print('Las configuraciones anteriores son exactamente las que se evaluarán en las siguientes pruebas.')

def ejecutar_configuracion(name, params, configuration, train_rows, eval_rows, stage='validacion'):
    spec = MODEL_SPECS[name]
    fitted, predictions = [], []
    with mlflow.start_run(run_name=f'{stage}-{configuration}') as run:
        mlflow.log_params({'modelo': name, 'familia':spec['family'], 'semilla': SEED, 'n_train': len(train_rows),
                           'n_eval': len(eval_rows), 'particion': 'temporal_70_15_15',
                           'objetivo': 'rangos_reportados', 'transformacion_y': 'log1p',
                           'preprocesamiento':spec['preprocessing'],
                           **{f'hp_{k}':v for k,v in params.items()}})
        for target in TARGETS:
            estimator = build_estimator(name, params)
            if name == 'catboost':
                fitted_model = estimator.fit(X.loc[train_rows], np.log1p(model_df.loc[train_rows, target]))
            else:
                steps=[('preprocesamiento',clone(preprocessor))]
                if spec['scale']: steps.append(('estandarizacion',StandardScaler()))
                steps.append(('modelo',estimator))
                fitted_model=Pipeline(steps).fit(X.loc[train_rows], np.log1p(model_df.loc[train_rows, target]))
            fitted.append(fitted_model)
            predictions.append(np.expm1(fitted_model.predict(X.loc[eval_rows])))
        raw_prediction = np.column_stack(predictions)
        y_true = model_df.loc[eval_rows, TARGETS].to_numpy()
        scores, ordered_prediction = metricas(y_true, raw_prediction)
        mlflow.log_metrics(scores)
        output = EXPERIMENT_DIR / f'predicciones_{stage}_{configuration}.csv'
        pd.DataFrame({'id':model_df.loc[eval_rows,'id'].astype(str), 'y_min':y_true[:,0], 'y_max':y_true[:,1],
                      'pred_min':ordered_prediction[:,0], 'pred_max':ordered_prediction[:,1]}).to_csv(output, index=False)
        mlflow.log_artifact(output)
        if LOG_MODELS_TO_MLFLOW:
            for target, fitted_model in zip(TARGETS, fitted):
                mlflow.sklearn.log_model(fitted_model, name=f'{target}_{configuration}',
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        return {'modelo':name, 'familia':spec['family'], 'configuracion':configuration,
                'parametros':readable(params), 'etapa':stage, 'run_id':run.info.run_id, **scores}, fitted

def probar_modelo(name):
    partial=[]
    for number, params in enumerate(parameter_candidates(name), start=1):
        configuration=f'{name}__{number:02d}'
        result,_=ejecutar_configuracion(name,params,configuration,train_idx,val_idx)
        partial.append(result)
    best=pd.DataFrame(partial).sort_values('mae_promedio').iloc[0]
    print(f"Conclusión - {name}: se probaron {len(partial)} configuraciones. "
          f"La mejor fue {best.configuracion}, con MAE promedio USD {best.mae_promedio:,.0f} "
          f"y R² mínimo/máximo {best.r2_min:.3f}/{best.r2_max:.3f}.")
    return partial

"""),
markdown("""
## 5. Línea base: mediana histórica

La línea base predice la mediana del entrenamiento para todas las vacantes. Es el mínimo que debe superar un modelo útil; de otro modo no justificaría su complejidad.
"""),
source("""
resultados = probar_modelo('baseline_mediana')
"""),
markdown("""
## 6. Prueba 1: Ridge

Ridge establece una referencia lineal regularizada. Después de convertir todas las variables a números, estandarizamos las columnas para que la penalización sea comparable. Probamos varios niveles de penalización para identificar si las relaciones principales pueden explicarse sin ensambles de árboles.
"""),
source("""
resultados.extend(probar_modelo('ridge'))
"""),
markdown("""
## 7. Prueba 2: HistGradientBoosting

HistGradientBoosting usa boosting de árboles por histogramas. Variamos la tasa de aprendizaje, el número de iteraciones, el tamaño del árbol y la regularización para controlar el equilibrio entre capacidad y generalización.
"""),
source("""
resultados.extend(probar_modelo('hist_gradient_boosting'))
"""),
markdown("""
## 8. Prueba 3: Gradient Boosting

Gradient Boosting clásico construye árboles pequeños de manera secuencial para corregir los errores del ensamble anterior. Empleamos pérdida Huber para reducir la influencia de observaciones alejadas sin eliminar la señal salarial.
"""),
source("""
resultados.extend(probar_modelo('gradient_boosting'))
"""),
markdown("""
## 9. Prueba 4: Random Forest

Random Forest entrena muchos árboles sobre muestras y variables aleatorias. Lo probamos como alternativa robusta para verificar si el boosting realmente aporta una mejora. Restringimos profundidad y hojas para controlar sobreajuste y tiempo local.
"""),
source("""
resultados.extend(probar_modelo('random_forest'))
"""),
markdown("""
## 10. Prueba 5: Extra Trees

Extra Trees añade más aleatoriedad al elegir cortes que Random Forest. Es una alternativa útil para reducir varianza y contrastar dos ensambles de árboles frente al boosting sin asumir cuál será mejor.
"""),
source("""
resultados.extend(probar_modelo('extra_trees'))
"""),
markdown("""
## 11. Prueba 6: LightGBM

LightGBM construye árboles por hojas y está optimizado para datos tabulares. Probamos número de hojas, tasa de aprendizaje, cantidad de árboles, muestreo de filas y columnas y regularización. Conservamos el mismo `TargetEncoder` para que la comparación de variables de entrada sea consistente.
"""),
source("""
resultados.extend(probar_modelo('lightgbm'))
"""),
markdown("""
## 12. Prueba 7: XGBoost

XGBoost es otro ensamble secuencial de árboles. Su espacio de búsqueda controla profundidad, peso mínimo de los nodos, muestreo y penalización L2. Esto permite contrastarlo con LightGBM sin asumir que una implementación será mejor para estos datos.
"""),
source("""
resultados.extend(probar_modelo('xgboost'))
"""),
markdown("""
## 13. Prueba 8: CatBoost

CatBoost recibe directamente las variables categóricas y calcula sus representaciones durante el entrenamiento. Por esta razón no aplicamos `TargetEncoder` en este candidato. Variamos profundidad, iteraciones, tasa de aprendizaje y regularización, manteniendo la misma partición temporal y las mismas métricas de los demás modelos.
"""),
source("""
resultados.extend(probar_modelo('catboost'))
"""),
markdown("""
## 14. Mejor configuración por modelo y selección global

Primero escogemos la configuración con menor MAE promedio dentro de cada familia y después comparamos únicamente esos ganadores. De esta forma la tabla final muestra un representante por modelo y conserva los hiperparámetros exactos que lo produjeron. La prueba final permanece sin observar: se utilizará una sola vez en el notebook 4 después de cerrar la selección.
"""),
source("""
all_results=pd.DataFrame(resultados).sort_values(['modelo','mae_promedio']).reset_index(drop=True)
best_by_model=(all_results.sort_values('mae_promedio').groupby('modelo',as_index=False).first()
               .sort_values('mae_promedio').reset_index(drop=True))
columns=['modelo','configuracion','parametros','mae_min','mae_max','mae_promedio','rmse_min','rmse_max',
         'r2_min','r2_max','mae_amplitud','cobertura_intervalo','incoherencia_raw','run_id']
display(best_by_model[columns].style.format({'mae_min':'USD {:,.0f}','mae_max':'USD {:,.0f}',
    'mae_promedio':'USD {:,.0f}','rmse_min':'USD {:,.0f}','rmse_max':'USD {:,.0f}',
    'mae_amplitud':'USD {:,.0f}','r2_min':'{:.3f}','r2_max':'{:.3f}',
    'cobertura_intervalo':'{:.1%}','incoherencia_raw':'{:.2%}'}))
winner_row=best_by_model.iloc[0]
winner=winner_row.modelo
winner_params=json.loads(winner_row.parametros)
print(f"Modelo seleccionado para el notebook 4: {winner} ({winner_row.configuracion}).")
print(f"MAE promedio de validación: USD {winner_row.mae_promedio:,.0f}; "
      f"R² mínimo/máximo: {winner_row.r2_min:.3f}/{winner_row.r2_max:.3f}.")
"""),
markdown("""
## 15. Conclusiones y paso al modelo definitivo

- El mejor modelo es el que aparece en `winner`, seleccionado por el menor MAE promedio en la validación temporal. Este será el modelo que pase al notebook 4.
- En MLflow se almacenará la corrida del experimento con el nombre y la familia del modelo, sus hiperparámetros y las métricas de validación.
- También se almacenará el archivo de resultados del experimento como artefacto. Si `LOG_MODELS_TO_MLFLOW=true`, se guardará además el modelo entrenado.
"""),
source("""
selection = {'version':'comparacion-v2', 'modelo_seleccionado':winner,
          'configuracion_seleccionada':winner_row.configuracion,
          'criterio':'menor_mae_promedio_validacion_temporal',
          'mejores_por_modelo':best_by_model.to_dict(orient='records'),
          'todos_los_resultados':all_results.to_dict(orient='records'),
          'features':list(X.columns),
          'preprocesamiento':MODEL_SPECS[winner]['preprocessing'],
          'feature_contract':feature_contract.to_dict(orient='records'),
          'parametros':winner_params, 'semilla':SEED}
hyperparameter_catalog.to_csv(EXPERIMENT_DIR/'catalogo_hiperparametros.csv',index=False)
planned_configurations.to_csv(EXPERIMENT_DIR/'configuraciones_planeadas.csv',index=False)
all_results.to_csv(EXPERIMENT_DIR/'resultados_todas_configuraciones.csv',index=False)
best_by_model.to_csv(EXPERIMENT_DIR/'mejor_configuracion_por_modelo.csv',index=False)
with open(EXPERIMENT_DIR/'resumen_experimentos.json', 'w') as file:
    json.dump(selection, file, indent=2, default=str)
with open(EXPERIMENT_DIR/'seleccion_modelo.json','w') as file:
    json.dump({'modelo_seleccionado':winner,'familia':MODEL_SPECS[winner]['family'],
               'configuracion_seleccionada':winner_row.configuracion,
               'parametros':winner_params,'feature_columns':list(X.columns),
               'categorical_columns':cat_cols,'numeric_columns':num_cols,
               'preprocesamiento':MODEL_SPECS[winner]['preprocessing'],
               'metricas_validacion':{key:winner_row[key] for key in ['mae_min','mae_max','mae_promedio',
                   'rmse_min','rmse_max','r2_min','r2_max','mae_amplitud','cobertura_intervalo']},
               'espacio_busqueda':MODEL_SPECS[winner]['search']},file,indent=2,default=str)
print('Artefactos guardados en:', EXPERIMENT_DIR)
""")]

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.14'}})
OUT.parent.mkdir(exist_ok=True)
nbf.write(notebook, OUT)
print(OUT)
