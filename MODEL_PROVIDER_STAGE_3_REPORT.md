# MODEL_PROVIDER — REPORTE SUBETAPA 3: Alineación del Contrato Backend ↔ Inferencia

**Fecha:** 2026-09-11  
**Repositorio:** `PDS-Microproyecto-Grupo-13/Microproyecto`  
**Servicio Backend:** `FastAPI (Puerto 8000)`  
**Servicio Inferencia:** `MLflow Scoring Server (Puerto 5001)`  
**Modelo:** `salary-predictor`  
**Alias:** `champion` (apuntando a versión 1)  

---

## 1. Resumen Ejecutivo

En esta Subetapa 3 se alineó de manera exhaustiva y retrocompatible el contrato de datos entre el servicio backend FastAPI y el servicio de inferencia MLflow PyFunc.

El backend ahora produce de forma estricta las **11 columnas canónicas** requeridas por el modelo `salary-predictor` registrado, resolviendo la obligatoriedad del campo `regions` mediante una estrategia de fallback segura (`"desconocido"`), habilitando opcionalmente `work_mode`, mapeando `country` hacia `countries` y `technologies` hacia `tags`, y validando cuantitativamente las predicciones devueltas por el modelo en [backend/app/clients/inference_client.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/backend/app/clients/inference_client.py).

Se ejecutó una prueba real de extremo a extremo atravesando:
`Cliente HTTP -> Backend (:8000) -> InferenceClient -> Inferencia (:5001) -> PyFunc LightGBM`
obteniendo HTTP 200 con predicciones salariales exactas y coherentes.

---

## 2. Estado Inicial Real del Contrato Backend

Al iniciar la inspección del checkout, se constató el siguiente estado en [backend/app/schemas/prediction.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/backend/app/schemas/prediction.py):
- `region` y `regions` estaban completamente ausentes en `SalaryPredictionRequest`.
- `to_mlflow_record()` generaba un payload con 10 columnas, incluyendo una columna espuria `"topics"` que no forma parte del contrato del modelo y omitiendo por completo `"regions"`.
- `work_mode` estaba hardcodeado a `None`, impidiendo que clientes que conocieran el modo de trabajo pudieran aportarlo.
- `MODEL_NAME` en [backend/app/core/config.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/backend/app/core/config.py) y en [docker-compose.yml](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/docker-compose.yml) (bloque `backend`) continuaba configurado con el valor obsoleto `salary_predict_model`.
- [backend/app/clients/inference_client.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/backend/app/clients/inference_client.py) no realizaba validaciones de sanidad numérica sobre las predicciones devueltas (unicidad de respuesta, valores finitos, positivos, `min <= max` ni consistencia del punto medio).

---

## 3. Cambios Implementados

### 3.1. Configuración de Nombre de Modelo y Alias
- Se actualizó el valor por defecto de `MODEL_NAME` a `salary-predictor` en [backend/app/core/config.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/backend/app/core/config.py).
- Se actualizó la variable de entorno `MODEL_NAME=salary-predictor` en el bloque `backend` de [docker-compose.yml](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/docker-compose.yml).
- Se sincronizaron los archivos de entorno [backend/.env.example](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/backend/.env.example) y `backend/.env`.

### 3.2. Esquema Público Retrocompatible ([backend/app/schemas/prediction.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/backend/app/schemas/prediction.py))
Se agregaron de forma opcional los campos:
```python
region: str | None = Field(default=None, max_length=100)
regions: str | None = Field(default=None, max_length=100)
work_mode: int | None = Field(default=None, ge=1, le=3)
```
- **Manejo de `regions`:** Se admite tanto `region` como `regions` en el payload JSON. En `to_mlflow_record()`, se resuelve:
  ```python
  raw_region = self.regions if self.regions is not None else self.region
  resolved_region = (raw_region or "desconocido").strip() or "desconocido"
  ```
  Si el cliente no envía región, se asigna limpiamente `"desconocido"`.
- **Manejo de `work_mode`:** Si el cliente envía `1`, `2` o `3`, se preserva su valor numérico; si se omite, se envía `None` y el modelo deriva la modalidad a partir de `has_remote` según la lógica canónica de `prepare_features()`.
- **Eliminación de campos espurios:** Se removió la columna `"topics"` del diccionario generado por `to_mlflow_record()`, produciendo exactamente las 11 columnas esperadas.

### 3.3. Robustecimiento de `InferenceClient` ([backend/app/clients/inference_client.py](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/backend/app/clients/inference_client.py))
Se añadieron salvaguardas estrictas al procesar la respuesta de `/invocations`:
1. **Unicidad:** Comprueba que la lista `predictions` contenga exactamente 1 elemento (`len(predictions) == 1`).
2. **Finitud y Positividad:** Valida `math.isfinite()` y que `min_usd > 0`, `max_usd > 0`, `mid_usd > 0`.
3. **Orden de Rango:** Valida que `min_usd <= max_usd`.
4. **Consistencia de Punto Medio:** Valida mediante `math.isclose()` que `midpoint == (min + max) / 2` con tolerancia adecuada.

---

## 4. Contrato Final Backend ↔ PyFunc

### 4.1. Esquema Público de Entrada (`SalaryPredictionRequest`)
```json
{
  "title": "str (2..160 chars)",
  "experience_level": "str ('EN' | 'MI' | 'SE' | 'EX')",
  "experience_years": "float | null (0..50, default: null)",
  "country": "str (2..100 chars)",
  "region": "str | null (opcional, default: null)",
  "regions": "str | null (opcional, default: null)",
  "work_mode": "int | null (opcional, 1..3, default: null)",
  "is_remote": "bool (default: false)",
  "company": "str | null (opcional, default: null)",
  "company_is_agency": "bool (default: false)",
  "technologies": "list[str] (opcional, default: [])",
  "topics": "list[str] (opcional, default: [])"
}
```

### 4.2. Tabla de Correspondencia Campo a Campo

| Backend Field | PyFunc Field | Transformación Aplicada | Requerido en Backend |
| :--- | :--- | :--- | :--- |
| `title` | `title` | `.strip()` | Obligatorio |
| `company` | `company` | `(self.company or "Sin información").strip() or "Sin información"` | Opcional (default: `None` -> `"Sin información"`) |
| `company_is_agency` | `company_is_agency` | Booleano nativo (`True`/`False`) | Opcional (default: `False`) |
| `country` | `countries` | `.strip()` | Obligatorio |
| `region` / `regions` | `regions` | `(self.regions or self.region or "desconocido").strip() or "desconocido"` | Opcional (default: `None` -> `"desconocido"`) |
| `experience_level` | `experience_level` | Valor categórico preservado (`EN`, `MI`, `SE`, `EX`) | Obligatorio |
| `experience_years` | `experience_years` | `float` numérico o `None` | Opcional (default: `None`) |
| `is_remote` | `has_remote` | Booleano nativo (`True`/`False`) | Opcional (default: `False`) |
| `work_mode` | `work_mode` | `int` (`1`, `2`, `3`) o `None` | Opcional (default: `None`) |
| `technologies` | `tags` | `"\|".join(self.technologies)` | Opcional (default: `[]` -> `""`) |
| *(sistema)* | `published` | `datetime.now(UTC).isoformat()` | Automático (ISO-8601 con UTC) |

---

## 5. Pruebas Reales de Integración (Backend → Inference)

### 5.1. Petición Completa con Región y Modalidad
**Llamada al Backend:**
```bash
curl -i -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Senior Machine Learning Engineer",
    "experience_level": "SE",
    "experience_years": 6.0,
    "country": "United States",
    "region": "Americas",
    "work_mode": 2,
    "is_remote": true,
    "company": "Tech Innovators Inc",
    "company_is_agency": false,
    "technologies": ["python", "machine learning", "docker", "kubernetes", "aws"],
    "topics": ["Data Science", "Machine Learning"]
  }'
```

**Payload Generado hacia MLflow (`http://inference:5001/invocations`):**
```json
{
  "dataframe_split": {
    "columns": [
      "title",
      "company",
      "company_is_agency",
      "countries",
      "regions",
      "experience_level",
      "experience_years",
      "has_remote",
      "work_mode",
      "tags",
      "published"
    ],
    "data": [
      [
        "Senior Machine Learning Engineer",
        "Tech Innovators Inc",
        false,
        "United States",
        "Americas",
        "SE",
        6.0,
        true,
        2,
        "python|machine learning|docker|kubernetes|aws",
        "2026-09-11T07:24:24.123456+00:00"
      ]
    ]
  }
}
```

**Respuesta Exitosa HTTP 200:**
```json
{
  "prediction": {
    "minimum_usd": 143410.50259451513,
    "maximum_usd": 211588.3403574479,
    "midpoint_usd": 177499.4214759815
  },
  "model": {
    "name": "salary-predictor",
    "alias": "champion"
  },
  "warnings": []
}
```

### 5.2. Petición Retrocompatible Mínima (Sin región, sin empresa, sin work_mode)
**Llamada al Backend:**
```bash
curl -i -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Backend Software Developer",
    "experience_level": "MI",
    "country": "Germany",
    "is_remote": false,
    "technologies": ["python", "sql", "fastapi"]
  }'
```

**Respuesta Exitosa HTTP 200:**
```json
{
  "prediction": {
    "minimum_usd": 109795.18049950516,
    "maximum_usd": 143139.78742587837,
    "midpoint_usd": 126467.48396269177
  },
  "model": {
    "name": "salary-predictor",
    "alias": "champion"
  },
  "warnings": [
    "No se informaron años de experiencia; el modelo imputó ese valor.",
    "No se informó empresa; la precisión puede ser menor para este perfil."
  ]
}
```

### 5.3. Evidencia de Logs Operacionales
- **En `mlops-backend`:**
  ```text
  INFO external_request_started
  INFO HTTP Request: POST http://inference:5001/invocations "HTTP/1.1 200 OK"
  INFO external_request_completed
  ```
- **En `mlops-inference`:**
  ```text
  INFO: 172.18.0.4:52386 - "POST /invocations HTTP/1.1" 200 OK
  ```

---

## 6. Resultados de la Suite de Pruebas

### 6.1. Pruebas de Backend (`pytest backend/tests -v`)
Se ejecutaron **21 tests** en [backend/tests](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/backend/tests), todos aprobados:
- Validación de 11 columnas exactas en request mínimo.
- Preservación de `region` y `regions`.
- Preservación de `work_mode`.
- Mapeo de `technologies` a `tags`.
- Configuración canónica `salary-predictor` y `champion`.
- Validación de respuestas de inferencia: unicidad, finitud, valores positivos, coherencia `min <= max` y consistencia de `midpoint`.
- Manejo y traducción de errores HTTP 502/ExternalServiceError.

```text
============================== 21 passed in 4.47s ==============================
```

### 6.2. Pruebas de Model Provider (`pytest model_provider/tests -v`)
Se revalidaron los **30 tests** de [model_provider/tests](file:///home/jorge/Descargas/mlops/micro-proyecto/Microproyecto/model_provider/tests) para garantizar que no hubo regresiones en promoción ni en serving:
```text
============================== 30 passed in 2.88s ==============================
```

---

## 7. Módulos Intactos y Restricciones Cumplidas

- **ml/ intacto:** No se alteraron pipelines, modelos `model.joblib`, PyFunc ni artefactos de DVC/MLflow.
- **frontend/ intacto:** No se modificó el código de React.
- **model_provider/inference intacto:** No se modificó el código de serving ni Dockerfiles/requirements en esta etapa.
- **MLflow Registry intacto:** La versión `1` continúa activa con el alias `champion`.

---

## 8. Estado de Contenedores y Siguiente Fase

- `mlops-tracking`: `Up (healthy)` en puerto 5000.
- `mlops-inference`: `Up (healthy)` en puerto 5001.
- `mlops-backend`: `Up (healthy)` en puerto 8000.
- El contrato Backend ↔ Inferencia se encuentra 100% estabilizado y verificado operacionalmente.

---

MODEL_PROVIDER_STAGE_3_STATUS=READY_FOR_STAGE_4
