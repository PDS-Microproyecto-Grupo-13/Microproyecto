# Ingesta de Datos de Foorilla

Alcance de este módulo: extraer datos de ofertas de empleo / salarios desde la API
de Foorilla, validarlos y escribirlos como un archivo CSV fechado para su entrega.


## Referencia de la API

- URL base: `https://foorilla.com/api/v1/`
- Autenticación: encabezado `Api-Key: YOUR_API_KEY`
- Límite de tasa: 5 solicitudes/segundo (restricción vinculante; también topado en 600/min)
- Docs: https://foorilla.com/api/v1/docs · Esquema: https://foorilla.com/api/v1/schema.json
- **Licencia: CC BY-SA 4.0 — se requiere atribución.** Ver `ATTRIBUTION.md`.

## Layout

```
src/ingestion/
├── config.py     # env-based config (API key, rate limit, etc.)
├── client.py     # Foorilla API client: auth, pagination, retries, throttling
├── schema.py     # pydantic validation of raw records (field names pending confirmation)
└── fetch.py      # orchestrates client -> validation -> dated CSV

data/raw/foorilla/
├── jobs_2026-08-15.csv                  # from /hiring/job/
├── jobs_2026-08-15_manifest.json        # counts, filters used, attribution
├── salaries_2026-08-15.csv              # from /insight/salary/
└── salaries_2026-08-15_manifest.json

notebooks/01_eda_ingested_data.ipynb    # starter notebook loading the latest jobs CSV
tests/unit/test_client.py               # mocked HTTP tests, no real API calls
ATTRIBUTION.md                          # CC BY-SA 4.0 attribution notice — read before publishing anything
```

## Configuración inicial

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar una extracción de ingesta

```bash
# Ofertas de empleo (por defecto), opcionalmente filtradas
python -m src.ingestion.fetch --endpoint jobs # ¡evitar esto!!!
python -m src.ingestion.fetch --endpoint jobs --topic 101 # Topic 101 = Data, AI and Machine Learning
python -m src.ingestion.fetch --endpoint jobs --title "ml engineer" --location "remote"

# Corrida de prueba rápida — detener tras unas pocas páginas en vez de recorrer todo el catálogo
python -m src.ingestion.fetch --endpoint jobs --max-pages 3

# Datos de salario
python -m src.ingestion.fetch --endpoint salaries
```

**⚠️ Una extracción de jobs sin filtrar es enorme.** Al 2026-08-15,
`/hiring/job/` sin filtros devuelve ~3.4 millones de registros (decenas de
miles de páginas). **En su lugar, filtrar por id de topic** — ver "Filtrar por
topic" más abajo para saber cómo buscar el id correcto. El filtrado por
`--title` (abajo) también sigue funcionando, pero el filtrado por topic es
más adecuado para acotar a "puestos de trabajo de IA".

```bash
python -m src.ingestion.fetch --endpoint jobs --title "machine learning engineer"
python -m src.ingestion.fetch --endpoint jobs --title "data scientist"
python -m src.ingestion.fetch --endpoint jobs --title "ai researcher"
```

Esto escribe `data/raw/foorilla/<endpoint>_<date>.csv` junto con un
`_manifest.json` correspondiente (que incluye el aviso de atribución — no
quitarlo al entregar los archivos) — **las filas de jobs se transmiten a
disco a medida que se validan**, así que se verá progreso parcial en disco
incluso a mitad de la corrida, y un `Ctrl+C` a mitad de camino deja un CSV
(parcial) válido en lugar de nada. Quien sea responsable del versionado se
encarga de estos archivos después; ese paso no se maneja en este módulo.

## Extracciones incrementales (poniéndose al día con publicaciones nuevas)

Una vez hecha una extracción completa inicial, usar `--published-after` para
obtener solo los empleos publicados desde entonces — mucho más rápido que
volver a extraer todo:

```bash
# Revisar qué fecha de corte usar, según el manifest de la última extracción
python -m src.ingestion.fetch --suggest-since

# Luego extraer solo lo nuevo
python -m src.ingestion.fetch --endpoint jobs --topic 101 --published-after 2026-08-15
```

**Advertencia:** `published_after` filtra por la fecha `published` del
empleo, no por el momento en que *se* extrajo — así que si Foorilla
reincorpora publicaciones más antiguas a su índice después del hecho, una
extracción incremental podría igual pasarlas por alto. Es un compromiso
razonable para este proyecto (punto ciego pequeño vs. horas ahorradas),
pero vale la pena hacer de vez en cuando una re-extracción completa (sin
`--published-after`) en lugar de depender de las incrementales para
siempre.

Cada extracción escribe su propio archivo fechado (`jobs_<pull-date>.csv`),
así que una extracción incremental no sobrescribirá la histórica completa —
quien sea responsable del versionado de datos deberá concatenar/deduplicar
entre estos archivos fechados en lugar de esperar un único archivo "los
datos".

## Pruebas

```bash
pytest tests/unit -v
```

Las pruebas simulan (mock) todas las llamadas HTTP (librería `responses`) —
no se necesita API key real ni acceso a la red para correr CI.

## Esquema de jobs

`JOB_CSV_COLUMNS` / `JobRaw` están confirmados al 2026-08-15 contra el
esquema real de Swagger/OpenAPI de la respuesta de `GET /hiring/job/` — no
solo contra la exportación manual anterior de CSV, la cual terminó siendo
diferente en algunos aspectos:

- **`is_agency` / `is_remote` están dentro de `company`**, no a nivel
  superior del job. La columna CSV `company_is_agency` refleja esto.
- **`views`, `clicks`, `foo_url`** (presentes en la exportación CSV del
  dashboard) **no aparecen en el esquema de la API en absoluto** —
  descartados de la salida de la ingesta ya que este módulo extrae desde
  la API, no desde la herramienta de exportación.
- **`topics` viene embebido en cada registro de job** — una lista de
  `{id, name, is_main, tags}`. No se necesita una búsqueda separada una vez
  que se tienen los datos del job; aplanado en `topics` (nombres separados
  por pipe) y `topic_ids`.
- **Existen campos de salario normalizados por moneda**: `salary_min_usd`/
  `_max_usd` y las variantes `_eur`, junto con `salary_min`/`_max` en la
  moneda original. **Usar las columnas USD/EUR para modelado
  entre países** — comparar `salary_min` en bruto entre monedas (INR vs.
  JPY vs. USD) no tiene sentido sin conversión, y este endpoint ya hizo esa
  conversión.

Salida CSV completa confirmada: `id`, `company`, `company_id`,
`company_is_agency`, `title`, `location`, `has_remote`, `work_mode`,
`published`, `expired`, `experience_level`, `experience_years`,
`language`, `salary_min`, `salary_max`, `salary_min_est`, `salary_max_est`,
`salary_currency`, `salary_min_usd`, `salary_max_usd`, `salary_min_eur`,
`salary_max_eur`, `topics`, `topic_ids`, `tags`, `regions`, `countries`,
`apply_url`.

## Filtrar por topic

La lista de Parameters documentada de `/hiring/job/` confirma un filtro
`topic` real — `array<integer>`, es decir, uno o más **ids** de topic (no
nombres). Conjunto completo de filtros disponible: `topic`, `tag`,
`region`, `country` (todos arrays de ids enteros), además de `title`,
`location`, `company`, `experience_level`, `language` (coincidencia
parcial de texto), `has_remote`/`company_remote`/`company_agency`
(booleanos), `work_mode` (entero), y `published_after`/`published_before`
(fechas).

Dado que la UI muestra *nombres* de topic (por ejemplo, "Data, AI, and
Machine Learning") pero la API requiere un *id*, hay que buscarlo primero:

```bash
python -m src.ingestion.fetch --find-topic "Data, AI"
# imprime: id=<N>  name='Data, AI, and Machine Learning'
```

Luego usar ese id para acotar la extracción real — mucho mejor que una
extracción sin filtrar de 3.4M de registros o adivinar cadenas de título:

```bash
python -m src.ingestion.fetch --endpoint jobs --topic <N>
python -m src.ingestion.fetch --endpoint jobs --topic <N> --topic <M>   # múltiples topics
```

Cualquier otro filtro documentado que aún no esté conectado como su propia
opción puede pasarse mediante `--extra KEY=VALUE` (repetible), por ejemplo
`--extra experience_level=senior`.

## ⚠️ El endpoint de salaries aún no está confirmado

Los nombres de campos de `/insight/salary/` no han sido confirmados de la
misma forma que los de jobs. `fetch_salaries()` actualmente infiere las
columnas del CSV a partir de lo que contenga el primer registro, en lugar
de una lista fija — obtener un registro de muestra (exportación del
dashboard o una llamada real a la API) antes de confiar en esto para una
ingesta real, y actualizar `SalaryRaw` en `schema.py` para que coincida,
siguiendo el mismo enfoque que se usó para jobs.