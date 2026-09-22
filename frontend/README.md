# SalaryPredict — Frontend Web Dashboard

Aplicación de interfaz de usuario de **SalaryPredict v1.0** desarrollada como una Single Page Application (SPA) con React 19, TypeScript y Vite.

---

## 1. Responsabilidad

- Proporcionar la interfaz interactiva para capturar las características de un perfil laboral (cargo, nivel de experiencia, años, país, modalidad remota, empresa, tecnologías y temas).
- Consumir el microservicio backend mediante rutas relativas `/api/...`.
- Presentar visualmente las proyecciones salariales anuales en USD (mínimo, máximo y punto medio), advertencias operacionales y los metadatos del modelo en ejecución (`name`, `alias`, `version`).
- Proveer vistas informativas y de exploración del proyecto (`Home`, `SalaryPrediction`, `ExploreData`, `Comparisons`, `About`).

---

## 2. Organización del Módulo

```text
frontend/
├── public/              # Favicons y recursos estáticos
├── src/
│   ├── app/             # Router principal y configuración de rutas
│   ├── components/      # Componentes UI reutilizables y navegación
│   ├── layouts/         # Layout general con Topbar y Sidebar
│   ├── pages/           # Vistas (Home, SalaryPrediction, ExploreData, Comparisons, About)
│   ├── services/        # Clientes HTTP tipados para predicción y analytics
│   └── styles/          # Estilos globales y Tailwind CSS
├── Dockerfile           # Multi-stage build (Node 22 -> Nginx alpine)
├── nginx.conf           # Configuración del reverse proxy para producción
├── package.json         # Dependencias y scripts
└── vite.config.ts       # Configuración de bundling y proxy local
```

---

## 3. Puesta en Marcha Local (Desarrollo)

Requisitos: **Node.js 20+** y **npm**.

```bash
# 1. Instalar dependencias
npm install

# 2. Iniciar servidor de desarrollo con Hot Module Replacement (HMR)
npm run dev
```

La aplicación quedará disponible en:
```text
http://localhost:5173
```

---

## 4. Calidad de Código y Compilación

```bash
# Linter de código estático (Oxlint)
npm run lint

# Verificación estricta de tipos TypeScript y build de producción
npm run build
```

---

## 5. Integración con el Monorepo

El frontend nunca se comunica directamente con el servidor de MLflow ni con el motor de inferencia:

```text
Navegador / SPA  ──>  Backend (:8000)  ──>  Inference (:5001 / :5002)
```

- **En desarrollo local**: `frontend/vite.config.ts` redirige las peticiones `/api` automáticamente hacia `http://localhost:8000`.
- **En Docker**: `frontend/nginx.conf` actúa como reverse proxy dirigiendo `/api/` hacia el servicio `backend:8000` dentro de la red privada de Docker (`mlops-net`).

---

## 6. Variables y Configuración

El frontend **no requiere archivo `.env`**. Todas las solicitudes hacia el backend se realizan mediante rutas relativas. La resolución del host y puerto es gestionada íntegramente por el proxy correspondiente (Vite o Nginx).

Home consulta exclusivamente `GET /api/v1/analytics/summary`. Mientras espera
muestra loading; un `404 analytics_not_published` produce el estado explícito
"Analítica aún no publicada"; errores temporales o de red muestran una opción de
reintento. No existen métricas mock ni persistencia/cache del snapshot en el
navegador.

---

## 7. Fuera de Alcance del Módulo

- **No ejecuta inferencia**: No almacena estimadores ni realiza cálculos de Machine Learning en el cliente.
- **No carga artefactos ni modelos**: Desconoce la estructura binaria de LightGBM o Scikit-Learn.
- **No consulta MLflow Registry directamente**: Depende exclusivamente de los contratos HTTP expuestos por el backend.

---

## 8. Documentación Relacionada

- [`../README.md`](../README.md) — Visión general y arquitectura E2E de SalaryPredict.
- [`../DEPLOYMENT.md`](../DEPLOYMENT.md) — Manual reproducible de despliegue, bootstrap y puesta en marcha del stack completo.
- [`../backend/README.md`](../backend/README.md) — Contrato de la API REST consumida por el frontend.
