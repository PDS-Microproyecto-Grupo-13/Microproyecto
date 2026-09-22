import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowRight,
  BarChart3,
  Code2,
  Database,
  DollarSign,
  LoaderCircle,
  RefreshCw,
  Sparkles,
  TriangleAlert,
} from "lucide-react";
import { Card } from "../../components/ui/Card/Card";
import { PageHeader } from "../../components/ui/PageHeader/PageHeader";
import {
  getAnalyticsSummary,
  type AnalyticsSummary,
  type CategoryCount,
  type SalaryBin,
} from "../../services/analyticsApi";
import { ApiError } from "../../services/http";
import styles from "./HomePage.module.css";

type AnalyticsState =
  | { status: "loading" }
  | { status: "success"; summary: AnalyticsSummary }
  | { status: "unpublished" }
  | { status: "error"; message: string };

const integerFormatter = new Intl.NumberFormat("es-CO", {
  maximumFractionDigits: 0,
});
const currencyFormatter = new Intl.NumberFormat("es-CO", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});
const percentFormatter = new Intl.NumberFormat("es-CO", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
const dateFormatter = new Intl.DateTimeFormat("es-CO", {
  day: "numeric",
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});

const seniorityLabels: Record<string, string> = {
  SE: "Senior",
  MI: "Mid",
  EN: "Entry",
  EX: "Executive",
  desconocido: "Desconocido",
};

const technologyLabels: Record<string, string> = {
  aws: "AWS",
  azure: "Azure",
  docker: "Docker",
  gcp: "GCP",
  kubernetes: "Kubernetes",
  machine_learning: "Machine Learning",
  power_bi: "Power BI",
  python: "Python",
  pytorch: "PyTorch",
  spark: "Spark",
  sql: "SQL",
  tableau: "Tableau",
  tensorflow: "TensorFlow",
};

function isAbortError(cause: unknown): boolean {
  return cause instanceof DOMException && cause.name === "AbortError";
}

function analyticsErrorMessage(cause: unknown): string {
  if (
    cause instanceof ApiError &&
    cause.status === 503 &&
    cause.code === "analytics_storage_unavailable"
  ) {
    return "La analítica no está disponible temporalmente. Intenta nuevamente en unos minutos.";
  }
  if (cause instanceof ApiError) return cause.message;
  return "No fue posible conectar con el backend de analítica.";
}

function salaryBinLabel(bin: SalaryBin): string {
  if (bin.upper_bound_usd === null) {
    return `${currencyFormatter.format(bin.lower_bound_usd)} o más`;
  }
  return `${currencyFormatter.format(bin.lower_bound_usd)} – ${currencyFormatter.format(
    bin.upper_bound_usd,
  )}`;
}

function categoryLabel(item: CategoryCount): string {
  return seniorityLabels[item.category] ?? item.category;
}

function LoadingState() {
  return (
    <section className={styles.statePanel} aria-live="polite" aria-busy="true">
      <LoaderCircle className={styles.spinner} size={30} />
      <h2>Cargando analítica</h2>
      <p>Consultando el resumen publicado por el backend.</p>
    </section>
  );
}

interface RetryStateProps {
  kind: "unpublished" | "error";
  message: string;
  onRetry: () => void;
}

function RetryState({ kind, message, onRetry }: RetryStateProps) {
  const Icon = kind === "unpublished" ? Database : TriangleAlert;
  return (
    <section
      className={styles.statePanel}
      role={kind === "error" ? "alert" : "status"}
    >
      <Icon className={styles.stateIcon} size={32} />
      <h2>
        {kind === "unpublished"
          ? "Analítica aún no publicada"
          : "Analítica no disponible"}
      </h2>
      <p>{message}</p>
      <button className={styles.retryButton} type="button" onClick={onRetry}>
        <RefreshCw size={16} />
        Reintentar
      </button>
    </section>
  );
}

function AnalyticsDashboard({ summary }: { summary: AnalyticsSummary }) {
  const { dataset, metadata, model } = summary;
  const maxSalaryBinCount = Math.max(
    ...dataset.salary_midpoint_distribution.map((bin) => bin.count),
    1,
  );
  const period = `${dateFormatter.format(
    new Date(dataset.data_range.published_min),
  )} – ${dateFormatter.format(new Date(dataset.data_range.published_max))}`;

  return (
    <>
      <section className={styles.metricsGrid} aria-label="Métricas principales">
        <Card>
          <div className={styles.metricCard}>
            <div className={styles.metricIconWrapper}>
              <Database size={24} />
            </div>
            <div className={styles.metricContent}>
              <span className={styles.metricValue}>
                {integerFormatter.format(dataset.counts.modelable_rows)}
              </span>
              <span className={styles.metricLabel}>Vacantes modelables</span>
              <span className={styles.metricDetail}>
                {integerFormatter.format(dataset.counts.validated_rows)} validadas
                {" · "}{metadata.snapshot_count} snapshots
              </span>
            </div>
          </div>
        </Card>

        <Card>
          <div className={styles.metricCard}>
            <div className={styles.metricIconWrapper}>
              <DollarSign size={24} />
            </div>
            <div className={styles.metricContent}>
              <span className={styles.metricValue}>
                {currencyFormatter.format(dataset.salary_midpoint.median_usd)}
              </span>
              <span className={styles.metricLabel}>
                Punto medio mediano anual
              </span>
              <span className={styles.metricDetail}>
                Media {currencyFormatter.format(dataset.salary_midpoint.mean_usd)} / año
              </span>
            </div>
          </div>
        </Card>

        <Card>
          <div className={styles.metricCard}>
            <div className={styles.metricIconWrapper}>
              <BarChart3 size={24} />
            </div>
            <div className={styles.metricContent}>
              <span className={styles.metricValue}>
                {currencyFormatter.format(model.mae_average_usd)}
              </span>
              <span className={styles.metricLabel}>MAE promedio test</span>
              <span className={styles.metricDetail}>
                {integerFormatter.format(model.evaluation_rows)} filas de evaluación
              </span>
            </div>
          </div>
        </Card>

        <Card>
          <div className={styles.metricCard}>
            <div className={styles.metricIconWrapper}>
              <Sparkles size={24} />
            </div>
            <div className={styles.metricContent}>
              <span className={styles.metricValue}>
                {percentFormatter.format(model.r2_salary_min)} –{" "}
                {percentFormatter.format(model.r2_salary_max)}
              </span>
              <span className={styles.metricLabel}>R² evaluación</span>
              <span className={styles.metricDetail}>
                {model.algorithm} · evaluación del snapshot
              </span>
            </div>
          </div>
        </Card>
      </section>

      <section className={styles.mainGrid}>
        <Card
          title="Distribución del punto medio salarial"
          subtitle="Vacantes modelables por rango anual USD"
          action={<BarChart3 size={18} color="var(--color-primary)" />}
        >
          <div className={styles.salaryChart} aria-label="Histograma salarial">
            {dataset.salary_midpoint_distribution.map((bin) => (
              <div className={styles.salaryBarColumn} key={bin.lower_bound_usd}>
                <span className={styles.salaryBarCount}>
                  {integerFormatter.format(bin.count)}
                </span>
                <div className={styles.salaryBarTrack}>
                  <div
                    className={styles.salaryBar}
                    style={{
                      height: `${Math.max(
                        (bin.count / maxSalaryBinCount) * 100,
                        2,
                      )}%`,
                    }}
                    title={`${salaryBinLabel(bin)}: ${integerFormatter.format(
                      bin.count,
                    )} vacantes (${percentFormatter.format(bin.proportion)})`}
                  />
                </div>
                <span className={styles.salaryBarLabel}>
                  {salaryBinLabel(bin)}
                </span>
              </div>
            ))}
          </div>
        </Card>

        <Card
          title="Acciones rápidas"
          subtitle="Accesos a las herramientas del producto"
        >
          <div className={styles.quickActionsList}>
            <Link to="/prediction" className={styles.actionItem}>
              <span>Calcular nueva predicción</span>
              <ArrowRight size={16} />
            </Link>
            <Link to="/explore" className={styles.actionItem}>
              <span>Explorar datos</span>
              <ArrowRight size={16} />
            </Link>
            <Link to="/comparisons" className={styles.actionItem}>
              <span>Comparar seniorities</span>
              <ArrowRight size={16} />
            </Link>
            <Link to="/about" className={styles.actionItem}>
              <span>Ver arquitectura del proyecto</span>
              <ArrowRight size={16} />
            </Link>
          </div>
        </Card>
      </section>

      <section className={styles.bottomGrid}>
        <Card
          title="Tecnologías más frecuentes"
          subtitle="Menciones detectadas en vacantes modelables"
          action={<Code2 size={18} color="var(--color-success)" />}
        >
          <div className={styles.techTagList}>
            {dataset.top_technologies.map((technology) => (
              <span className={styles.techTag} key={technology.technology}>
                {technologyLabels[technology.technology] ?? technology.technology}
                <span className={styles.techCount}>
                  {integerFormatter.format(technology.count)}
                </span>
                <span className={styles.techProportion}>
                  {percentFormatter.format(technology.proportion)}
                </span>
              </span>
            ))}
          </div>
        </Card>

        <Card
          title="Distribución por seniority"
          subtitle={`Vacantes publicadas: ${period}`}
        >
          <div className={styles.distributionList}>
            {dataset.seniority_distribution.map((item) => (
              <div className={styles.distributionRow} key={item.category}>
                <span className={styles.distributionLabel}>
                  {categoryLabel(item)}
                </span>
                <div className={styles.distributionValue}>
                  <strong>{integerFormatter.format(item.count)}</strong>
                  <span>{percentFormatter.format(item.proportion)}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </section>
    </>
  );
}

export function HomePage() {
  const [state, setState] = useState<AnalyticsState>({ status: "loading" });
  const [requestVersion, setRequestVersion] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    getAnalyticsSummary({ signal: controller.signal })
      .then((summary) => {
        if (active) setState({ status: "success", summary });
      })
      .catch((cause: unknown) => {
        if (!active || isAbortError(cause)) return;
        if (
          cause instanceof ApiError &&
          cause.status === 404 &&
          cause.code === "analytics_not_published"
        ) {
          setState({ status: "unpublished" });
          return;
        }
        setState({ status: "error", message: analyticsErrorMessage(cause) });
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [requestVersion]);

  const retry = () => {
    setState({ status: "loading" });
    setRequestVersion((current) => current + 1);
  };

  return (
    <div>
      <PageHeader
        title="Dashboard General"
        subtitle="Analítica agregada del mercado tecnológico y evaluación del snapshot"
        badge={
          state.status === "success"
            ? `Analytics v${state.summary.schema_version}`
            : "Analytics"
        }
      />

      {state.status === "loading" && <LoadingState />}
      {state.status === "unpublished" && (
        <RetryState
          kind="unpublished"
          message="El backend está listo, pero todavía no recibió un snapshot analítico."
          onRetry={retry}
        />
      )}
      {state.status === "error" && (
        <RetryState kind="error" message={state.message} onRetry={retry} />
      )}
      {state.status === "success" && (
        <AnalyticsDashboard summary={state.summary} />
      )}
    </div>
  );
}
