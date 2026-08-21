import { Link } from "react-router-dom";
import {
  Database,
  DollarSign,
  Briefcase,
  Sparkles,
  BarChart3,
  ArrowRight,
  TrendingUp,
} from "lucide-react";
import { PageHeader } from "../../components/ui/PageHeader/PageHeader";
import { Card } from "../../components/ui/Card/Card";
import styles from "./HomePage.module.css";

export function HomePage() {
  return (
    <div>
      <PageHeader
        title="Dashboard General"
        subtitle="Monitoreo del mercado tecnológico y predicciones salariales basadas en Machine Learning"
        badge="ML Model v1.0"
      />

      {/* Metrics Row */}
      <section className={styles.metricsGrid} aria-label="Métricas Principales">
        <Card>
          <div className={styles.metricCard}>
            <div className={styles.metricIconWrapper}>
              <Database size={24} />
            </div>
            <div className={styles.metricContent}>
              <span className={styles.metricValue}>1,450+</span>
              <span className={styles.metricLabel}>Ofertas Ingeridas</span>
              <span className={styles.metricTrend}>+12.5% este mes</span>
            </div>
          </div>
        </Card>

        <Card>
          <div className={styles.metricCard}>
            <div className={styles.metricIconWrapper}>
              <DollarSign size={24} />
            </div>
            <div className={styles.metricContent}>
              <span className={styles.metricValue}>$52,800</span>
              <span className={styles.metricLabel}>Salario Medio Anual</span>
              <span className={styles.metricTrend}>USD mercado tech</span>
            </div>
          </div>
        </Card>

        <Card>
          <div className={styles.metricCard}>
            <div className={styles.metricIconWrapper}>
              <Briefcase size={24} />
            </div>
            <div className={styles.metricContent}>
              <span className={styles.metricValue}>18</span>
              <span className={styles.metricLabel}>Roles Clasificados</span>
              <span className={styles.metricTrend}>Data, Dev & DevOps</span>
            </div>
          </div>
        </Card>

        <Card>
          <div className={styles.metricCard}>
            <div className={styles.metricIconWrapper}>
              <Sparkles size={24} />
            </div>
            <div className={styles.metricContent}>
              <span className={styles.metricValue}>94.2%</span>
              <span className={styles.metricLabel}>R² Score Modelo</span>
              <span className={styles.metricTrend}>Gradient Boosting</span>
            </div>
          </div>
        </Card>
      </section>

      {/* Main Grid: Charts & Quick Actions */}
      <section className={styles.mainGrid}>
        <Card
          title="Distribución Salarial por Rango (Placeholder)"
          subtitle="Densidad y distribución estadística calculada sobre el dataset ingerido"
          action={<BarChart3 size={18} color="var(--color-primary)" />}
        >
          <div className={styles.chartPlaceholder}>
            <div className={styles.chartBarsVisual}>
              <div className={styles.bar} style={{ height: "40%" }} />
              <div className={styles.bar} style={{ height: "65%" }} />
              <div className={styles.bar} style={{ height: "90%" }} />
              <div className={styles.bar} style={{ height: "75%" }} />
              <div className={styles.bar} style={{ height: "50%" }} />
              <div className={styles.bar} style={{ height: "30%" }} />
            </div>
            <span>[ Visualización interactiva de distribución salarial ]</span>
          </div>
        </Card>

        <Card
          title="Acciones Rápidas"
          subtitle="Accesos directos del pipeline"
        >
          <div className={styles.quickActionsList}>
            <Link to="/prediction" className={styles.actionItem}>
              <span>Calcular nueva predicción</span>
              <ArrowRight size={16} />
            </Link>
            <Link to="/explore" className={styles.actionItem}>
              <span>Explorar dataset completo</span>
              <ArrowRight size={16} />
            </Link>
            <Link to="/comparisons" className={styles.actionItem}>
              <span>Comparar roles y seniorities</span>
              <ArrowRight size={16} />
            </Link>
            <Link to="/about" className={styles.actionItem}>
              <span>Ver arquitectura MLOps</span>
              <ArrowRight size={16} />
            </Link>
          </div>
        </Card>
      </section>

      {/* Bottom Grid: Demand & Recent Samples */}
      <section className={styles.bottomGrid}>
        <Card
          title="Habilidades y Tecnologías Destacadas"
          subtitle="Frecuencia en ofertas con mejor remuneración"
          action={<TrendingUp size={18} color="var(--color-success)" />}
        >
          <div className={styles.techTagList}>
            <span className={styles.techTag}>Python <span className={styles.techCount}>420</span></span>
            <span className={styles.techTag}>React <span className={styles.techCount}>380</span></span>
            <span className={styles.techTag}>AWS <span className={styles.techCount}>310</span></span>
            <span className={styles.techTag}>Docker <span className={styles.techCount}>290</span></span>
            <span className={styles.techTag}>SQL <span className={styles.techCount}>270</span></span>
            <span className={styles.techTag}>TypeScript <span className={styles.techCount}>240</span></span>
            <span className={styles.techTag}>Kubernetes <span className={styles.techCount}>180</span></span>
            <span className={styles.techTag}>PyTorch / ML <span className={styles.techCount}>150</span></span>
          </div>
        </Card>

        <Card
          title="Muestra de Registros Procesados"
          subtitle="Últimos registros estandarizados por el pipeline de ingesta"
        >
          <div className={styles.tablePlaceholder}>
            <div className={styles.tableRow}>
              <div>
                <span className={styles.roleTitle}>Senior MLOps Engineer</span>
                <p style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>Remoto • LATAM / USA</p>
              </div>
              <span className={styles.roleSalary}>$75,000 - $95,000</span>
            </div>
            <div className={styles.tableRow}>
              <div>
                <span className={styles.roleTitle}>Full Stack Developer</span>
                <p style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>Híbrido • React & Node</p>
              </div>
              <span className={styles.roleSalary}>$45,000 - $60,000</span>
            </div>
            <div className={styles.tableRow}>
              <div>
                <span className={styles.roleTitle}>Data Scientist</span>
                <p style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>Remoto • Python & NLP</p>
              </div>
              <span className={styles.roleSalary}>$55,000 - $70,000</span>
            </div>
          </div>
        </Card>
      </section>
    </div>
  );
}
