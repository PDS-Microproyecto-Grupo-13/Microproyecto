import {
  ArrowLeftRight,
  ChevronDown,
  Scale,
  Sparkles,
  BarChart,
} from "lucide-react";
import { PageHeader } from "../../components/ui/PageHeader/PageHeader";
import { Card } from "../../components/ui/Card/Card";
import styles from "./ComparisonsPage.module.css";

export function ComparisonsPage() {
  return (
    <div className={styles.container}>
      <PageHeader
        title="Comparaciones de Mercado"
        subtitle="Analiza contrastes salariales entre diferentes especialidades, tecnologías y niveles de experiencia"
        badge="Benchmark MLOps"
      />

      {/* Selector Profile A vs Profile B */}
      <Card
        title="Configuración de Comparativa"
        subtitle="Selecciona dos perfiles para evaluar diferenciales salariales"
        action={<ArrowLeftRight size={18} color="var(--color-primary)" />}
      >
        <div className={styles.selectorGrid}>
          <div className={styles.selectBox}>
            <span className={styles.selectLabel}>Perfil A (Referencia)</span>
            <div className={styles.selectMock}>
              <span>Machine Learning Engineer (Senior)</span>
              <ChevronDown size={16} />
            </div>
          </div>

          <div className={styles.vsBadge}>VS</div>

          <div className={styles.selectBox}>
            <span className={styles.selectLabel}>Perfil B (Comparación)</span>
            <div className={styles.selectMock}>
              <span>Data Engineer (Senior)</span>
              <ChevronDown size={16} />
            </div>
          </div>
        </div>
      </Card>

      {/* Comparison Insights & Visuals */}
      <div className={styles.comparisonGrid}>
        <Card
          title="Diferencial Salarial Estimado"
          subtitle="Contraste de percentiles salariales calculados por el modelo"
          action={<BarChart size={18} color="var(--color-primary)" />}
        >
          <div className={styles.chartPlaceholder}>
            <span>[ Gráfico de Barras Comparativas Perfil A vs Perfil B ]</span>
            <p style={{ fontSize: "0.78rem" }}>
              Perfil A: ~$74,000 USD vs Perfil B: ~$68,500 USD (+8.0% Brecha)
            </p>
          </div>
        </Card>

        <Card
          title="Conclusiones del Modelo ML"
          subtitle="Factores explicativos generados a partir de los datos"
          action={<Sparkles size={18} color="var(--color-primary)" />}
        >
          <div className={styles.insightList}>
            <div className={styles.insightItem}>
              <div className={styles.insightBullet} />
              <p>
                <strong>Especialización ML:</strong> Mayor prima salarial en puestos que exigen PyTorch, MLOps e infraestructura cloud avanzada.
              </p>
            </div>
            <div className={styles.insightItem}>
              <div className={styles.insightBullet} />
              <p>
                <strong>Efecto Seniority:</strong> La brecha salarial se amplía un 14% adicional a partir del 5to año de experiencia.
              </p>
            </div>
            <div className={styles.insightItem}>
              <div className={styles.insightBullet} />
              <p>
                <strong>Flexibilidad Geográfica:</strong> Las ofertas 100% remotas reducen la diferencia regional en un 32%.
              </p>
            </div>
          </div>
        </Card>

        <Card
          title="Demanda de Habilidades Cruzadas"
          subtitle="Habilidades compartidas y exclusivas de cada rol"
          action={<Scale size={18} color="var(--color-primary)" />}
        >
          <div className={styles.chartPlaceholder}>
            <span>[ Visualización Radar / Diagrama de Venn de Habilidades ]</span>
            <p style={{ fontSize: "0.78rem" }}>
              Comunes: Python, SQL, Docker, Git • Exclusivas: MLflow vs Spark/Kafka
            </p>
          </div>
        </Card>

        <Card
          title="Evolución y Tendencia"
          subtitle="Comportamiento del salario medio en los últimos periodos"
        >
          <div className={styles.chartPlaceholder}>
            <span>[ Gráfico de Líneas de Tendencia Temporal ]</span>
            <p style={{ fontSize: "0.78rem" }}>
              Crecimiento promedio anual: +9.4% en roles de Data & AI
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}
