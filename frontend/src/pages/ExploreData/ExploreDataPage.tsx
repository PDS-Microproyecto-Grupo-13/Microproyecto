import {
  Search,
  ChevronDown,
  BarChart2,
  PieChart,
  MapPin,
  Table as TableIcon,
} from "lucide-react";
import { PageHeader } from "../../components/ui/PageHeader/PageHeader";
import { Card } from "../../components/ui/Card/Card";
import styles from "./ExploreDataPage.module.css";

export function ExploreDataPage() {
  return (
    <div className={styles.container}>
      <PageHeader
        title="Explorar Datos"
        subtitle="Analiza el mercado laboral con visualizaciones interactivas y filtros de segmentación"
        badge="Dataset EDA"
      />

      {/* Filters Placeholder Bar */}
      <section className={styles.filtersBar} aria-label="Filtros del dataset">
        <div className={styles.filterInput}>
          <Search size={16} />
          <span>Buscar por tecnología o palabra clave...</span>
        </div>

        <div className={styles.filterSelect}>
          <span>Todos los roles</span>
          <ChevronDown size={16} />
        </div>

        <div className={styles.filterSelect}>
          <span>Todos los seniorities</span>
          <ChevronDown size={16} />
        </div>

        <div className={styles.filterSelect}>
          <span>Todas las ubicaciones</span>
          <ChevronDown size={16} />
        </div>
      </section>

      {/* Exploration Cards Grid */}
      <section className={styles.cardsGrid}>
        <Card
          title="Distribución Salarial por Nivel de Experiencia"
          subtitle="Rangos salariales mínimos, medianos y máximos por seniority"
          action={<BarChart2 size={18} color="var(--color-primary)" />}
        >
          <div className={styles.visualPlaceholder}>
            <span>[ Gráfica Boxplot / Barras de Distribución por Seniority ]</span>
            <p style={{ fontSize: "0.78rem" }}>
              Junior ($25k-$38k) • Mid-Level ($40k-$62k) • Senior ($65k-$98k) • Lead ($90k-$130k)
            </p>
          </div>
        </Card>

        <Card
          title="Salarios por Modalidad de Trabajo"
          subtitle="Proporción y remuneración en Remoto vs Híbrido vs Presencial"
          action={<PieChart size={18} color="var(--color-primary)" />}
        >
          <div className={styles.visualPlaceholder}>
            <span>[ Gráfica de Donut / Distribución de Modalidades ]</span>
            <p style={{ fontSize: "0.78rem" }}>
              62% Remoto • 28% Híbrido • 10% Presencial
            </p>
          </div>
        </Card>

        <Card
          title="Ubicación Geográfica y Mercados"
          subtitle="Comparativa salarial promedio según país / región de la oferta"
          action={<MapPin size={18} color="var(--color-primary)" />}
        >
          <div className={styles.visualPlaceholder}>
            <span>[ Mapa de Calor / Visualización Geográfica por País ]</span>
            <p style={{ fontSize: "0.78rem" }}>
              LATAM, Norteamérica, Europa y ofertas globales
            </p>
          </div>
        </Card>

        <Card
          title="Muestra de Datos Curados"
          subtitle="Vista preliminar tabular de los datos ingeridos"
          action={<TableIcon size={18} color="var(--color-primary)" />}
        >
          <div className={styles.tableContainer}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.th}>Título del Puesto</th>
                  <th className={styles.th}>Seniority</th>
                  <th className={styles.th}>Modalidad</th>
                  <th className={styles.th}>Salario USD</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className={styles.td}>ML Engineer</td>
                  <td className={styles.td}>Senior</td>
                  <td className={styles.td}>Remoto</td>
                  <td className={styles.td}>$85,000</td>
                </tr>
                <tr>
                  <td className={styles.td}>Backend Python Dev</td>
                  <td className={styles.td}>Mid</td>
                  <td className={styles.td}>Híbrido</td>
                  <td className={styles.td}>$52,000</td>
                </tr>
                <tr>
                  <td className={styles.td}>Data Analyst</td>
                  <td className={styles.td}>Junior</td>
                  <td className={styles.td}>Presencial</td>
                  <td className={styles.td}>$32,000</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Card>
      </section>
    </div>
  );
}
