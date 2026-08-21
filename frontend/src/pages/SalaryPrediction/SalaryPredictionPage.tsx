import {
  Sparkles,
  ChevronDown,
  Sliders,
  TrendingUp,
  Info,
} from "lucide-react";
import { PageHeader } from "../../components/ui/PageHeader/PageHeader";
import { Card } from "../../components/ui/Card/Card";
import styles from "./SalaryPredictionPage.module.css";

export function SalaryPredictionPage() {
  return (
    <div className={styles.container}>
      <PageHeader
        title="Predicción Salarial"
        subtitle="Calcula el salario esperado mediante inferencia sobre el modelo de Machine Learning"
        badge="Inferencia ML"
      />

      <div className={styles.grid}>
        {/* Input Parameters Form (Visual Placeholder) */}
        <Card
          title="Parámetros del Perfil"
          subtitle="Define las variables para ejecutar la estimación salarial"
          action={<Sliders size={18} color="var(--color-primary)" />}
        >
          <div className={styles.formPlaceholder}>
            <div className={styles.formGroup}>
              <label className={styles.label}>Rol Profesional</label>
              <div className={styles.inputMock}>
                <span>Ej. Data Scientist / ML Engineer</span>
                <ChevronDown size={16} />
              </div>
            </div>

            <div className={styles.formGroup}>
              <label className={styles.label}>Nivel de Experiencia (Seniority)</label>
              <div className={styles.inputMock}>
                <span>Senior (5+ años)</span>
                <ChevronDown size={16} />
              </div>
            </div>

            <div className={styles.formGroup}>
              <label className={styles.label}>Ubicación / Región</label>
              <div className={styles.inputMock}>
                <span>Remoto - LATAM / USA</span>
                <ChevronDown size={16} />
              </div>
            </div>

            <div className={styles.formGroup}>
              <label className={styles.label}>Modalidad de Trabajo</label>
              <div className={styles.inputMock}>
                <span>100% Remoto</span>
                <ChevronDown size={16} />
              </div>
            </div>

            <div className={styles.formGroup}>
              <label className={styles.label}>Tecnologías y Stack Principal</label>
              <div className={styles.inputMock}>
                <span>Python, Docker, FastAPI, SQL, AWS</span>
              </div>
            </div>

            <div className={styles.buttonMock}>
              <Sparkles size={18} />
              <span>Ejecutar Predicción (Placeholder)</span>
            </div>
          </div>
        </Card>

        {/* Prediction Results Preview (Visual Placeholder) */}
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-xl)" }}>
          <Card
            title="Estimación del Modelo"
            subtitle="Resultado generado a partir de variables ingresadas"
            action={<TrendingUp size={18} color="var(--color-success)" />}
          >
            <div className={styles.resultBox}>
              <span className={styles.resultLabel}>Salario Anual Estimado</span>
              <span className={styles.resultAmount}>$68,400 USD</span>
              <span className={styles.resultRange}>
                Rango estimado: $59,000 — $78,000 USD
              </span>
            </div>

            <h4
              style={{
                fontSize: "0.9rem",
                fontWeight: 600,
                marginBottom: "var(--space-sm)",
                color: "var(--color-text-primary)",
              }}
            >
              Factores de Mayor Impacto (Feature Importance)
            </h4>

            <div className={styles.factorsList}>
              <div className={styles.factorItem}>
                <span>Seniority (Años exp.)</span>
                <div className={styles.factorBarWrapper}>
                  <div className={styles.factorBarFill} style={{ width: "85%" }} />
                </div>
                <strong>+38%</strong>
              </div>

              <div className={styles.factorItem}>
                <span>Modalidad Remota Intl.</span>
                <div className={styles.factorBarWrapper}>
                  <div className={styles.factorBarFill} style={{ width: "65%" }} />
                </div>
                <strong>+24%</strong>
              </div>

              <div className={styles.factorItem}>
                <span>Stack Cloud (AWS/GCP)</span>
                <div className={styles.factorBarWrapper}>
                  <div className={styles.factorBarFill} style={{ width: "45%" }} />
                </div>
                <strong>+16%</strong>
              </div>
            </div>
          </Card>

          <Card
            title="Información del Modelo"
            subtitle="Metadatos de la última versión desplegada"
            action={<Info size={18} color="var(--color-text-secondary)" />}
          >
            <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-xs)", fontSize: "0.85rem" }}>
              <p><strong>Algoritmo:</strong> GradientBoostingRegressor</p>
              <p><strong>Dataset Base:</strong> Foorilla Ingestion (Aug 2026)</p>
              <p><strong>Métrica MAE:</strong> ± $4,250 USD</p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
