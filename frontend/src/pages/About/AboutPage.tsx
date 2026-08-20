import {
  Info,
  GitBranch,
  Layers,
  Users,
} from "lucide-react";
import { PageHeader } from "../../components/ui/PageHeader/PageHeader";
import { Card } from "../../components/ui/Card/Card";
import styles from "./AboutPage.module.css";

export function AboutPage() {
  return (
    <div className={styles.container}>
      <PageHeader
        title="Acerca del Proyecto"
        subtitle="Arquitectura integral del sistema SalaryPredict y ciclo de vida MLOps"
        badge="Documentación"
      />

      <div className={styles.grid}>
        {/* Project Purpose */}
        <Card
          title="Objetivo de SalaryPredict"
          subtitle="Estimación de compensaciones en el sector tecnológico"
          action={<Info size={18} color="var(--color-primary)" />}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-sm)", fontSize: "0.9rem", lineHeight: 1.6, color: "var(--color-text-secondary)" }}>
            <p>
              <strong>SalaryPredict</strong> es un sistema integral de Machine Learning enfocado en predecir y analizar salarios del mercado laboral tecnológico a partir de datos estructurados recolectados de ofertas de empleo reales.
            </p>
            <p>
              El proyecto implementa buenas prácticas de <strong>MLOps</strong> para garantizar la reproducibilidad de datos, el seguimiento riguroso de experimentos, el despliegue continuo de modelos y una interfaz moderna e interactiva.
            </p>
          </div>
        </Card>

        {/* Pipeline Architecture */}
        <Card
          title="Arquitectura del Pipeline MLOps"
          subtitle="Flujo de datos de extremo a extremo"
          action={<GitBranch size={18} color="var(--color-primary)" />}
        >
          <div className={styles.pipelineList}>
            <div className={styles.pipelineStep}>
              <div className={styles.stepNumber}>1</div>
              <div className={styles.stepContent}>
                <span className={styles.stepTitle}>Ingesta & Extracción</span>
                <span className={styles.stepDesc}>API Client automatizado y schemas validados (Pydantic).</span>
              </div>
            </div>

            <div className={styles.pipelineStep}>
              <div className={styles.stepNumber}>2</div>
              <div className={styles.stepContent}>
                <span className={styles.stepTitle}>Versionado de Datos</span>
                <span className={styles.stepDesc}>DVC (Data Version Control) para datasets crudos y procesados.</span>
              </div>
            </div>

            <div className={styles.pipelineStep}>
              <div className={styles.stepNumber}>3</div>
              <div className={styles.stepContent}>
                <span className={styles.stepTitle}>Experimentación & Tracking</span>
                <span className={styles.stepDesc}>MLflow para registro de métricas, parámetros y artefactos.</span>
              </div>
            </div>

            <div className={styles.pipelineStep}>
              <div className={styles.stepNumber}>4</div>
              <div className={styles.stepContent}>
                <span className={styles.stepTitle}>Servicio de Inferencia</span>
                <span className={styles.stepDesc}>FastAPI REST endpoints con validación y baja latencia.</span>
              </div>
            </div>

            <div className={styles.pipelineStep}>
              <div className={styles.stepNumber}>5</div>
              <div className={styles.stepContent}>
                <span className={styles.stepTitle}>Frontend SPA</span>
                <span className={styles.stepDesc}>React + TypeScript + Vite con navegación modular.</span>
              </div>
            </div>
          </div>
        </Card>

        {/* Tech Stack */}
        <Card
          title="Stack Tecnológico"
          subtitle="Componentes del ecosistema de desarrollo"
          action={<Layers size={18} color="var(--color-primary)" />}
        >
          <div className={styles.techGrid}>
            <div className={styles.techBox}>
              <span className={styles.techBoxName}>React 19</span>
              <span className={styles.techBoxCategory}>Frontend</span>
            </div>
            <div className={styles.techBox}>
              <span className={styles.techBoxName}>TypeScript</span>
              <span className={styles.techBoxCategory}>Type Safety</span>
            </div>
            <div className={styles.techBox}>
              <span className={styles.techBoxName}>Vite</span>
              <span className={styles.techBoxCategory}>Bundler</span>
            </div>
            <div className={styles.techBox}>
              <span className={styles.techBoxName}>React Router</span>
              <span className={styles.techBoxCategory}>Routing SPA</span>
            </div>
            <div className={styles.techBox}>
              <span className={styles.techBoxName}>Python 3.11</span>
              <span className={styles.techBoxCategory}>Backend / ML</span>
            </div>
            <div className={styles.techBox}>
              <span className={styles.techBoxName}>MLflow & DVC</span>
              <span className={styles.techBoxCategory}>MLOps</span>
            </div>
          </div>
        </Card>

        {/* Project Team */}
        <Card
          title="Equipo de Trabajo"
          subtitle="Microproyecto de MLOps"
          action={<Users size={18} color="var(--color-primary)" />}
        >
          <div className={styles.teamList}>
            <div className={styles.teamItem}>
              <strong>Grupo 13</strong>
              <span className={styles.teamRole}>Equipo MLOps</span>
            </div>
            <div className={styles.teamItem}>
              <span>Especialidad</span>
              <span className={styles.teamRole}>Machine Learning Engineering</span>
            </div>
            <div className={styles.teamItem}>
              <span>Versión Base</span>
              <span className={styles.teamRole}>v0.1.0-alpha</span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
