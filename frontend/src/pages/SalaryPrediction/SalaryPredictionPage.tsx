import { useState } from "react";
import type { FormEvent } from "react";
import { Info, LoaderCircle, Sparkles, TrendingUp } from "lucide-react";
import { PageHeader } from "../../components/ui/PageHeader/PageHeader";
import { Card } from "../../components/ui/Card/Card";
import { predictSalary, type SalaryPredictionResponse } from "../../services/salaryApi";
import styles from "./SalaryPredictionPage.module.css";

const currency = new Intl.NumberFormat("es-CO", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export function SalaryPredictionPage() {
  const [result, setResult] = useState<SalaryPredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    const data = new FormData(event.currentTarget);
    try {
      const technologies = String(data.get("technologies") ?? "").split(",").map((value) => value.trim()).filter(Boolean);
      const years = String(data.get("experience_years") ?? "").trim();
      setResult(await predictSalary({
        title: String(data.get("title")), experience_level: String(data.get("experience_level")),
        experience_years: years ? Number(years) : null, country: String(data.get("country")),
        is_remote: data.get("is_remote") === "on", company: String(data.get("company") ?? "") || null,
        company_is_agency: data.get("company_is_agency") === "on", technologies,
        topics: ["Data Science", "Machine Learning"],
      }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "No fue posible ejecutar la predicción.");
    } finally { setLoading(false); }
  }

  return (
    <div className={styles.container}>
      <PageHeader title="Predicción Salarial" subtitle="Estima un rango anual en USD con el modelo CatBoost desplegado en MLflow" badge="Inferencia ML" />
      <div className={styles.grid}>
        <Card title="Parámetros del perfil" subtitle="La solicitud se procesa mediante la API del proyecto">
          <form className={styles.form} onSubmit={submit}>
            <label className={styles.formGroup}><span className={styles.label}>Rol profesional</span><input name="title" defaultValue="Data Scientist" required minLength={2} /></label>
            <div className={styles.twoColumns}>
              <label className={styles.formGroup}><span className={styles.label}>Nivel de experiencia</span><select name="experience_level" defaultValue="SE"><option value="EN">Entrada</option><option value="MI">Intermedio</option><option value="SE">Senior</option><option value="EX">Ejecutivo</option></select></label>
              <label className={styles.formGroup}><span className={styles.label}>Años de experiencia</span><input name="experience_years" type="number" min="0" max="50" step="0.5" defaultValue="5" /></label>
            </div>
            <label className={styles.formGroup}><span className={styles.label}>País</span><input name="country" defaultValue="Colombia" required /></label>
            <label className={styles.formGroup}><span className={styles.label}>Empresa (opcional)</span><input name="company" placeholder="Ej. Acme Analytics" /></label>
            <label className={styles.formGroup}><span className={styles.label}>Tecnologías, separadas por coma</span><input name="technologies" defaultValue="Python, SQL, AWS, Docker" /></label>
            <div className={styles.checks}><label><input name="is_remote" type="checkbox" defaultChecked /> Trabajo remoto</label><label><input name="company_is_agency" type="checkbox" /> Publicado por agencia</label></div>
            <button className={styles.submitButton} type="submit" disabled={loading}>{loading ? <LoaderCircle className={styles.spinner} size={18} /> : <Sparkles size={18} />}{loading ? "Calculando..." : "Ejecutar predicción"}</button>
            {error && <p className={styles.error} role="alert">{error}</p>}
          </form>
        </Card>
        <div className={styles.resultsColumn}>
          <Card title="Estimación del modelo" subtitle={result ? "Predicción recibida desde MLflow" : "Complete el perfil para calcular el rango"} action={<TrendingUp size={18} color="var(--color-success)" />}>
            <div className={styles.resultBox} aria-live="polite"><span className={styles.resultLabel}>Punto medio estimado</span><span className={styles.resultAmount}>{result ? currency.format(result.prediction.midpoint_usd) : "—"}</span><span className={styles.resultRange}>{result ? `${currency.format(result.prediction.minimum_usd)} – ${currency.format(result.prediction.maximum_usd)}` : "El resultado aparecerá aquí"}</span></div>
            {result?.warnings.map((warning) => <p className={styles.warning} key={warning}>{warning}</p>)}
            <p className={styles.disclaimer}>Estimación académica basada en vacantes publicadas; no constituye una oferta ni asesoría laboral.</p>
          </Card>
          <Card title="Modelo desplegado" subtitle="Selector estable para facilitar nuevas versiones" action={<Info size={18} color="var(--color-text-secondary)" />}>
            <dl className={styles.metadata}><div><dt>Familia</dt><dd>CatBoost, dos salidas</dd></div><div><dt>Modelo</dt><dd>{result?.model.name ?? "salary_predict_model"}</dd></div><div><dt>Alias</dt><dd>{result?.model.alias ?? "champion"}</dd></div><div><dt>Respuesta</dt><dd>Mínimo, máximo y punto medio</dd></div></dl>
          </Card>
        </div>
      </div>
    </div>
  );
}
