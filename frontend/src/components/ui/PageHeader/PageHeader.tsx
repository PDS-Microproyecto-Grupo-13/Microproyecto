import type { ReactNode } from "react";
import styles from "./PageHeader.module.css";

export interface PageHeaderProps {
  title: string;
  subtitle?: string;
  badge?: string;
  actions?: ReactNode;
}

export function PageHeader({
  title,
  subtitle,
  badge,
  actions,
}: PageHeaderProps) {
  return (
    <header className={styles.container}>
      <div className={styles.titleArea}>
        <div className={styles.headingWrapper}>
          <h1 className={styles.title}>{title}</h1>
          {badge && <span className={styles.badge}>{badge}</span>}
        </div>
        {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
      </div>

      {actions && <div className={styles.actions}>{actions}</div>}
    </header>
  );
}
