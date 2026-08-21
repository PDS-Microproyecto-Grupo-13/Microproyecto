import type { ReactNode } from "react";
import styles from "./Card.module.css";

export interface CardProps {
  title?: string;
  subtitle?: string;
  action?: ReactNode;
  footer?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function Card({
  title,
  subtitle,
  action,
  footer,
  className,
  children,
}: CardProps) {
  const hasHeader = title || subtitle || action;

  return (
    <section className={`${styles.card} ${className ?? ""}`}>
      {hasHeader && (
        <header className={styles.header}>
          <div className={styles.headerText}>
            {title && <h3 className={styles.title}>{title}</h3>}
            {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
          </div>
          {action && <div className={styles.action}>{action}</div>}
        </header>
      )}

      <div className={styles.content}>{children}</div>

      {footer && <footer className={styles.footer}>{footer}</footer>}
    </section>
  );
}
