import { useLocation } from "react-router-dom";
import { Menu, Bell, User, Activity } from "lucide-react";
import styles from "./Topbar.module.css";

interface TopbarProps {
  onMenuToggle?: () => void;
}

const pageTitles: Record<string, string> = {
  "/": "Dashboard General",
  "/prediction": "Módulo de Predicción Salarial",
  "/explore": "Exploración de Mercado",
  "/comparisons": "Comparativas y Benchmarks",
  "/about": "Documentación y Arquitectura",
};

export function Topbar({ onMenuToggle }: TopbarProps) {
  const location = useLocation();
  const currentTitle = pageTitles[location.pathname] ?? "SalaryPredict";

  return (
    <header className={styles.topbar}>
      <div className={styles.leftSection}>
        {onMenuToggle && (
          <button
            type="button"
            className={styles.menuButton}
            onClick={onMenuToggle}
            aria-label="Abrir menú de navegación"
          >
            <Menu size={20} />
          </button>
        )}
        <div className={styles.titleWrapper}>
          <span className={styles.title}>{currentTitle}</span>
          <span className={styles.badge}>MLOps Platform</span>
        </div>
      </div>

      <div className={styles.rightSection}>
        {/* API / Pipeline Status Placeholder */}
        <div className={styles.statusIndicator} title="Estado del servicio de ML">
          <Activity size={16} className={styles.statusIcon} />
          <span className={styles.statusText}>API En Línea</span>
        </div>

        {/* Notifications Placeholder */}
        <button
          type="button"
          className={styles.iconButton}
          aria-label="Notificaciones"
          title="Sin notificaciones pendientes"
        >
          <Bell size={18} />
          <span className={styles.notificationDot} />
        </button>

        <div className={styles.divider} />

        {/* User Profile Slot Placeholder */}
        <div className={styles.profileSection}>
          <div className={styles.avatar}>
            <User size={18} />
          </div>
          <div className={styles.profileInfo}>
            <span className={styles.profileName}>Equipo MLOps</span>
            <span className={styles.profileRole}>Grupo 13</span>
          </div>
        </div>
      </div>
    </header>
  );
}
