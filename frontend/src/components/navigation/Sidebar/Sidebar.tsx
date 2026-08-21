import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Calculator,
  BarChart3,
  ArrowLeftRight,
  Info,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import styles from "./Sidebar.module.css";

export interface NavigationItem {
  label: string;
  path: string;
  icon: LucideIcon;
}

const navigationItems: NavigationItem[] = [
  {
    label: "Inicio",
    path: "/",
    icon: LayoutDashboard,
  },
  {
    label: "Predicción Salarial",
    path: "/prediction",
    icon: Calculator,
  },
  {
    label: "Explorar Datos",
    path: "/explore",
    icon: BarChart3,
  },
  {
    label: "Comparaciones",
    path: "/comparisons",
    icon: ArrowLeftRight,
  },
  {
    label: "Acerca del Proyecto",
    path: "/about",
    icon: Info,
  },
];

interface SidebarProps {
  isOpen?: boolean;
  onClose?: () => void;
}

export function Sidebar({ isOpen = false, onClose }: SidebarProps) {
  return (
    <>
      {isOpen && (
        <div
          className={styles.backdrop}
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`${styles.sidebar} ${isOpen ? styles.sidebarOpen : ""}`}
        aria-label="Navegación principal"
      >
        {/* Brand Header */}
        <div className={styles.brand}>
          <div className={styles.brandIconWrapper}>
            <Sparkles className={styles.brandIcon} size={22} />
          </div>
          <div className={styles.brandInfo}>
            <span className={styles.brandTitle}>SalaryPredict</span>
            <span className={styles.brandSubtitle}>ML Intelligence</span>
          </div>
        </div>

        {/* Navigation Section */}
        <nav className={styles.nav}>
          <span className={styles.navCategory}>MENU PRINCIPAL</span>
          <ul className={styles.navList}>
            {navigationItems.map((item) => (
              <li key={item.path}>
                <NavLink
                  to={item.path}
                  end={item.path === "/"}
                  className={({ isActive }) =>
                    `${styles.navItem} ${isActive ? styles.navItemActive : ""}`
                  }
                  onClick={onClose}
                >
                  {({ isActive }) => (
                    <>
                      <item.icon
                        className={`${styles.navIcon} ${
                          isActive ? styles.navIconActive : ""
                        }`}
                        size={20}
                      />
                      <span className={styles.navLabel}>{item.label}</span>
                      {isActive && <div className={styles.activeIndicator} />}
                    </>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        {/* Footer / Meta info */}
        <div className={styles.footer}>
          <div className={styles.systemBadge}>
            <span className={styles.statusDot} />
            <span>MLOps Grupo 13</span>
          </div>
          <span className={styles.versionText}>v0.1.0 • Pipeline Activo</span>
        </div>
      </aside>
    </>
  );
}
