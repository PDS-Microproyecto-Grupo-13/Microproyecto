import { useState } from "react";
import { Outlet } from "react-router-dom";
import { Sidebar } from "../../components/navigation/Sidebar/Sidebar";
import { Topbar } from "../../components/navigation/Topbar/Topbar";
import styles from "./DashboardLayout.module.css";

export function DashboardLayout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const handleMenuToggle = () => {
    setIsSidebarOpen((prev) => !prev);
  };

  const handleSidebarClose = () => {
    setIsSidebarOpen(false);
  };

  return (
    <div className={styles.layout}>
      <Sidebar isOpen={isSidebarOpen} onClose={handleSidebarClose} />
      <div className={styles.mainWrapper}>
        <Topbar onMenuToggle={handleMenuToggle} />
        <main className={styles.content}>
          <div className={styles.contentContainer}>
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
