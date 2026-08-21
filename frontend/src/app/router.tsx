import { createBrowserRouter } from "react-router-dom";
import { DashboardLayout } from "../layouts/DashboardLayout/DashboardLayout";
import { HomePage } from "../pages/Home/HomePage";
import { SalaryPredictionPage } from "../pages/SalaryPrediction/SalaryPredictionPage";
import { ExploreDataPage } from "../pages/ExploreData/ExploreDataPage";
import { ComparisonsPage } from "../pages/Comparisons/ComparisonsPage";
import { AboutPage } from "../pages/About/AboutPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <DashboardLayout />,
    children: [
      {
        index: true,
        element: <HomePage />,
      },
      {
        path: "prediction",
        element: <SalaryPredictionPage />,
      },
      {
        path: "explore",
        element: <ExploreDataPage />,
      },
      {
        path: "comparisons",
        element: <ComparisonsPage />,
      },
      {
        path: "about",
        element: <AboutPage />,
      },
    ],
  },
]);
