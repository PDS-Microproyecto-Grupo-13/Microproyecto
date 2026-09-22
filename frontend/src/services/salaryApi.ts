import { requestJson } from "./http";

export interface SalaryPredictionRequest {
  title: string; experience_level: string; experience_years: number | null;
  country: string; is_remote: boolean; company: string | null;
  company_is_agency: boolean; technologies: string[]; topics: string[];
}
export interface SalaryPredictionResponse {
  prediction: { minimum_usd: number; maximum_usd: number; midpoint_usd: number };
  model: { name: string; alias: string; version?: string | null }; warnings: string[];
}
export async function predictSalary(request: SalaryPredictionRequest): Promise<SalaryPredictionResponse> {
  return requestJson<SalaryPredictionResponse>("/api/v1/predictions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(request),
  });
}
