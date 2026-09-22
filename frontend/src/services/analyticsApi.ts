import { requestJson } from "./http";

export interface AnalyticsCounts {
  raw_snapshot_rows: number;
  validated_rows: number;
  modelable_rows: number;
}

export interface AnalyticsDataRange {
  published_min: string;
  published_max: string;
}

export interface SalaryMidpoint {
  currency: "USD";
  period: "annual";
  mean_usd: number;
  median_usd: number;
  minimum_usd: number;
  maximum_usd: number;
}

export interface SalaryBin {
  lower_bound_usd: number;
  upper_bound_usd: number | null;
  count: number;
  proportion: number;
}

export interface CategoryCount {
  category: string;
  count: number;
  proportion: number;
}

export interface TechnologyCount {
  technology: string;
  count: number;
  proportion: number;
}

export interface DatasetAnalytics {
  source: string;
  target_scope: string;
  counts: AnalyticsCounts;
  data_range: AnalyticsDataRange;
  salary_midpoint: SalaryMidpoint;
  salary_midpoint_distribution: SalaryBin[];
  seniority_distribution: CategoryCount[];
  work_mode_distribution: CategoryCount[];
  top_technologies: TechnologyCount[];
}

export interface ModelEvaluationMetrics {
  algorithm: string;
  evaluation_rows: number;
  mae_average_usd: number;
  r2_salary_min: number;
  r2_salary_max: number;
  predicted_range_coverage: number;
  uncertainty_margin_usd: number;
  uncertainty_nominal_coverage: number;
  uncertainty_test_coverage: number;
}

export interface AnalyticsMetadata {
  dataset_fingerprint: string;
  snapshot_count: number;
  git_commit: string | null;
  dvc_yaml_hash: string | null;
  params_hash: string | null;
}

export interface AnalyticsSummary {
  schema_version: "1.0";
  dataset: DatasetAnalytics;
  model: ModelEvaluationMetrics;
  metadata: AnalyticsMetadata;
}

interface GetAnalyticsSummaryOptions {
  signal?: AbortSignal;
}

export function getAnalyticsSummary(
  options: GetAnalyticsSummaryOptions = {},
): Promise<AnalyticsSummary> {
  return requestJson<AnalyticsSummary>("/api/v1/analytics/summary", {
    method: "GET",
    headers: { Accept: "application/json" },
    signal: options.signal,
  });
}
