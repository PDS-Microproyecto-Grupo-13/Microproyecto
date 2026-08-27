import argparse
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for demo model registration."""
    parser = argparse.ArgumentParser(
        description="Registers a lightweight demo model into MLflow Model Registry for infrastructure testing."
    )
    parser.add_argument(
        "--model-name",
        "-m",
        default=os.getenv("MODEL_NAME", "salary_predict_model"),
        help="Target registered model name in MLflow (default: salary_predict_model).",
    )
    parser.add_argument(
        "--tracking-uri",
        "-u",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow Tracking Server URI (default: http://localhost:5000).",
    )
    parser.add_argument(
        "--version-tag",
        "-t",
        default="v1",
        choices=["v1", "v2"],
        help="Demo version configuration to train (v1 = LinearRegression, v2 = GradientBoosting).",
    )
    parser.add_argument(
        "--set-champion",
        action="store_true",
        default=True,
        help="Automatically assign the 'champion' alias to this newly registered version.",
    )
    return parser.parse_args()


def train_demo_model(version_tag: str):
    """Trains a simple toy regressor in memory on synthetic salary data."""
    # Features: [years_experience, is_remote, skills_count]
    np.random.seed(42 if version_tag == "v1" else 100)
    n_samples = 200

    years_exp = np.random.uniform(0.5, 15.0, size=n_samples)
    is_remote = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    skills_count = np.random.randint(1, 10, size=n_samples)

    # Base salary formula: 30k + 6k * years + 10k * is_remote + 2k * skills
    salary = (
        30000.0
        + (6200.0 if version_tag == "v1" else 6500.0) * years_exp
        + 8500.0 * is_remote
        + 2200.0 * skills_count
        + np.random.normal(0, 3000, size=n_samples)
    )

    df_features = pd.DataFrame(
        {
            "years_experience": years_exp,
            "is_remote": is_remote,
            "skills_count": skills_count,
        }
    )

    if version_tag == "v1":
        model = LinearRegression()
        algorithm_name = "LinearRegression"
    else:
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        algorithm_name = "GradientBoostingRegressor"

    model.fit(df_features, salary)
    r2_score = float(model.score(df_features, salary))

    return model, df_features, algorithm_name, r2_score


def main() -> None:
    args = parse_args()

    print("=================================================================")
    print(" Registering Demo Model for SalaryPredict MLOps Infrastructure")
    print(f" Model Name:    {args.model_name}")
    print(f" Tracking URI:  {args.tracking_uri}")
    print(f" Version Tag:   {args.version_tag}")
    print("=================================================================")

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    experiment_name = "salary_predict_demo"
    mlflow.set_experiment(experiment_name)

    model, df_sample, algorithm, r2 = train_demo_model(args.version_tag)

    with mlflow.start_run(run_name=f"demo_model_{args.version_tag}") as run:
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("version_tag", args.version_tag)
        mlflow.log_param("n_samples", len(df_sample))
        mlflow.log_metric("r2_score", r2)

        # Log and register model into Model Registry
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=args.model_name,
            input_example=df_sample.head(2),
        )

        print(f"✓ Model logged to run ID: {run.info.run_id}")
        print(f"✓ Model artifact URI:    {model_info.model_uri}")

    # Resolve the latest version number that was just registered
    registered_model = client.get_registered_model(args.model_name)
    latest_version = registered_model.latest_versions[-1].version

    print(f"✓ Registered Model Version: {latest_version}")

    if args.set_champion:
        client.set_registered_model_alias(
            name=args.model_name,
            alias="champion",
            version=latest_version,
        )
        print(f"✓ Alias 'champion' assigned to Version {latest_version}")

    print("\n--- Model Verification & Invocation Example ---")
    print("To query the model once inference server is running:")
    print("curl -X POST http://localhost:5001/invocations \\")
    print("  -H 'Content-Type: application/json' \\")
    print(
        '  -d \'{"dataframe_split": {"columns": ["years_experience", "is_remote", "skills_count"], "data": [[5.0, 1, 4]]}}\''
    )
    print("=================================================================\n")


if __name__ == "__main__":
    main()
