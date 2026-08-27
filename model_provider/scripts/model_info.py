import argparse
import datetime
import os
import sys

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for model inspection."""
    parser = argparse.ArgumentParser(
        description="Displays operational metadata and version resolution for a registered MLflow model."
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Name of the registered model.",
    )
    parser.add_argument(
        "--alias",
        "-a",
        default=None,
        help="Optional alias to inspect and resolve (e.g. 'champion').",
    )
    parser.add_argument(
        "--version",
        "-v",
        default=None,
        help="Optional concrete version number to inspect.",
    )
    parser.add_argument(
        "--tracking-uri",
        "-u",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow Tracking Server URI.",
    )
    return parser.parse_args()


def format_timestamp(ts_millis: int | None) -> str:
    """Formats epoch milliseconds into readable UTC ISO format."""
    if ts_millis is None:
        return "N/A"
    return datetime.datetime.fromtimestamp(ts_millis / 1000.0, tz=datetime.UTC).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def inspect_model(
    client: MlflowClient,
    model_name: str,
    alias: str | None = None,
    version: str | None = None,
) -> None:
    """Retrieves and prints detailed model registration and version details."""
    try:
        registered_model = client.get_registered_model(name=model_name)
    except MlflowException as exc:
        print(f"[ERROR] Registered model '{model_name}' does not exist: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 65)
    print(f" Registered Model: {registered_model.name}")
    print(f" Description:      {registered_model.description or 'No description'}")
    print(f" Created:          {format_timestamp(registered_model.creation_timestamp)}")
    print(f" Last Updated:     {format_timestamp(registered_model.last_updated_timestamp)}")
    print(f" Aliases:          {registered_model.aliases or '{}'}")
    print("=" * 65)

    target_version = None

    if version is not None:
        try:
            target_version = client.get_model_version(name=model_name, version=str(version))
            print(f"\n[Version Details - Version {version}]")
        except MlflowException as exc:
            print(
                f"[ERROR] Version '{version}' not found for '{model_name}': {exc}", file=sys.stderr
            )
            sys.exit(1)
    elif alias is not None:
        try:
            target_version = client.get_model_version_by_alias(name=model_name, alias=alias)
            print(f"\n[Alias Resolution - Alias '{alias}' -> Version {target_version.version}]")
        except MlflowException as exc:
            print(
                f"[ERROR] Alias '{alias}' is not configured for model '{model_name}': {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # Show all latest versions / aliases
        print("\nRegistered Model Versions:")
        for mv in registered_model.latest_versions:
            aliases_for_v = [
                k for k, v in registered_model.aliases.items() if str(v) == str(mv.version)
            ]
            alias_badge = f" [Aliases: {', '.join(aliases_for_v)}]" if aliases_for_v else ""
            print(f"  • Version {mv.version} (Status: {mv.status}){alias_badge}")
            print(f"    - Run ID:   {mv.run_id}")
            print(f"    - Source:   {mv.source}")
            print(f"    - Created:  {format_timestamp(mv.creation_timestamp)}")
        return

    if target_version:
        print(f"  • Version:     {target_version.version}")
        print(f"  • Status:      {target_version.status}")
        print(f"  • Run ID:      {target_version.run_id}")
        print(f"  • Source URI:  {target_version.source}")
        print(f"  • Created:     {format_timestamp(target_version.creation_timestamp)}")
        if target_version.tags:
            print(f"  • Tags:        {target_version.tags}")


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    inspect_model(
        client=client,
        model_name=args.model,
        alias=args.alias,
        version=args.version,
    )


if __name__ == "__main__":
    main()
