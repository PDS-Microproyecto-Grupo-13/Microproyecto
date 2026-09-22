from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from ml_pipeline.analytics import analytics
from ml_pipeline.common.logging import configure_logging
from ml_pipeline.data.collect import collect
from ml_pipeline.data.preprocess import preprocess
from ml_pipeline.data.validate import validate
from ml_pipeline.modeling.evaluate import evaluate
from ml_pipeline.modeling.qualify import qualify
from ml_pipeline.modeling.train import train
from ml_pipeline.publish_analytics import PublishAnalyticsError, publish_analytics
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.mlflow_tracker import track
from ml_pipeline.tracking.registry import register_candidate

LOGGER = logging.getLogger(__name__)
COMMANDS: dict[str, Callable[[Settings], object]] = {
    "collect": collect,
    "validate": validate,
    "preprocess": preprocess,
    "qualify": qualify,
    "train": train,
    "evaluate": evaluate,
    "analytics": analytics,
    "track": track,
    "register-candidate": register_candidate,
    "publish-analytics": publish_analytics,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml_pipeline",
        description="Reproducible model candidate pipeline",
    )
    parser.add_argument("command", choices=COMMANDS, help="Pipeline stage to execute")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        COMMANDS[args.command](Settings.load())
    except PublishAnalyticsError as error:
        LOGGER.error("%s | result=failed | error=%s", args.command, error)
        return error.exit_code
    except Exception as error:
        LOGGER.error("%s | result=failed | error=%s", args.command, error)
        return 1
    return 0
