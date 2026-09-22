from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from app.core.exceptions import ApplicationError
from app.schemas.analytics import (
    AnalyticsPublishResponse,
    AnalyticsStorageEnvelope,
    AnalyticsSummary,
)

_STORAGE_LOCK = threading.Lock()


def canonicalize_summary(summary: AnalyticsSummary) -> bytes:
    """Return the backend's canonical UTF-8 JSON representation for hashing."""
    return json.dumps(
        summary.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def calculate_artifact_hash(summary: AnalyticsSummary) -> str:
    return hashlib.sha256(canonicalize_summary(summary)).hexdigest()


class AnalyticsStorageService:
    """Persist and retrieve the backend-owned current analytics snapshot."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path

    def publish(self, summary: AnalyticsSummary) -> AnalyticsPublishResponse:
        artifact_hash = calculate_artifact_hash(summary)

        with _STORAGE_LOCK:
            existing = self._read_envelope_if_present()
            if existing is not None and existing.artifact_hash == artifact_hash:
                return AnalyticsPublishResponse(
                    artifact_hash=artifact_hash,
                    schema_version=summary.schema_version,
                    status="unchanged",
                    stored_at=existing.stored_at,
                )

            publication_status: Literal["created", "replaced"] = (
                "created" if existing is None else "replaced"
            )
            stored_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            envelope = AnalyticsStorageEnvelope(
                artifact_hash=artifact_hash,
                storage_version=1,
                stored_at=stored_at,
                summary=summary,
            )
            self._write_atomically(envelope)

        return AnalyticsPublishResponse(
            artifact_hash=artifact_hash,
            schema_version=summary.schema_version,
            status=publication_status,
            stored_at=stored_at,
        )

    def get_summary(self) -> AnalyticsSummary:
        with _STORAGE_LOCK:
            envelope = self._read_envelope_if_present()
        if envelope is None:
            raise ApplicationError(
                message="Analytics have not been published",
                status_code=404,
                error_code="analytics_not_published",
                headers={"Cache-Control": "no-store"},
            )
        return envelope.summary

    def _read_envelope_if_present(self) -> AnalyticsStorageEnvelope | None:
        try:
            raw = self.storage_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as error:
            raise self._storage_unavailable() from error

        try:
            payload = json.loads(raw)
            envelope = AnalyticsStorageEnvelope.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as error:
            raise self._storage_unavailable() from error

        if calculate_artifact_hash(envelope.summary) != envelope.artifact_hash:
            raise self._storage_unavailable()
        return envelope

    def _write_atomically(self, envelope: AnalyticsStorageEnvelope) -> None:
        parent = self.storage_path.parent
        temporary_path: Path | None = None
        serialized = (
            json.dumps(
                envelope.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

        try:
            parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=parent,
                prefix=f".{self.storage_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(serialized)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.storage_path)
            temporary_path = None
        except OSError as error:
            raise self._storage_unavailable() from error
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _storage_unavailable() -> ApplicationError:
        return ApplicationError(
            message="Analytics storage is unavailable",
            status_code=503,
            error_code="analytics_storage_unavailable",
            headers={"Cache-Control": "no-store"},
        )
