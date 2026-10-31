"""Point-in-time snapshot storage for optional model enrichments.

The catalog deliberately treats a manifest as the publication marker.  Data and
manifest files are first written to temporary files in the destination directory;
the manifest is moved into place last, so an interrupted write is never returned
by :meth:`SnapshotCatalog.resolve`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pandas as pd


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MANIFEST_SUFFIX = ".manifest.json"
_SECRET_QUERY_KEYS = {"apikey", "api_key", "api-token", "api_token", "token"}
_SECRET_QUERY_PATTERN = re.compile(
    r"(?i)(?:^|[_-])(?:api[_-]?)?(?:key|token|secret|signature|sig|password|passwd|auth)(?:$|[_-])"
)
_RIGHTS_CLASSES = {"local_use_only", "provider_licensed", "redistributable"}


def _utc_datetime(value: datetime | str) -> datetime:
    """Parse an instant and return a timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}") from exc

    # Naive values are interpreted as UTC rather than the host's local timezone.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_component(value: str, *, field: str) -> str:
    component = str(value).strip()
    if not _SAFE_COMPONENT.fullmatch(component):
        raise ValueError(
            f"{field} must contain only letters, numbers, '.', '_' or '-': {value!r}"
        )
    return component


def _safe_source_url(value: str) -> str:
    source = str(value).strip()
    if not source:
        raise ValueError("source_url must not be empty")
    parsed = urlsplit(source)
    if parsed.username or parsed.password:
        raise ValueError("source_url must not contain credentials")
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query_keys = {name.casefold() for name, _ in query_pairs}
    if query_keys & _SECRET_QUERY_KEYS:
        raise ValueError("source_url must not contain credentials")
    for name, value in query_pairs:
        normalized_name = re.sub(r"[^a-z0-9_-]+", "_", name.casefold())
        if _SECRET_QUERY_PATTERN.search(normalized_name):
            raise ValueError("source_url must not contain credentials")
        if str(value).strip().lower().startswith("bearer "):
            raise ValueError("source_url must not contain credentials")
    return source


def _validate_extra_manifest_urls(payload: Any, *, path: str = "extra_manifest") -> None:
    if isinstance(payload, str):
        parsed = urlsplit(payload.strip())
        if parsed.scheme and parsed.netloc:
            _safe_source_url(payload)
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_path = f"{path}.{key}"
            if "url" in str(key).casefold():
                if isinstance(value, str):
                    _safe_source_url(value)
                    continue
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        if isinstance(item, str):
                            _safe_source_url(item)
                        else:
                            raise ValueError(
                                f"{key_path}[{index}] must contain only URL strings"
                            )
                    continue
            _validate_extra_manifest_urls(value, path=key_path)
        return
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            _validate_extra_manifest_urls(item, path=f"{path}[{index}]")


def _manifest_envelope_sha256(manifest: dict[str, Any]) -> str:
    protected = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_envelope_sha256", "manifest_integrity"}
    }
    canonical = json.dumps(
        protected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_temp(directory: Path, data: bytes) -> Path:
    """Durably write bytes to an unpublished file in ``directory``."""
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".snapshot-",
        suffix=".tmp",
        dir=directory,
        delete=False,
    )
    path = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


class SnapshotCatalog:
    """Publish and resolve immutable, point-in-time CSV snapshots.

    Snapshots are stored under ``<root>/<season>/<topic>/<provider>``.  Resolution
    is based on the manifest timestamp, never filesystem modification times.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def manifest_path(snapshot_path: str | Path) -> Path:
        path = Path(snapshot_path)
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Snapshot must be a CSV file: {path}")
        return path.with_name(f"{path.stem}{_MANIFEST_SUFFIX}")

    def publish_frame(
        self,
        frame: pd.DataFrame,
        *,
        season_name: str,
        topic: str,
        provider: str,
        captured_at: datetime | str,
        source_url: str,
        rights_class: str,
        paid: bool = False,
        schema_version: str = "1",
        extra_manifest: dict[str, Any] | None = None,
    ) -> Path:
        """Publish ``frame`` and its provenance manifest.

        The returned path is the CSV artifact.  Its SHA-256 digest is calculated
        from the exact bytes written to disk, and is checked again on every load.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if len(frame.columns) == 0:
            raise ValueError("Snapshot must contain at least one column")
        if frame.columns.has_duplicates:
            raise ValueError("Snapshot columns must be unique")

        season = _safe_component(season_name, field="season_name")
        topic_name = _safe_component(topic, field="topic")
        provider_name = _safe_component(provider, field="provider")
        captured = _utc_datetime(captured_at)
        rights = str(rights_class).strip()
        if rights not in _RIGHTS_CLASSES:
            raise ValueError(
                "rights_class must be one of: " + ", ".join(sorted(_RIGHTS_CLASSES))
            )
        source = _safe_source_url(source_url)

        csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        digest = hashlib.sha256(csv_bytes).hexdigest()
        capture_id = f"{captured:%Y%m%dT%H%M%S%fZ}-{digest[:12]}"
        directory = self.root / season / topic_name / provider_name
        directory.mkdir(parents=True, exist_ok=True)
        snapshot_path = directory / f"{capture_id}.csv"
        manifest_path = self.manifest_path(snapshot_path)

        schema = [
            {"name": str(column), "dtype": str(frame[column].dtype)}
            for column in frame.columns
        ]
        manifest: dict[str, Any] = {
            "manifest_version": 1,
            "season_name": season,
            "topic": topic_name,
            "provider": provider_name,
            "captured_at": _iso_utc(captured),
            "source_url": source,
            "schema_version": str(schema_version),
            "schema": schema,
            "paid": bool(paid),
            # Keep the policy label explicit and provide the shorter spelling for
            # consumers that display provenance without knowing catalog internals.
            "rights_class": rights,
            "rights": rights,
            "hash": {"algorithm": "sha256", "value": digest},
            "sha256": digest,
            "row_count": int(len(frame)),
            "file_name": snapshot_path.name,
        }
        if extra_manifest:
            protected = set(manifest)
            overlap = protected.intersection(extra_manifest)
            if overlap:
                raise ValueError(
                    "extra_manifest cannot replace catalog fields: "
                    + ", ".join(sorted(overlap))
                )
            _validate_extra_manifest_urls(extra_manifest)
            manifest.update(extra_manifest)
        manifest["manifest_envelope_sha256"] = _manifest_envelope_sha256(manifest)
        manifest["manifest_integrity"] = {
            "algorithm": "sha256",
            "scope": "manifest-minus-integrity-fields",
            "value": manifest["manifest_envelope_sha256"],
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")

        csv_temp = _write_temp(directory, csv_bytes)
        manifest_temp: Path | None = None
        try:
            manifest_temp = _write_temp(directory, manifest_bytes)
            # A manifest is the publication marker, so it is installed last.
            os.replace(csv_temp, snapshot_path)
            os.replace(manifest_temp, manifest_path)
        finally:
            csv_temp.unlink(missing_ok=True)
            if manifest_temp is not None:
                manifest_temp.unlink(missing_ok=True)

        return snapshot_path

    def read_manifest(self, snapshot_path: str | Path) -> dict[str, Any]:
        """Read and minimally validate the sidecar for ``snapshot_path``."""
        path = Path(snapshot_path)
        manifest_path = self.manifest_path(path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing snapshot manifest: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid snapshot manifest: {manifest_path}") from exc
        required = {
            "captured_at",
            "source_url",
            "schema",
            "provider",
            "paid",
            "rights_class",
            "sha256",
        }
        missing = required.difference(manifest)
        if missing:
            raise ValueError(
                f"Snapshot manifest {manifest_path} is missing: {', '.join(sorted(missing))}"
            )
        _utc_datetime(manifest["captured_at"])
        if "source_url" in manifest:
            _safe_source_url(manifest["source_url"])
        _validate_extra_manifest_urls(manifest, path="manifest")
        integrity = manifest.get("manifest_integrity")
        envelope = manifest.get("manifest_envelope_sha256")
        if integrity is not None or envelope is not None:
            expected = _manifest_envelope_sha256(manifest)
            envelope_value = str(envelope or "")
            if not envelope_value:
                raise ValueError(f"Snapshot manifest {manifest_path} is missing envelope hash")
            if isinstance(integrity, dict):
                if integrity.get("algorithm") != "sha256" or integrity.get(
                    "scope"
                ) != "manifest-minus-integrity-fields":
                    raise ValueError(
                        f"Snapshot manifest {manifest_path} has invalid integrity metadata"
                    )
                if str(integrity.get("value", "")) != envelope_value:
                    raise ValueError(
                        f"Snapshot manifest {manifest_path} has inconsistent integrity metadata"
                    )
            if not hmac.compare_digest(expected, envelope_value):
                raise ValueError(f"Snapshot manifest {manifest_path} failed integrity validation")
        return manifest

    def resolve_all(
        self,
        *,
        season_name: str,
        topic: str,
        cutoff: datetime | str | None = None,
        provider: str | None = None,
    ) -> list[Path]:
        """Return every eligible snapshot for ``topic``, newest first."""
        season = _safe_component(season_name, field="season_name")
        topic_name = _safe_component(topic, field="topic")
        provider_name = (
            _safe_component(provider, field="provider") if provider is not None else None
        )
        limit = _utc_datetime(cutoff or datetime.now(timezone.utc))
        topic_dir = self.root / season / topic_name

        if provider_name is None:
            manifests = topic_dir.glob(f"*/*{_MANIFEST_SUFFIX}")
        else:
            manifests = (topic_dir / provider_name).glob(f"*{_MANIFEST_SUFFIX}")

        candidates: list[tuple[datetime, str, Path]] = []
        for manifest_path in manifests:
            snapshot_name = manifest_path.name[: -len(_MANIFEST_SUFFIX)] + ".csv"
            snapshot_path = manifest_path.with_name(snapshot_name)
            manifest = self.read_manifest(snapshot_path)
            if manifest.get("season_name") != season or manifest.get("topic") != topic_name:
                raise ValueError(f"Manifest location does not match its metadata: {manifest_path}")
            if manifest.get("provider") != manifest_path.parent.name:
                raise ValueError(f"Manifest location does not match its provider: {manifest_path}")
            if provider_name is not None and manifest.get("provider") != provider_name:
                raise ValueError(f"Manifest location does not match its provider: {manifest_path}")
            captured = _utc_datetime(manifest["captured_at"])
            if captured <= limit:
                candidates.append((captured, str(manifest.get("provider", "")), snapshot_path))
        ordered = sorted(candidates, key=lambda item: (item[0], item[1], str(item[2])), reverse=True)
        return [item[2] for item in ordered]

    def resolve(
        self,
        *,
        season_name: str,
        topic: str,
        cutoff: datetime | str | None = None,
        provider: str | None = None,
        required: bool = False,
    ) -> Path | None:
        """Return the newest published snapshot captured at or before ``cutoff``."""
        season = _safe_component(season_name, field="season_name")
        topic_name = _safe_component(topic, field="topic")
        limit = _utc_datetime(cutoff or datetime.now(timezone.utc))
        candidates = self.resolve_all(
            season_name=season,
            topic=topic_name,
            cutoff=limit,
            provider=provider,
        )
        if not candidates:
            if required:
                raise FileNotFoundError(
                    f"No {topic_name!r} snapshot for {season!r} at or before cutoff "
                    f"{_iso_utc(limit)}"
                )
            return None
        return candidates[0]

    def load_frame(self, snapshot_path: str | Path) -> pd.DataFrame:
        """Load a snapshot after checking the manifest's SHA-256 digest."""
        path = Path(snapshot_path)
        manifest = self.read_manifest(path)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise FileNotFoundError(f"Missing snapshot data: {path}") from exc
        actual = hashlib.sha256(payload).hexdigest()
        expected = str(manifest["sha256"])
        hash_record = manifest.get("hash", {})
        if isinstance(hash_record, dict):
            if hash_record.get("algorithm") != "sha256" or hash_record.get("value") != expected:
                raise ValueError(f"Inconsistent hash metadata in {self.manifest_path(path)}")
        if not hmac.compare_digest(actual, expected):
            raise ValueError(
                f"Snapshot hash mismatch for {path}: expected {expected}, got {actual}"
            )
        frame = pd.read_csv(path)
        expected_columns = [entry.get("name") for entry in manifest["schema"]]
        if list(frame.columns) != expected_columns:
            raise ValueError(f"Snapshot schema mismatch for {path}")
        if len(frame) != int(manifest.get("row_count", len(frame))):
            raise ValueError(f"Snapshot row-count mismatch for {path}")
        return frame
