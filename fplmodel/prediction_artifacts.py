"""Season metadata helpers for prediction CSV artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

import pandas as pd


PREDICTION_FILE_PATTERN = re.compile(r"predictions_gw(\d+)\.csv")
SEASON_NAME_PATTERN = re.compile(r"\d{4}-\d{2}")
REQUIRED_IDENTITY_COLUMNS = {"player_id", "full_name", "team_id", "team_name"}


@dataclass(frozen=True)
class LegacyPredictionMigration:
    season_name: str
    migrated_gameweeks: tuple[int, ...]
    already_tagged_gameweeks: tuple[int, ...]
    backup_dir: Path | None


@dataclass(frozen=True)
class PredictionArtifactInspection:
    source_season_name: str
    ready_gameweeks: tuple[int, ...]
    migratable_gameweeks: tuple[int, ...]
    issues: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class PredictionSeasonSource:
    season_name: str
    predictions_dir: Path
    bootstrap_path: Path
    archived: bool


def infer_bootstrap_season_name(bootstrap_path: Path) -> str | None:
    """Infer a season such as ``2025-26`` from cached FPL event deadlines."""
    try:
        with Path(bootstrap_path).open("r", encoding="utf-8") as handle:
            bootstrap = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None

    events = bootstrap.get("events", []) if isinstance(bootstrap, dict) else []
    deadlines = pd.to_datetime(
        [event.get("deadline_time") for event in events if isinstance(event, dict)],
        errors="coerce",
        utc=True,
    )
    valid_deadlines = deadlines[~pd.isna(deadlines)]
    if len(valid_deadlines) == 0:
        return None
    start_year = int(valid_deadlines.min().year)
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _prediction_files(output_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in Path(output_dir).glob("predictions_gw*.csv"):
        match = PREDICTION_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        gameweek = int(match.group(1))
        if 1 <= gameweek <= 38:
            files.append((gameweek, path))
    return sorted(files)


def _prediction_directory_season(output_dir: Path) -> str | None:
    season_names: set[str] = set()
    for gameweek, path in _prediction_files(output_dir):
        try:
            metadata = pd.read_csv(path, usecols=["season_name", "gameweek"])
        except (OSError, ValueError):
            return None
        seasons = {str(value) for value in metadata["season_name"].dropna().unique()}
        gameweeks = {
            int(value)
            for value in pd.to_numeric(
                metadata["gameweek"], errors="coerce"
            ).dropna().unique()
        }
        if len(seasons) != 1 or gameweeks != {gameweek}:
            return None
        season_names.update(seasons)
    return next(iter(season_names)) if len(season_names) == 1 else None


def discover_prediction_season_sources(
    *,
    output_root: Path,
    current_bootstrap_path: Path,
) -> dict[str, PredictionSeasonSource]:
    """Find selectable prediction seasons and their matching bootstrap snapshots."""
    output_root = Path(output_root)
    sources: dict[str, PredictionSeasonSource] = {}
    archive_root = output_root / "seasons"
    if archive_root.is_dir():
        for season_dir in archive_root.iterdir():
            if not season_dir.is_dir() or not SEASON_NAME_PATTERN.fullmatch(
                season_dir.name
            ):
                continue
            bootstrap_path = season_dir / "bootstrap-static.json"
            if (
                not bootstrap_path.is_file()
                or infer_bootstrap_season_name(bootstrap_path) != season_dir.name
                or _prediction_directory_season(season_dir) != season_dir.name
            ):
                continue
            sources[season_dir.name] = PredictionSeasonSource(
                season_name=season_dir.name,
                predictions_dir=season_dir,
                bootstrap_path=bootstrap_path,
                archived=True,
            )

    root_season = _prediction_directory_season(output_root)
    if root_season is not None and root_season not in sources:
        sources[root_season] = PredictionSeasonSource(
            season_name=root_season,
            predictions_dir=output_root,
            bootstrap_path=Path(current_bootstrap_path),
            archived=False,
        )
    elif _prediction_files(output_root):
        bootstrap_season = infer_bootstrap_season_name(current_bootstrap_path)
        if bootstrap_season is not None and bootstrap_season not in sources:
            sources[bootstrap_season] = PredictionSeasonSource(
                season_name=bootstrap_season,
                predictions_dir=output_root,
                bootstrap_path=Path(current_bootstrap_path),
                archived=False,
            )

    return dict(sorted(sources.items(), reverse=True))


def legacy_prediction_gameweeks(output_dir: Path) -> list[int]:
    """Return gameweeks whose prediction files lack season metadata."""
    gameweeks: list[int] = []
    for gameweek, path in _prediction_files(output_dir):
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except (OSError, ValueError):
            continue
        if "season_name" not in columns:
            gameweeks.append(gameweek)
    return gameweeks


def _load_bootstrap_identities(bootstrap_path: Path) -> tuple[str, pd.DataFrame]:
    season_name = infer_bootstrap_season_name(bootstrap_path)
    if season_name is None:
        raise ValueError("Could not infer a season from the cached FPL bootstrap data.")
    try:
        with Path(bootstrap_path).open("r", encoding="utf-8") as handle:
            bootstrap = json.load(handle)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("Could not read the cached FPL bootstrap data.") from exc

    teams = {
        int(team["id"]): str(team["name"])
        for team in bootstrap.get("teams", [])
        if isinstance(team, dict) and "id" in team and "name" in team
    }
    rows = []
    for player in bootstrap.get("elements", []):
        if not isinstance(player, dict):
            continue
        try:
            team_id = int(player["team"])
            rows.append(
                {
                    "player_id": int(player["id"]),
                    "full_name": f"{player['first_name']} {player['second_name']}",
                    "team_id": team_id,
                    "team_name": teams[team_id],
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        raise ValueError("Cached FPL bootstrap data contains no usable player identities.")
    return season_name, pd.DataFrame(rows).set_index("player_id")


def _validate_prediction_identity(
    frame: pd.DataFrame,
    path: Path,
    gameweek: int,
    identities: pd.DataFrame,
) -> None:
    missing_columns = sorted(REQUIRED_IDENTITY_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError(f"{path.name} is empty.")

    player_ids = pd.to_numeric(frame["player_id"], errors="coerce")
    if player_ids.isna().any():
        raise ValueError(f"{path.name} contains invalid player IDs.")
    player_ids = player_ids.astype(int)
    unknown_ids = sorted(set(player_ids) - set(identities.index))
    if unknown_ids:
        raise ValueError(
            f"{path.name} contains player IDs not present in the cached FPL data: "
            f"{unknown_ids[:10]}"
        )

    expected = identities.loc[player_ids].reset_index(drop=True)
    actual = frame.reset_index(drop=True)
    identity_mismatches = []
    for column in ("full_name", "team_name"):
        mismatch = actual[column].astype(str).ne(expected[column].astype(str))
        if mismatch.any():
            identity_mismatches.append(column)
    actual_team_ids = pd.to_numeric(actual["team_id"], errors="coerce")
    if actual_team_ids.isna().any() or not actual_team_ids.astype(int).equals(
        expected["team_id"].astype(int)
    ):
        identity_mismatches.append("team_id")
    if identity_mismatches:
        raise ValueError(
            f"{path.name} does not match cached FPL player identity fields: "
            f"{sorted(set(identity_mismatches))}"
        )

    if "gameweek" in frame.columns:
        gameweeks = {
            int(value)
            for value in pd.to_numeric(frame["gameweek"], errors="coerce").dropna().unique()
        }
        if gameweeks and gameweeks != {gameweek}:
            raise ValueError(f"{path.name} contains gameweek metadata {sorted(gameweeks)}.")


def inspect_prediction_artifacts(
    *,
    output_dir: Path,
    bootstrap_path: Path,
    expected_season_name: str,
) -> PredictionArtifactInspection:
    """Classify prediction files after validating metadata and FPL identities."""
    source_season = infer_bootstrap_season_name(bootstrap_path)
    if source_season is None:
        raise ValueError("Could not infer a season from the cached FPL bootstrap data.")
    identity_cache: dict[Path, tuple[str, pd.DataFrame]] = {}
    ready: list[int] = []
    migratable: list[int] = []
    issues: list[tuple[int, str]] = []
    for gameweek, path in _prediction_files(output_dir):
        try:
            gameweek_bootstrap = Path(output_dir) / f"bootstrap_gw{gameweek}.json"
            identity_path = (
                gameweek_bootstrap if gameweek_bootstrap.is_file() else Path(bootstrap_path)
            )
            if identity_path not in identity_cache:
                identity_cache[identity_path] = _load_bootstrap_identities(identity_path)
            gameweek_season, identities = identity_cache[identity_path]
            frame = pd.read_csv(path)
            _validate_prediction_identity(frame, path, gameweek, identities)
        except (OSError, ValueError) as exc:
            issues.append((gameweek, str(exc)))
            continue

        if gameweek_season != expected_season_name:
            issues.append(
                (
                    gameweek,
                    f"Cached FPL data is for {gameweek_season}, not "
                    f"{expected_season_name}.",
                )
            )
            continue
        if "season_name" not in frame.columns:
            migratable.append(gameweek)
            continue

        seasons = {str(value) for value in frame["season_name"].dropna().unique()}
        if seasons != {expected_season_name}:
            issues.append(
                (
                    gameweek,
                    f"{path.name} is tagged for {sorted(seasons) or ['unknown']}, "
                    f"not {expected_season_name}.",
                )
            )
            continue
        artifact_gameweeks = {
            int(value)
            for value in pd.to_numeric(
                frame.get("gameweek", pd.Series(dtype=float)), errors="coerce"
            ).dropna().unique()
        }
        if artifact_gameweeks != {gameweek}:
            issues.append(
                (
                    gameweek,
                    f"{path.name} has missing or invalid gameweek metadata.",
                )
            )
            continue
        ready.append(gameweek)

    return PredictionArtifactInspection(
        source_season_name=source_season,
        ready_gameweeks=tuple(ready),
        migratable_gameweeks=tuple(migratable),
        issues=tuple(issues),
    )


def archive_prediction_file(
    *,
    predictions_path: Path,
    output_root: Path,
    bootstrap_path: Path,
    season_name: str,
    gameweek: int,
) -> Path:
    """Copy one validated prediction file into its durable season archive."""
    if not SEASON_NAME_PATTERN.fullmatch(str(season_name)):
        raise ValueError("season_name must use YYYY-YY format")
    if not 1 <= int(gameweek) <= 38:
        raise ValueError("gameweek must be between 1 and 38")

    source_season, identities = _load_bootstrap_identities(bootstrap_path)
    if source_season != season_name:
        raise ValueError(
            f"Bootstrap data is for {source_season}, not archive season {season_name}."
        )
    predictions_path = Path(predictions_path)
    frame = pd.read_csv(predictions_path)
    _validate_prediction_identity(frame, predictions_path, int(gameweek), identities)
    season_values = frame.get("season_name", pd.Series(dtype=str))
    seasons = {str(value) for value in season_values.dropna().unique()}
    artifact_gameweeks = {
        int(value)
        for value in pd.to_numeric(
            frame.get("gameweek", pd.Series(dtype=float)), errors="coerce"
        ).dropna().unique()
    }
    if seasons != {season_name} or artifact_gameweeks != {int(gameweek)}:
        raise ValueError(
            f"{predictions_path.name} metadata does not match {season_name} GW{gameweek}."
        )

    archive_dir = Path(output_root) / "seasons" / season_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_bootstrap = archive_dir / "bootstrap-static.json"
    gameweek_bootstrap = archive_dir / f"bootstrap_gw{int(gameweek)}.json"
    archived_predictions = archive_dir / f"predictions_gw{int(gameweek)}.csv"
    copies = (
        (Path(bootstrap_path), archived_bootstrap),
        (Path(bootstrap_path), gameweek_bootstrap),
        (predictions_path, archived_predictions),
    )
    temporary_paths: list[Path] = []
    try:
        for source, destination in copies:
            temporary = destination.with_name(f".{destination.name}.archive.tmp")
            shutil.copy2(source, temporary)
            temporary_paths.append(temporary)
            temporary.replace(destination)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return archived_predictions


def _append_metadata_without_reserializing(
    path: Path,
    *,
    row_count: int,
    season_name: str,
    gameweek: int,
) -> bytes:
    """Append metadata fields while preserving every existing CSV byte."""
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if len(lines) != row_count + 1:
        raise ValueError(
            f"{path.name} contains multiline or blank CSV rows and cannot be migrated "
            "without reserializing prediction values. Regenerate it instead."
        )

    def append_fields(line: bytes, fields: bytes) -> bytes:
        if line.endswith(b"\r\n"):
            return line[:-2] + fields + b"\r\n"
        if line.endswith((b"\n", b"\r")):
            return line[:-1] + fields + line[-1:]
        return line + fields

    migrated = [append_fields(lines[0], b",season_name,gameweek")]
    metadata = f",{season_name},{gameweek}".encode("ascii")
    migrated.extend(append_fields(line, metadata) for line in lines[1:])
    return b"".join(migrated)


def migrate_legacy_prediction_files(
    *,
    output_dir: Path,
    bootstrap_path: Path,
    expected_season_name: str,
) -> LegacyPredictionMigration:
    """Validate legacy predictions, back them up, then add season/gameweek metadata."""
    source_season, identities = _load_bootstrap_identities(bootstrap_path)
    if source_season != expected_season_name:
        raise ValueError(
            f"Cached FPL data is for {source_season}, so prediction files cannot be "
            f"tagged as {expected_season_name}."
        )

    prediction_files = _prediction_files(output_dir)
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {output_dir}")

    pending: list[tuple[int, Path, pd.DataFrame]] = []
    already_tagged: list[int] = []
    for gameweek, path in prediction_files:
        frame = pd.read_csv(path)
        _validate_prediction_identity(frame, path, gameweek, identities)
        if "season_name" in frame.columns:
            seasons = {str(value) for value in frame["season_name"].dropna().unique()}
            artifact_gameweeks = {
                int(value)
                for value in pd.to_numeric(
                    frame.get("gameweek", pd.Series(dtype=float)), errors="coerce"
                ).dropna().unique()
            }
            if seasons != {expected_season_name} or artifact_gameweeks != {gameweek}:
                raise ValueError(f"{path.name} already contains conflicting metadata.")
            already_tagged.append(gameweek)
            continue
        pending.append((gameweek, path, frame))

    if not pending:
        return LegacyPredictionMigration(
            season_name=expected_season_name,
            migrated_gameweeks=(),
            already_tagged_gameweeks=tuple(already_tagged),
            backup_dir=None,
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = (
        Path(output_dir)
        / "prediction_backups"
        / f"legacy_{expected_season_name}_{timestamp}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)
    temporary_paths: list[Path] = []
    try:
        for gameweek, path, frame in pending:
            shutil.copy2(path, backup_dir / path.name)
            temporary_path = path.with_name(f".{path.name}.migration.tmp")
            temporary_path.write_bytes(
                _append_metadata_without_reserializing(
                    path,
                    row_count=len(frame),
                    season_name=expected_season_name,
                    gameweek=gameweek,
                )
            )
            temporary_paths.append(temporary_path)
        for (_, path, _), temporary_path in zip(pending, temporary_paths):
            temporary_path.replace(path)
    except Exception:
        for _, path, _ in pending:
            backup_path = backup_dir / path.name
            if backup_path.exists():
                shutil.copy2(backup_path, path)
        raise
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)

    return LegacyPredictionMigration(
        season_name=expected_season_name,
        migrated_gameweeks=tuple(gameweek for gameweek, _, _ in pending),
        already_tagged_gameweeks=tuple(already_tagged),
        backup_dir=backup_dir,
    )


__all__ = [
    "LegacyPredictionMigration",
    "PredictionArtifactInspection",
    "PredictionSeasonSource",
    "archive_prediction_file",
    "discover_prediction_season_sources",
    "infer_bootstrap_season_name",
    "inspect_prediction_artifacts",
    "legacy_prediction_gameweeks",
    "migrate_legacy_prediction_files",
]
