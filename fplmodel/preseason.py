"""Transparent preseason player-usage priors.

The functions in this module intentionally derive priors only from observed
usage.  Missing observations stay missing and therefore cannot quietly look like
zero minutes or a benching.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    STRICT_GW1_MIN_EVIDENCED_PLAYERS_PER_TEAM,
    STRICT_GW1_MIN_PLAYER_EVIDENCE_QUALITY,
    STRICT_GW1_MIN_PRESEASON_MATCHES,
    STRICT_GW1_MIN_SOURCE_CONFIDENCE,
    STRICT_GW1_PLAYER_EVIDENCE_MAX_AGE_DAYS,
    STRICT_GW1_TEAM_PRIOR_MAX_AGE_DAYS,
)
from .snapshot_catalog import SnapshotCatalog


USAGE_COLUMNS = (
    "team_preseason_matches",
    "preseason_appearances",
    "preseason_starts",
    "preseason_minutes",
    "preseason_last_two_minutes",
    "source_confidence",
)

SNAPSHOT_TOPICS = (
    "preseason_player_usage",
    "preseason_role_evidence",
    "tournament_workload",
    "team_priors",
    "fixture_odds",
    "provider_expected_points",
)

PLAYER_IDENTITY_COLUMNS = {"player_code", "player_id", "full_name", "team_name", "team_code"}
PLAYER_SHARED_PRECEDENCE = {
    "readiness_multiplier": ("tournament_workload",)
}
ROLE_COMPONENT_WEIGHTS = {
    "attacking_role_delta": 0.45,
    "penalty_role_delta": 0.35,
    "corner_role_delta": 0.12,
    "direct_fk_role_delta": 0.08,
}
GW1_PLAYER_REQUIRED_COLUMNS = {"player_code", "prior_p_start", "prior_p_appearance"}
GW1_TEAM_PRIOR_REQUIRED_COLUMNS = {
    "team_code",
    "team_id",
    "team_goals_for_ma5",
    "team_goals_against_ma5",
    "team_goal_diff_ma5",
    "team_clean_sheet_ma5",
    "team_conceded_two_plus_ma5",
    "team_match_points_ma5",
    "team_prior_matches",
    "team_prior_source",
}


@dataclass(frozen=True)
class PreseasonSnapshotContext:
    """Resolved optional enrichment inputs and their auditable provenance."""

    player_enrichment: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_priors: pd.DataFrame = field(default_factory=pd.DataFrame)
    fixture_odds: pd.DataFrame = field(default_factory=pd.DataFrame)
    snapshot_paths: tuple[Path, ...] = ()
    manifests: tuple[dict[str, Any], ...] = ()

    def manifest_payload(self, *, cutoff: datetime | str, mode: str) -> dict[str, Any]:
        return {
            "snapshot_mode": mode,
            "cutoff": str(cutoff),
            "snapshots": [
                {
                    "path": "/".join(path.parts[-4:]),
                    "topic": manifest.get("topic"),
                    "provider": manifest.get("provider"),
                    "captured_at": manifest.get("captured_at"),
                    "schema_version": manifest.get("schema_version"),
                    "paid": bool(manifest.get("paid", False)),
                    "rights_class": manifest.get("rights_class"),
                    "source_url": manifest.get("source_url"),
                    "sha256": manifest.get("sha256"),
                    "manifest_envelope_sha256": manifest.get(
                        "manifest_envelope_sha256"
                    ),
                    "row_count": manifest.get("row_count"),
                }
                for path, manifest in zip(self.snapshot_paths, self.manifests)
            ],
        }


def _identity_key(frame: pd.DataFrame) -> str:
    if "player_code" in frame.columns:
        return "player_code"
    if "player_id" in frame.columns:
        return "player_id"
    raise ValueError("Player enrichment snapshots must include player_code or player_id")


def _validate_identity_columns(
    merged: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    key: str,
    topic: str,
) -> None:
    shared = (set(merged.columns) & set(frame.columns)) - {key}
    for column in sorted(shared):
        if column in PLAYER_IDENTITY_COLUMNS:
            left = merged[[key, column]].dropna(subset=[column]).rename(
                columns={column: f"{column}_left"}
            )
            right = frame[[key, column]].dropna(subset=[column]).rename(
                columns={column: f"{column}_right"}
            )
            if left.empty or right.empty:
                continue
            joined = left.merge(right, on=key, how="inner")
            if joined.empty:
                continue
            mismatch = joined[
                joined[f"{column}_left"].astype(str) != joined[f"{column}_right"].astype(str)
            ]
            if not mismatch.empty:
                raise ValueError(
                    f"Player enrichment snapshots disagree on {column} for topic {topic}"
                )


def _merge_player_frames(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    nonempty = [
        (topic, frame.copy())
        for topic, frame in frames
        if frame is not None and not frame.empty
    ]
    if not nonempty:
        return pd.DataFrame()
    merged: pd.DataFrame | None = None
    for topic, frame in nonempty:
        key = _identity_key(frame)
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"Player enrichment snapshot must have unique non-null {key}")
        topic_frame = frame.copy()
        for column in PLAYER_SHARED_PRECEDENCE:
            if column in topic_frame.columns:
                topic_frame = topic_frame.rename(columns={column: f"{column}__{topic}"})
        if merged is None:
            merged = topic_frame
            continue
        merged_key = _identity_key(merged)
        if merged_key != key:
            raise ValueError("Player enrichment snapshots must use the same identity key")
        _validate_identity_columns(merged, frame, key=key, topic=topic)
        overlaps = (set(merged.columns) & set(topic_frame.columns)) - {key}
        allowed_shared = PLAYER_IDENTITY_COLUMNS
        true_conflicts = overlaps - allowed_shared
        if true_conflicts:
            raise ValueError(
                "Player enrichment snapshots have ambiguous columns: "
                + ", ".join(sorted(true_conflicts))
            )
        topic_frame = topic_frame.drop(
            columns=list(PLAYER_IDENTITY_COLUMNS - {key}),
            errors="ignore",
        )
        merged = merged.merge(topic_frame, on=key, how="outer", validate="one_to_one")
    if merged is None:
        return pd.DataFrame()
    for column in PLAYER_SHARED_PRECEDENCE:
        owned_columns = [f"{column}__{owner}" for owner in PLAYER_SHARED_PRECEDENCE[column]]
        present_columns = [name for name in owned_columns if name in merged.columns]
        if not present_columns:
            continue
        winner = pd.Series(pd.NA, index=merged.index, dtype="object")
        for source_column in present_columns:
            winner = winner.where(winner.notna(), merged[source_column])
        merged[column] = winner
    for column in PLAYER_SHARED_PRECEDENCE:
        owned_columns = [name for name in merged.columns if name.startswith(f"{column}__")]
        if owned_columns:
            merged = merged.drop(columns=owned_columns)
    return merged if merged is not None else pd.DataFrame()


def _resolve_provider_expected_points(
    catalog: SnapshotCatalog,
    *,
    season_name: str,
    cutoff: datetime | str,
    required: bool,
    target_gameweek: int | None,
) -> tuple[pd.DataFrame, Path | None, dict[str, Any] | None]:
    for path in catalog.resolve_all(
        season_name=season_name,
        topic="provider_expected_points",
        cutoff=cutoff,
    ):
        frame = catalog.load_frame(path)
        manifest = catalog.read_manifest(path)
        if "gameweek" not in frame.columns:
            raise ValueError("provider_expected_points snapshot must include gameweek")
        if target_gameweek is None:
            return frame, path, manifest
        filtered = frame[
            pd.to_numeric(frame["gameweek"], errors="coerce").eq(int(target_gameweek))
        ].copy()
        if not filtered.empty:
            return filtered, path, manifest
    if required:
        if target_gameweek is None:
            raise FileNotFoundError(
                f"No 'provider_expected_points' snapshot for {season_name!r} at or before cutoff {cutoff}"
            )
        raise FileNotFoundError(
            "No 'provider_expected_points' snapshot matches "
            f"gameweek {int(target_gameweek)} at or before cutoff {cutoff}"
        )
    return pd.DataFrame(), None, None


def _resolve_preseason_role_evidence(
    catalog: SnapshotCatalog,
    *,
    season_name: str,
    cutoff: datetime | str,
    required: bool,
    target_gameweek: int | None,
) -> tuple[pd.DataFrame, Path | None, dict[str, Any] | None]:
    """Resolve the newest role snapshot that contains the requested gameweek."""
    for path in catalog.resolve_all(
        season_name=season_name,
        topic="preseason_role_evidence",
        cutoff=cutoff,
    ):
        frame = catalog.load_frame(path)
        manifest = catalog.read_manifest(path)
        if "role_target_gameweek" not in frame.columns:
            raise ValueError(
                "preseason_role_evidence snapshot must include role_target_gameweek"
            )
        if target_gameweek is not None:
            frame = frame[
                pd.to_numeric(frame["role_target_gameweek"], errors="coerce").eq(
                    int(target_gameweek)
                )
            ].copy()
        if not frame.empty:
            return derive_preseason_role_evidence(frame), path, manifest
    if required:
        target = "" if target_gameweek is None else f" for gameweek {int(target_gameweek)}"
        raise FileNotFoundError(
            f"No 'preseason_role_evidence' snapshot{target} at or before cutoff {cutoff}"
        )
    return pd.DataFrame(), None, None


def _resolve_tournament_workload(
    catalog: SnapshotCatalog,
    *,
    season_name: str,
    cutoff: datetime | str,
    required: bool,
    target_gameweek: int | None,
) -> tuple[pd.DataFrame, Path | None, dict[str, Any] | None]:
    """Resolve workload evidence, respecting an optional GW expiry column."""
    paths = catalog.resolve_all(
        season_name=season_name,
        topic="tournament_workload",
        cutoff=cutoff,
    )
    if paths:
        # Workload snapshots are full replacements. An expired latest snapshot
        # must not resurrect an older readiness judgement.
        path = paths[0]
        frame = normalize_tournament_workload(catalog.load_frame(path))
        manifest = catalog.read_manifest(path)
        if target_gameweek is not None:
            valid_through = pd.to_numeric(
                frame["workload_valid_through_gameweek"], errors="coerce"
            )
            frame = frame[valid_through.ge(int(target_gameweek))].copy()
        if not frame.empty:
            return frame, path, manifest
    if required:
        raise FileNotFoundError(
            f"No eligible 'tournament_workload' snapshot at or before cutoff {cutoff}"
        )
    return pd.DataFrame(), None, None


def _validate_observation_cutoff(
    frame: pd.DataFrame,
    *,
    column: str,
    topic: str,
    manifest: dict[str, Any],
    cutoff: datetime | str,
) -> None:
    if column not in frame.columns or frame.empty:
        return
    observed = pd.to_datetime(frame[column], errors="coerce", utc=True)
    populated = frame[column].notna()
    if bool((populated & observed.isna()).any()):
        raise ValueError(f"{topic} has invalid {column} timestamps")
    captured_at = pd.to_datetime(manifest.get("captured_at"), errors="coerce", utc=True)
    cutoff_at = pd.to_datetime(cutoff, errors="coerce", utc=True)
    if pd.isna(captured_at) or pd.isna(cutoff_at):
        raise ValueError(f"{topic} has invalid snapshot timing metadata")
    if bool((observed.dropna() > captured_at).any()):
        raise ValueError(f"{topic} observation occurs after snapshot capture")
    if bool((observed.dropna() > cutoff_at).any()):
        raise ValueError(f"{topic} observation occurs after prediction cutoff")


def _validate_strict_gw1_context(
    context: PreseasonSnapshotContext,
    *,
    expected_team_codes: set[int] | None = None,
    raw_usage: pd.DataFrame | None = None,
    usage_manifest: dict[str, Any] | None = None,
    cutoff: datetime | str | None = None,
    expected_players: pd.DataFrame | None = None,
    expected_teams: pd.DataFrame | None = None,
    team_prior_manifest: dict[str, Any] | None = None,
    season_name: str | None = None,
) -> None:
    if context.player_enrichment.empty:
        raise ValueError("Strict GW1 preseason enrichment requires player evidence")
    missing_player_columns = GW1_PLAYER_REQUIRED_COLUMNS - set(context.player_enrichment.columns)
    if missing_player_columns:
        raise ValueError(
            "Strict GW1 preseason enrichment is missing player columns: "
            + ", ".join(sorted(missing_player_columns))
        )
    player_evidence = context.player_enrichment[
        ["prior_p_start", "prior_p_appearance"]
    ].notna().any(axis=1)
    if not bool(player_evidence.any()):
        raise ValueError("Strict GW1 preseason enrichment requires non-null player evidence")

    if expected_players is not None:
        _validate_strict_gw1_player_usage(
            raw_usage if raw_usage is not None else pd.DataFrame(),
            expected_players=expected_players,
            usage_manifest=usage_manifest,
            cutoff=cutoff,
        )

    team_priors = context.team_priors
    if team_priors.empty:
        raise ValueError("Strict GW1 preseason enrichment requires team priors")
    missing_team_columns = GW1_TEAM_PRIOR_REQUIRED_COLUMNS - set(team_priors.columns)
    if missing_team_columns:
        raise ValueError(
            "Strict GW1 preseason enrichment is missing team-prior columns: "
            + ", ".join(sorted(missing_team_columns))
        )
    required_frame = team_priors[list(GW1_TEAM_PRIOR_REQUIRED_COLUMNS)]
    if required_frame.isna().any().any():
        raise ValueError("Strict GW1 team priors must be fully populated")
    if team_priors["team_code"].duplicated().any():
        raise ValueError("Strict GW1 team priors must have unique team_code values")
    if int(team_priors["team_code"].nunique()) != 20 or len(team_priors) != 20:
        raise ValueError("Strict GW1 team priors must cover exactly 20 teams")
    if expected_team_codes is not None:
        expected = {int(value) for value in expected_team_codes}
        actual = set(
            pd.to_numeric(team_priors["team_code"], errors="raise").astype(int)
        )
        if actual != expected:
            raise ValueError(
                "Strict GW1 team priors do not match the current team codes; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    if expected_teams is not None:
        _validate_strict_gw1_team_priors(
            team_priors,
            expected_teams=expected_teams,
            manifest=team_prior_manifest,
            cutoff=cutoff,
            season_name=season_name,
        )


def _validate_strict_gw1_team_priors(
    team_priors: pd.DataFrame,
    *,
    expected_teams: pd.DataFrame,
    manifest: dict[str, Any] | None,
    cutoff: datetime | str | None,
    season_name: str | None,
) -> None:
    expected = expected_teams.copy()
    if "team_code" not in expected.columns and "code" in expected.columns:
        expected = expected.rename(columns={"code": "team_code"})
    if not {"team_id", "team_code"}.issubset(expected.columns):
        raise ValueError("Strict GW1 validation requires current team_id and team_code")
    for column in ("team_id", "team_code"):
        values = pd.to_numeric(expected[column], errors="coerce")
        if (
            values.isna().any()
            or values.mod(1).ne(0).any()
            or values.duplicated().any()
        ):
            raise ValueError(f"Current teams must have unique integral {column} values")
        expected[column] = values.astype(int)
    actual = team_priors[["team_id", "team_code"]].copy()
    for column in ("team_id", "team_code"):
        values = pd.to_numeric(actual[column], errors="coerce")
        if values.isna().any() or values.mod(1).ne(0).any():
            raise ValueError(f"Strict GW1 team priors have invalid {column}")
        actual[column] = values.astype(int)
    expected_pairs = set(map(tuple, expected[["team_code", "team_id"]].to_numpy()))
    actual_pairs = set(map(tuple, actual[["team_code", "team_id"]].to_numpy()))
    if actual_pairs != expected_pairs:
        raise ValueError("Strict GW1 team priors have stale team_code-to-team_id mapping")

    if manifest is None or cutoff is None or season_name is None:
        raise ValueError("Strict GW1 team priors require cutoff-safe manifest metadata")
    try:
        current_start = int(str(season_name).split("-", 1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("Strict GW1 team priors require a valid season name") from exc
    expected_prior_season = f"{current_start - 1}-{current_start % 100:02d}"
    if manifest.get("prior_season") != expected_prior_season:
        raise ValueError(
            "Strict GW1 team-prior manifest prior_season must be "
            f"{expected_prior_season}"
        )
    if "team_prior_as_of" not in team_priors.columns:
        raise ValueError("Strict GW1 team priors require team_prior_as_of")
    as_of = pd.to_datetime(team_priors["team_prior_as_of"], errors="coerce", utc=True)
    cutoff_at = pd.to_datetime(cutoff, errors="coerce", utc=True)
    captured_at = pd.to_datetime(manifest.get("captured_at"), errors="coerce", utc=True)
    if as_of.isna().any() or pd.isna(cutoff_at) or pd.isna(captured_at):
        raise ValueError("Strict GW1 team priors have invalid as-of metadata")
    prior_window_start = pd.Timestamp(
        year=current_start - 1, month=7, day=1, tz="UTC"
    )
    prior_window_end = pd.Timestamp(
        year=current_start, month=6, day=30, hour=23, minute=59, second=59, tz="UTC"
    )
    if bool(((as_of < prior_window_start) | (as_of > prior_window_end)).any()):
        raise ValueError(
            f"Strict GW1 team-prior as-of dates fall outside {expected_prior_season}"
        )
    if bool((as_of > captured_at).any()) or bool((as_of > cutoff_at).any()):
        raise ValueError("Strict GW1 team priors contain post-cutoff evidence")
    max_age = pd.Timedelta(days=int(STRICT_GW1_TEAM_PRIOR_MAX_AGE_DAYS))
    if bool(((cutoff_at - as_of) > max_age).any()):
        raise ValueError(
            "Strict GW1 team priors are older than "
            f"{int(STRICT_GW1_TEAM_PRIOR_MAX_AGE_DAYS)} days"
        )


def _strict_integer_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    populated = frame[column].notna()
    invalid = populated & (
        values.isna()
        | ~np.isfinite(values)
        | values.lt(0)
        | values.mod(1).ne(0)
    )
    if bool(invalid.any()):
        raise ValueError(f"Strict GW1 player usage has invalid {column}")
    return values.astype(float)


def _validate_strict_gw1_player_usage(
    usage: pd.DataFrame,
    *,
    expected_players: pd.DataFrame,
    usage_manifest: dict[str, Any] | None,
    cutoff: datetime | str | None,
) -> None:
    """Validate final-GW1 usage against the live bootstrap player universe."""
    raw_required = {
        "player_code",
        "team_preseason_matches",
        "preseason_appearances",
        "preseason_starts",
        "preseason_minutes",
        "preseason_last_two_minutes",
        "source_confidence",
    }
    missing = raw_required - set(usage.columns)
    if missing:
        raise ValueError(
            "Strict GW1 player usage is missing raw evidence columns: "
            + ", ".join(sorted(missing))
        )
    if usage_manifest is None or cutoff is None:
        raise ValueError("Strict GW1 player usage requires cutoff-safe manifest metadata")
    captured_at = pd.to_datetime(usage_manifest.get("captured_at"), errors="coerce", utc=True)
    cutoff_at = pd.to_datetime(cutoff, errors="coerce", utc=True)
    if pd.isna(captured_at) or pd.isna(cutoff_at):
        raise ValueError("Strict GW1 player usage has invalid freshness metadata")
    age = cutoff_at - captured_at
    if age < pd.Timedelta(0):
        raise ValueError("Strict GW1 player usage was captured after the cutoff")
    max_age = pd.Timedelta(days=int(STRICT_GW1_PLAYER_EVIDENCE_MAX_AGE_DAYS))
    if age > max_age:
        raise ValueError(
            "Strict GW1 player usage is older than "
            f"{int(STRICT_GW1_PLAYER_EVIDENCE_MAX_AGE_DAYS)} days"
        )

    if expected_players.empty or "player_code" not in expected_players.columns:
        raise ValueError("Strict GW1 validation requires the live player-code universe")
    expected = expected_players.copy()
    expected_codes_numeric = pd.to_numeric(expected["player_code"], errors="coerce")
    if (
        expected_codes_numeric.isna().any()
        or expected_codes_numeric.duplicated().any()
        or expected_codes_numeric.mod(1).ne(0).any()
    ):
        raise ValueError("Live player universe must have unique integral player_code values")
    expected["player_code"] = expected_codes_numeric.astype(int)
    if "team_code" not in expected.columns:
        raise ValueError("Live player universe must include team_code")
    expected_team_numeric = pd.to_numeric(expected["team_code"], errors="coerce")
    if expected_team_numeric.isna().any() or expected_team_numeric.mod(1).ne(0).any():
        raise ValueError("Live player universe must have integral team_code values")
    expected["team_code"] = expected_team_numeric.astype(int)

    usage_codes_numeric = pd.to_numeric(usage["player_code"], errors="coerce")
    if (
        usage_codes_numeric.isna().any()
        or usage_codes_numeric.duplicated().any()
        or usage_codes_numeric.mod(1).ne(0).any()
    ):
        raise ValueError(
            "Strict GW1 player usage must have unique integral player_code values"
        )
    checked = usage.copy()
    checked["player_code"] = usage_codes_numeric.astype(int)
    expected_code_set = set(expected["player_code"])
    actual_code_set = set(checked["player_code"])
    if actual_code_set != expected_code_set:
        raise ValueError(
            "Strict GW1 player usage does not match the current player roster; "
            f"missing={sorted(expected_code_set - actual_code_set)}, "
            f"extra={sorted(actual_code_set - expected_code_set)}"
        )

    identity = checked.merge(
        expected,
        on="player_code",
        how="left",
        suffixes=("_snapshot", "_live"),
        validate="one_to_one",
    )
    for column in ("player_id", "team_code"):
        snapshot_column = f"{column}_snapshot"
        live_column = f"{column}_live"
        if snapshot_column not in identity.columns or live_column not in identity.columns:
            continue
        left = pd.to_numeric(identity[snapshot_column], errors="coerce")
        right = pd.to_numeric(identity[live_column], errors="coerce")
        mismatch = identity[snapshot_column].notna() & left.ne(right)
        if bool(mismatch.any()):
            codes = sorted(identity.loc[mismatch, "player_code"].astype(int).tolist())
            raise ValueError(
                f"Strict GW1 player usage has stale {column} for player codes {codes}"
            )
    for column in ("full_name", "team_name"):
        snapshot_column = f"{column}_snapshot"
        live_column = f"{column}_live"
        if snapshot_column not in identity.columns or live_column not in identity.columns:
            continue
        left = identity[snapshot_column].fillna("").astype(str).str.strip().str.casefold()
        right = identity[live_column].fillna("").astype(str).str.strip().str.casefold()
        mismatch = identity[snapshot_column].notna() & left.ne(right)
        if bool(mismatch.any()):
            codes = sorted(identity.loc[mismatch, "player_code"].astype(int).tolist())
            raise ValueError(
                f"Strict GW1 player usage has stale {column} for player codes {codes}"
            )

    matches = _strict_integer_series(checked, "team_preseason_matches")
    appearances = _strict_integer_series(checked, "preseason_appearances")
    starts = _strict_integer_series(checked, "preseason_starts")
    for column in ("preseason_minutes", "preseason_last_two_minutes", "source_confidence"):
        values = pd.to_numeric(checked[column], errors="coerce")
        invalid = checked[column].notna() & (values.isna() | ~np.isfinite(values))
        if bool(invalid.any()):
            raise ValueError(f"Strict GW1 player usage has invalid {column}")
        if column != "source_confidence" and bool((values.dropna() < 0).any()):
            raise ValueError(f"Strict GW1 player usage has negative {column}")
    impossible_counts = (
        (starts.notna() & appearances.notna() & starts.gt(appearances))
        | (appearances.notna() & matches.notna() & appearances.gt(matches))
    )
    if bool(impossible_counts.any()):
        raise ValueError("Strict GW1 player usage has starts/appearances above team matches")
    minutes = pd.to_numeric(checked["preseason_minutes"], errors="coerce")
    recent_minutes = pd.to_numeric(
        checked["preseason_last_two_minutes"], errors="coerce"
    )
    if bool((recent_minutes.notna() & minutes.notna() & recent_minutes.gt(minutes)).any()):
        raise ValueError("Strict GW1 player usage has recent minutes above total minutes")
    confidence = pd.to_numeric(checked["source_confidence"], errors="coerce")
    if bool(((confidence.dropna() < 0) | (confidence.dropna() > 1)).any()):
        raise ValueError("Strict GW1 player usage has source_confidence outside [0, 1]")

    derived = derive_preseason_player_priors(checked)
    meaningful = (
        matches.ge(float(STRICT_GW1_MIN_PRESEASON_MATCHES))
        & confidence.ge(float(STRICT_GW1_MIN_SOURCE_CONFIDENCE))
        & pd.to_numeric(derived["preseason_evidence_components"], errors="coerce").ge(2)
        & pd.to_numeric(derived["preseason_prior_quality"], errors="coerce").ge(
            float(STRICT_GW1_MIN_PLAYER_EVIDENCE_QUALITY)
        )
        & pd.to_numeric(derived["prior_p_start"], errors="coerce").between(0, 1)
        & pd.to_numeric(derived["prior_p_appearance"], errors="coerce").between(0, 1)
        & pd.to_numeric(derived["prior_p_appearance"], errors="coerce").ge(
            pd.to_numeric(derived["prior_p_start"], errors="coerce")
        )
    )
    coverage = expected[["player_code", "team_code"]].copy()
    coverage["meaningful"] = coverage["player_code"].map(
        dict(zip(derived["player_code"].astype(int), meaningful.astype(bool)))
    ).fillna(False)
    selectable = pd.Series(True, index=expected.index)
    if "can_select" in expected.columns:
        selectable &= expected["can_select"].fillna(True).astype(bool)
    if "removed" in expected.columns:
        selectable &= ~expected["removed"].fillna(False).astype(bool)
    coverage["selectable"] = selectable.to_numpy(dtype=bool)
    counts = (
        coverage[coverage["selectable"]]
        .groupby("team_code")["meaningful"]
        .sum()
        .astype(int)
    )
    team_codes = sorted(expected["team_code"].unique().tolist())
    minimum = int(STRICT_GW1_MIN_EVIDENCED_PLAYERS_PER_TEAM)
    shortfalls = [
        f"team {team_code}: {int(counts.get(team_code, 0))}/{minimum}"
        for team_code in team_codes
        if int(counts.get(team_code, 0)) < minimum
    ]
    if shortfalls:
        raise ValueError(
            "Strict GW1 player usage lacks meaningful club coverage: "
            + ", ".join(shortfalls)
        )


def load_preseason_snapshot_context(
    catalog: SnapshotCatalog,
    *,
    season_name: str,
    cutoff: datetime | str,
    mode: str = "auto",
    required_topics: tuple[str, ...] = (),
    target_gameweek: int | None = None,
    expected_team_codes: set[int] | None = None,
    expected_players: pd.DataFrame | None = None,
    expected_teams: pd.DataFrame | None = None,
) -> PreseasonSnapshotContext:
    """Resolve every optional snapshot without allowing post-cutoff evidence.

    ``auto`` uses the latest eligible snapshot and tolerates absent topics,
    ``strict`` requires ``required_topics``, and ``off`` disables enrichment.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"auto", "strict", "off"}:
        raise ValueError("snapshot mode must be one of: auto, strict, off")
    unknown_required = set(required_topics) - set(SNAPSHOT_TOPICS)
    if unknown_required:
        raise ValueError(f"Unknown required snapshot topics: {sorted(unknown_required)}")
    if normalized_mode == "off":
        return PreseasonSnapshotContext()

    frames: dict[str, pd.DataFrame] = {}
    paths: list[Path] = []
    manifests: list[dict[str, Any]] = []
    manifests_by_topic: dict[str, dict[str, Any]] = {}
    for topic in SNAPSHOT_TOPICS:
        if topic == "provider_expected_points":
            frame, path, manifest = _resolve_provider_expected_points(
                catalog,
                season_name=season_name,
                cutoff=cutoff,
                required=normalized_mode == "strict" and topic in required_topics,
                target_gameweek=target_gameweek,
            )
            if path is None or manifest is None or frame.empty:
                continue
        elif topic == "preseason_role_evidence":
            frame, path, manifest = _resolve_preseason_role_evidence(
                catalog,
                season_name=season_name,
                cutoff=cutoff,
                required=normalized_mode == "strict" and topic in required_topics,
                target_gameweek=target_gameweek,
            )
            if path is None or manifest is None or frame.empty:
                continue
        elif topic == "tournament_workload":
            frame, path, manifest = _resolve_tournament_workload(
                catalog,
                season_name=season_name,
                cutoff=cutoff,
                required=normalized_mode == "strict" and topic in required_topics,
                target_gameweek=target_gameweek,
            )
            if path is None or manifest is None or frame.empty:
                continue
        else:
            path = catalog.resolve(
                season_name=season_name,
                topic=topic,
                cutoff=cutoff,
                required=normalized_mode == "strict" and topic in required_topics,
            )
            if path is None:
                continue
            frame = catalog.load_frame(path)
            manifest = catalog.read_manifest(path)
        if (
            normalized_mode == "strict"
            and topic in required_topics
            and not manifest.get("manifest_envelope_sha256")
        ):
            raise ValueError(
                f"Strict snapshot {topic!r} is missing manifest integrity metadata"
            )
        observed_column = {
            "preseason_role_evidence": "role_last_observed_at",
            "tournament_workload": "last_tournament_appearance_at",
        }.get(topic)
        if observed_column is not None:
            _validate_observation_cutoff(
                frame,
                column=observed_column,
                topic=topic,
                manifest=manifest,
                cutoff=cutoff,
            )
        provenance = f"{manifest.get('provider')}:{manifest.get('captured_at')}"
        provenance_column = {
            "preseason_player_usage": "preseason_data_source",
            "preseason_role_evidence": "role_data_source",
            "tournament_workload": "workload_data_source",
            "provider_expected_points": "provider_expected_points_source",
            "fixture_odds": "fixture_odds_source",
            "team_priors": "team_prior_snapshot_source",
        }[topic]
        # Manifest provenance is authoritative; input CSVs cannot override it.
        frame[provenance_column] = provenance
        if topic == "preseason_player_usage":
            if (
                "readiness_multiplier" in frame.columns
                and frame["readiness_multiplier"].notna().any()
            ):
                raise ValueError(
                    "preseason_player_usage cannot supply readiness_multiplier; "
                    "use a validated tournament_workload snapshot"
                )
            frame = frame.drop(columns=["readiness_multiplier"], errors="ignore")
            if any(
                column in frame.columns
                for column in (
                    "team_preseason_matches",
                    "preseason_appearances",
                    "preseason_starts",
                    "preseason_minutes",
                    "preseason_last_two_minutes",
                )
            ):
                frame = derive_preseason_player_priors(frame)
        frames[topic] = frame
        paths.append(path)
        manifests.append(manifest)
        manifests_by_topic[topic] = manifest

    player_enrichment = _merge_player_frames(
        [
            ("preseason_player_usage", frames.get("preseason_player_usage", pd.DataFrame())),
            ("preseason_role_evidence", frames.get("preseason_role_evidence", pd.DataFrame())),
            ("tournament_workload", frames.get("tournament_workload", pd.DataFrame())),
            ("provider_expected_points", frames.get("provider_expected_points", pd.DataFrame())),
        ]
    )
    context = PreseasonSnapshotContext(
        player_enrichment=player_enrichment,
        team_priors=frames.get("team_priors", pd.DataFrame()),
        fixture_odds=frames.get("fixture_odds", pd.DataFrame()),
        snapshot_paths=tuple(paths),
        manifests=tuple(manifests),
    )
    if normalized_mode == "strict" and target_gameweek == 1:
        _validate_strict_gw1_context(
            context,
            expected_team_codes=expected_team_codes,
            raw_usage=frames.get("preseason_player_usage", pd.DataFrame()),
            usage_manifest=manifests_by_topic.get("preseason_player_usage"),
            cutoff=cutoff,
            expected_players=expected_players,
            expected_teams=expected_teams,
            team_prior_manifest=manifests_by_topic.get("team_priors"),
            season_name=season_name,
        )
    return context


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    # Counts and minutes cannot be negative.  Treat malformed negative values as
    # unknown rather than turning them into evidence against the player.
    return values.mask(values < 0)


def _validated_player_codes(frame: pd.DataFrame, *, topic: str) -> pd.Series:
    if "player_code" not in frame.columns:
        raise ValueError(f"{topic} must contain player_code")
    numeric = pd.to_numeric(frame["player_code"], errors="coerce")
    if (
        numeric.isna().any()
        or numeric.duplicated().any()
        or numeric.mod(1).ne(0).any()
    ):
        raise ValueError(f"{topic} player_code must be unique and integral")
    return numeric.astype(int)


def derive_preseason_role_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    """Validate role observations and derive one bounded directional signal.

    Component deltas are normalized editorial judgements in ``[-1, 1]``. They
    describe change from the player's established baseline, not absolute role,
    which avoids rewarding duties already represented in historical results.
    """
    if not isinstance(evidence, pd.DataFrame):
        raise TypeError("evidence must be a pandas DataFrame")
    if evidence.empty:
        return evidence.copy()
    result = evidence.copy()
    result["player_code"] = _validated_player_codes(
        result, topic="preseason_role_evidence"
    )
    for column in ROLE_COMPONENT_WEIGHTS:
        if column not in result.columns:
            result[column] = np.nan
    populated = result[list(ROLE_COMPONENT_WEIGHTS)].notna().any(axis=1)
    result = result[populated].copy()
    if result.empty:
        raise ValueError("preseason_role_evidence has no populated role observations")

    for column in ROLE_COMPONENT_WEIGHTS:
        values = pd.to_numeric(result[column], errors="coerce")
        invalid = result[column].notna() & (
            values.isna() | ~np.isfinite(values) | values.lt(-1) | values.gt(1)
        )
        if bool(invalid.any()):
            raise ValueError(f"{column} must contain finite values in [-1, 1]")
        result[column] = values

    required = {
        "role_target_gameweek",
        "role_evidence_matches",
        "role_evidence_confidence",
        "role_last_observed_at",
    }
    missing = required - set(result.columns)
    if missing:
        raise ValueError(
            "preseason_role_evidence is missing columns: "
            + ", ".join(sorted(missing))
        )
    gameweek = pd.to_numeric(result["role_target_gameweek"], errors="coerce")
    matches = pd.to_numeric(result["role_evidence_matches"], errors="coerce")
    confidence = pd.to_numeric(result["role_evidence_confidence"], errors="coerce")
    if (
        gameweek.isna().any()
        or ~np.isfinite(gameweek).all()
        or gameweek.lt(1).any()
        or gameweek.mod(1).ne(0).any()
    ):
        raise ValueError("role_target_gameweek must contain positive integers")
    if (
        matches.isna().any()
        or ~np.isfinite(matches).all()
        or matches.lt(1).any()
        or matches.mod(1).ne(0).any()
    ):
        raise ValueError("role_evidence_matches must contain positive integers")
    if (
        confidence.isna().any()
        or ~np.isfinite(confidence).all()
        or confidence.lt(0).any()
        or confidence.gt(1).any()
    ):
        raise ValueError("role_evidence_confidence must be in [0, 1]")
    observed_at = pd.to_datetime(
        result["role_last_observed_at"], errors="coerce", utc=True
    )
    if observed_at.isna().any():
        raise ValueError("role_last_observed_at must contain valid timestamps")

    result["role_target_gameweek"] = gameweek.astype(int)
    result["role_evidence_matches"] = matches.astype(int)
    result["role_evidence_confidence"] = confidence.astype(float)
    components = result[list(ROLE_COMPONENT_WEIGHTS)]
    score = sum(
        components[column].fillna(0.0) * float(weight)
        for column, weight in ROLE_COMPONENT_WEIGHTS.items()
    )
    result["role_change_score"] = score.clip(-1.0, 1.0)
    result["role_evidence_components"] = components.notna().sum(axis=1).astype(int)
    depth = (matches / 3.0).clip(0.0, 1.0)
    result["role_evidence_quality"] = (confidence * depth).clip(0.0, 1.0)
    return result.reset_index(drop=True)


def normalize_tournament_workload(workload: pd.DataFrame) -> pd.DataFrame:
    """Validate an auditable workload snapshot without inventing readiness."""
    if not isinstance(workload, pd.DataFrame):
        raise TypeError("workload must be a pandas DataFrame")
    if workload.empty:
        return workload.copy()
    result = workload.copy()
    result["player_code"] = _validated_player_codes(
        result, topic="tournament_workload"
    )
    evidence_columns = (
        "tournament_minutes",
        "tournament_matches",
        "tournament_starts",
        "last_tournament_appearance_at",
        "readiness_multiplier",
        "workload_evidence_confidence",
    )
    populated_columns = [column for column in evidence_columns if column in result.columns]
    if not populated_columns:
        raise ValueError("tournament_workload has no workload evidence columns")
    result = result[result[populated_columns].notna().any(axis=1)].copy()
    if result.empty:
        raise ValueError("tournament_workload has no populated workload observations")
    required = {
        "tournament_minutes",
        "tournament_matches",
        "tournament_starts",
        "last_tournament_appearance_at",
        "readiness_multiplier",
        "workload_evidence_confidence",
        "workload_valid_through_gameweek",
    }
    missing = required - set(result.columns)
    if missing:
        raise ValueError(
            "tournament_workload is missing columns: " + ", ".join(sorted(missing))
        )
    minutes = pd.to_numeric(result["tournament_minutes"], errors="coerce")
    matches = pd.to_numeric(result["tournament_matches"], errors="coerce")
    starts = pd.to_numeric(result["tournament_starts"], errors="coerce")
    readiness = pd.to_numeric(result["readiness_multiplier"], errors="coerce")
    confidence = pd.to_numeric(
        result["workload_evidence_confidence"], errors="coerce"
    )
    if minutes.isna().any() or ~np.isfinite(minutes).all() or minutes.lt(0).any():
        raise ValueError("tournament_minutes must be finite and non-negative")
    for column, values in (
        ("tournament_matches", matches),
        ("tournament_starts", starts),
    ):
        if (
            values.isna().any()
            or ~np.isfinite(values).all()
            or values.lt(0).any()
            or values.mod(1).ne(0).any()
        ):
            raise ValueError(f"{column} must contain non-negative integers")
    if starts.gt(matches).any():
        raise ValueError("tournament_starts cannot exceed tournament_matches")
    if (
        readiness.isna().any()
        or ~np.isfinite(readiness).all()
        or readiness.lt(0).any()
        or readiness.gt(1).any()
    ):
        raise ValueError("readiness_multiplier must be in [0, 1]")
    if (
        confidence.isna().any()
        or ~np.isfinite(confidence).all()
        or confidence.lt(0).any()
        or confidence.gt(1).any()
    ):
        raise ValueError("workload_evidence_confidence must be in [0, 1]")
    observed_at = pd.to_datetime(
        result["last_tournament_appearance_at"], errors="coerce", utc=True
    )
    if observed_at.isna().any():
        raise ValueError("last_tournament_appearance_at must contain valid timestamps")
    valid_through = pd.to_numeric(
        result["workload_valid_through_gameweek"], errors="coerce"
    )
    if (
        valid_through.isna().any()
        or ~np.isfinite(valid_through).all()
        or valid_through.lt(1).any()
        or valid_through.mod(1).ne(0).any()
    ):
        raise ValueError(
            "workload_valid_through_gameweek must contain positive integers"
        )
    result["workload_valid_through_gameweek"] = valid_through.astype(int)
    result["tournament_minutes"] = minutes.astype(float)
    result["tournament_matches"] = matches.astype(int)
    result["tournament_starts"] = starts.astype(int)
    result["readiness_multiplier"] = readiness.astype(float)
    result["workload_evidence_confidence"] = confidence.astype(float)
    return result.reset_index(drop=True)


def _safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid_denominator = denominator.where(denominator > 0)
    return (numerator / valid_denominator).replace([np.inf, -np.inf], np.nan).clip(0, 1)


def _weighted_available(
    components: Mapping[str, pd.Series],
    weights: Mapping[str, float],
) -> pd.Series:
    """Weighted row mean that excludes unknown components from the denominator."""
    if not components:
        raise ValueError("At least one prior component is required")
    index = next(iter(components.values())).index
    numerator = pd.Series(0.0, index=index)
    denominator = pd.Series(0.0, index=index)
    for name, values in components.items():
        weight = float(weights[name])
        known = values.notna()
        numerator = numerator.add(values.fillna(0.0) * weight)
        denominator = denominator.add(known.astype(float) * weight)
    return numerator.div(denominator.where(denominator > 0)).clip(0, 1)


def derive_preseason_player_priors(usage: pd.DataFrame) -> pd.DataFrame:
    """Derive bounded start and appearance priors from friendly usage.

    Parameters
    ----------
    usage:
        One row per player. ``player_code`` is the stable FPL player code.  The
        supported evidence columns are listed in :data:`USAGE_COLUMNS`; absent or
        null evidence is excluded from the calculation rather than filled with
        zero.

    Returns
    -------
    pandas.DataFrame
        A player-keyed frame containing the priors, evidence quality, and the
        component shares used to calculate them.  ``preseason_prior_quality`` is
        evidence confidence, not a measure of whether the observed usage was
        positive.
    """
    if not isinstance(usage, pd.DataFrame):
        raise TypeError("usage must be a pandas DataFrame")
    if "player_code" not in usage.columns:
        raise ValueError("usage must contain player_code")
    if usage["player_code"].isna().any():
        raise ValueError("player_code must not be null")
    if usage["player_code"].duplicated().any():
        raise ValueError("usage must contain at most one row per player_code")

    matches = _numeric(usage, "team_preseason_matches")
    appearances = _numeric(usage, "preseason_appearances")
    starts = _numeric(usage, "preseason_starts")
    minutes = _numeric(usage, "preseason_minutes")
    recent_minutes = _numeric(usage, "preseason_last_two_minutes")
    source_confidence = _numeric(usage, "source_confidence").clip(0, 1)

    appearance_share = _safe_share(appearances, matches)
    start_share = _safe_share(starts, matches)
    minutes_share = _safe_share(minutes, matches * 90.0)
    # This field explicitly represents the final two friendlies.  When a club has
    # played only one match, compare against one match rather than inventing a
    # second zero-minute appearance.
    recent_match_count = matches.clip(upper=2)
    recent_minutes_share = _safe_share(recent_minutes, recent_match_count * 90.0)

    prior_start = _weighted_available(
        {
            "starts": start_share,
            "minutes": minutes_share,
            "recent_minutes": recent_minutes_share,
        },
        {"starts": 0.45, "minutes": 0.25, "recent_minutes": 0.30},
    )
    prior_appearance = _weighted_available(
        {
            "appearances": appearance_share,
            "minutes": minutes_share,
            "recent_minutes": recent_minutes_share,
        },
        {"appearances": 0.55, "minutes": 0.20, "recent_minutes": 0.25},
    )
    # Starting implies appearing.  Preserve NaN when both quantities are unknown.
    prior_appearance = pd.concat(
        [prior_appearance, prior_start], axis=1
    ).max(axis=1, skipna=True)
    no_appearance_evidence = prior_appearance.isna() & prior_start.isna()
    prior_appearance = prior_appearance.mask(no_appearance_evidence)

    evidence_components = pd.concat(
        [appearance_share, start_share, minutes_share, recent_minutes_share], axis=1
    )
    evidence_components.columns = [
        "appearance_share",
        "start_share",
        "minutes_share",
        "recent_minutes_share",
    ]
    completeness = evidence_components.notna().mean(axis=1)
    depth = (matches / 4.0).clip(0, 1)
    # Unknown source confidence does not count as a negative observation.  It does
    # reduce the audit quality through an explicit 50% provenance default.
    provenance_quality = source_confidence.fillna(0.5)
    quality = (completeness * depth.fillna(0.0) * provenance_quality).clip(0, 1)

    # Preserve the source snapshot intact so workload, role, and provenance fields
    # remain available to prediction outputs.  Derived fields are overwritten to
    # prevent a stale prior already present in the source from surviving a rebuild.
    result = usage.copy()
    result["prior_p_start"] = prior_start
    result["prior_p_appearance"] = prior_appearance
    result["preseason_prior_quality"] = quality
    result["preseason_appearance_share"] = appearance_share
    result["preseason_start_share"] = start_share
    result["preseason_minutes_share"] = minutes_share
    result["preseason_recent_minutes_share"] = recent_minutes_share
    result["preseason_evidence_components"] = (
        evidence_components.notna().sum(axis=1).astype(int)
    )
    result["preseason_evidence_depth"] = depth
    return result.reset_index(drop=True)
