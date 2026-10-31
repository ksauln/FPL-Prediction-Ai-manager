#!/usr/bin/env python3
"""Build conservative preseason usage from selected positive-minute evidence.

The importer deliberately does not discover matches or infer zero-minute rows.
Every club must have exactly two explicitly selected evidence matches.  FPL-Core
Insights supplies positive-minute rows only when a selected club side clears a
minimum completeness gate; exact first-party rows may replace or supplement
those observations.  Players with no positive observation remain blank in the
full live-roster output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fplmodel.config import (  # noqa: E402
    STRICT_GW1_MIN_EVIDENCED_PLAYERS_PER_TEAM,
    STRICT_GW1_MIN_PLAYER_EVIDENCE_QUALITY,
    STRICT_GW1_MIN_PRESEASON_MATCHES,
    STRICT_GW1_MIN_SOURCE_CONFIDENCE,
)
from fplmodel.preseason import derive_preseason_player_priors  # noqa: E402
from fplmodel.snapshot_catalog import _safe_source_url  # noqa: E402


DEFAULT_CORE_SOURCE_URL = (
    "https://github.com/olbauday/FPL-Core-Insights/tree/"
    "1196637fe5e6dda981a7a1f4ac48c89b118affde/"
    "data/2026-2027/By%20Tournament/Friendlies/GW0"
)
SELECTION_COLUMNS = {
    "team_code",
    "evidence_match_id",
    "core_match_id",
    "played_at",
    "finished",
    "official_complete",
    "absence_safe",
}
SUPPLEMENT_COLUMNS = {
    "player_code",
    "team_name",
    "evidence_match_id",
    "played_at",
    "minutes",
    "source_url",
}
CORE_PLAYER_COLUMNS = {"player_id", "player_code", "team_code"}
CORE_MATCH_COLUMNS = {
    "match_id",
    "kickoff_time",
    "home_team",
    "away_team",
    "finished",
}
CORE_STAT_COLUMNS = {"player_id", "match_id", "minutes_played"}
SOURCE_CONFIDENCE = 0.75
MIN_SIDE_PLAYERS = 10
MIN_SIDE_MINUTES = 810.0
MAX_SIDE_MINUTES = 1320.0
MAX_PLAYER_MINUTES = 120.0
MIN_OFFICIAL_PLAYERS = 8
MIN_PARTIAL_OFFICIAL_MINUTES = 450.0
MIN_COMPLETE_OFFICIAL_MINUTES = 600.0


def _require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def _integral(values: pd.Series, *, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all() or numeric.mod(1).ne(0).any():
        raise ValueError(f"{label} must contain finite integers")
    return numeric.astype(int)


def _nonempty_text(values: pd.Series, *, label: str) -> pd.Series:
    text = values.fillna("").astype(str).str.strip()
    if text.eq("").any():
        raise ValueError(f"{label} must not be blank")
    return text


def _finished(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _boolean_series(values: pd.Series, *, label: str) -> pd.Series:
    normalized = values.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().casefold()
    )
    mapping = {True: True, False: False, "true": True, "false": False, "1": True, "0": False}
    result = normalized.map(mapping)
    if result.isna().any():
        raise ValueError(f"{label} must contain booleans")
    return result.astype(bool)


def _load_live_roster(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with path.open("r", encoding="utf-8") as handle:
        bootstrap = json.load(handle)
    team_rows = []
    teams_by_id: dict[int, dict[str, object]] = {}
    for raw in bootstrap.get("teams", []):
        team_id = int(raw["id"])
        row = {
            "team_id": team_id,
            "team_code": int(raw["code"]),
            "team_name": str(raw["name"]).strip(),
        }
        team_rows.append(row)
        teams_by_id[team_id] = row
    teams = pd.DataFrame(team_rows)
    if teams.empty or teams["team_code"].duplicated().any() or teams["team_name"].duplicated().any():
        raise ValueError("Bootstrap teams must have unique team codes and names")

    player_rows = []
    for raw in bootstrap.get("elements", []):
        team = teams_by_id.get(int(raw["team"]))
        if team is None:
            raise ValueError("Bootstrap player references an unknown team")
        can_select = raw.get("can_select", True)
        if can_select is None:
            can_select = True
        player_rows.append(
            {
                "player_code": int(raw["code"]),
                "player_id": int(raw["id"]),
                "full_name": f"{raw['first_name']} {raw['second_name']}",
                "team_name": team["team_name"],
                "team_code": team["team_code"],
                "can_select": bool(can_select),
                "removed": bool(raw.get("removed", False)),
            }
        )
    players = pd.DataFrame(player_rows)
    if players.empty:
        raise ValueError("Bootstrap contains no players")
    if players["player_code"].duplicated().any() or players["player_id"].duplicated().any():
        raise ValueError("Bootstrap players must have unique ids and codes")
    return teams, players


def _load_selection(path: Path, teams: pd.DataFrame, *, cutoff: object) -> pd.DataFrame:
    selection = pd.read_csv(path, dtype={"evidence_match_id": "string", "core_match_id": "string"})
    _require_columns(selection, SELECTION_COLUMNS, label="Match selection")
    selection = selection.copy()
    selection["team_code"] = _integral(selection["team_code"], label="selection team_code")
    selection["evidence_match_id"] = _nonempty_text(
        selection["evidence_match_id"], label="selection evidence_match_id"
    )
    selection["core_match_id"] = selection["core_match_id"].fillna("").astype(str).str.strip()
    selection["finished"] = _boolean_series(
        selection["finished"], label="selection finished"
    )
    if not bool(selection["finished"].all()):
        raise ValueError("Every selected evidence match must be explicitly finished")
    selection["official_complete"] = _boolean_series(
        selection["official_complete"], label="selection official_complete"
    )
    selection["absence_safe"] = _boolean_series(
        selection["absence_safe"], label="selection absence_safe"
    )
    selection["played_at_parsed"] = pd.to_datetime(
        selection["played_at"], errors="coerce", utc=True
    )
    if selection["played_at_parsed"].isna().any():
        raise ValueError("selection played_at must contain valid timestamps")
    cutoff_at = pd.to_datetime(cutoff, errors="coerce", utc=True)
    if pd.isna(cutoff_at):
        raise ValueError("cutoff must be a valid timestamp")
    if bool(selection["played_at_parsed"].gt(cutoff_at).any()):
        raise ValueError("Selected evidence must be played on or before the cutoff")
    if selection.duplicated(["team_code", "evidence_match_id"]).any():
        raise ValueError("Match selection contains duplicate club/evidence-match rows")
    populated_core = selection[selection["core_match_id"].ne("")]
    if populated_core.duplicated(["team_code", "core_match_id"]).any():
        raise ValueError("Match selection contains duplicate club/Core-match rows")
    if bool((selection["official_complete"] & ~selection["absence_safe"]).any()):
        raise ValueError("official_complete selections must be absence_safe")
    core_not_safe = selection["core_match_id"].ne("") & ~selection["absence_safe"]
    if bool(core_not_safe.any()):
        raise ValueError("selected Core sides must be absence_safe after the completeness gate")
    unsafe_without_official = (
        selection["core_match_id"].eq("")
        & selection["absence_safe"]
        & ~selection["official_complete"]
    )
    if bool(unsafe_without_official.any()):
        raise ValueError("absence-safe non-Core selections must be official_complete")

    expected_codes = set(teams["team_code"].astype(int))
    actual_codes = set(selection["team_code"].astype(int))
    if actual_codes != expected_codes:
        raise ValueError(
            "Match selection must cover every live club exactly; "
            f"missing={sorted(expected_codes - actual_codes)}, "
            f"extra={sorted(actual_codes - expected_codes)}"
        )
    counts = selection.groupby("team_code").size()
    wrong = counts[counts.ne(2)]
    if not wrong.empty:
        details = ", ".join(f"{int(code)}={int(count)}" for code, count in wrong.items())
        raise ValueError(f"Every club must select exactly two evidence matches: {details}")
    selection = selection.merge(
        teams[["team_code", "team_name"]], on="team_code", how="left", validate="many_to_one"
    )
    return selection


def _load_core_observations(
    core_dir: Path,
    selection: pd.DataFrame,
    live_players: pd.DataFrame,
    *,
    source_url: str,
) -> tuple[pd.DataFrame, set[tuple[int, str]], set[tuple[int, str]]]:
    core_players = pd.read_csv(core_dir / "players.csv")
    core_matches = pd.read_csv(core_dir / "matches.csv")
    core_stats = pd.read_csv(core_dir / "playermatchstats.csv")
    _require_columns(core_players, CORE_PLAYER_COLUMNS, label="Core players")
    _require_columns(core_matches, CORE_MATCH_COLUMNS, label="Core matches")
    _require_columns(core_stats, CORE_STAT_COLUMNS, label="Core player-match stats")

    if core_players["player_id"].duplicated().any() or core_players["player_code"].duplicated().any():
        raise ValueError("Core player ids and codes must be unique")
    if core_matches["match_id"].duplicated().any():
        raise ValueError("Core match ids must be unique")
    if core_stats.duplicated(["player_id", "match_id"]).any():
        raise ValueError("Core player-match rows must be unique")

    core_players = core_players.copy()
    core_players["player_id"] = _integral(core_players["player_id"], label="Core player_id")
    core_players["player_code"] = _integral(core_players["player_code"], label="Core player_code")
    core_players["team_code"] = _integral(core_players["team_code"], label="Core team_code")
    core_matches = core_matches.copy()
    core_matches["match_id"] = _nonempty_text(core_matches["match_id"], label="Core match_id")
    core_stats = core_stats.copy()
    core_stats["player_id"] = _integral(core_stats["player_id"], label="Core stat player_id")
    core_stats["match_id"] = _nonempty_text(core_stats["match_id"], label="Core stat match_id")

    live_identity = live_players[["player_code", "team_code"]].rename(
        columns={"team_code": "live_team_code"}
    )
    rows: list[pd.DataFrame] = []
    discarded: set[tuple[int, str]] = set()
    accepted: set[tuple[int, str]] = set()
    for selected in selection.itertuples(index=False):
        core_match_id = str(selected.core_match_id).strip()
        if not core_match_id:
            continue
        match = core_matches[core_matches["match_id"].eq(core_match_id)]
        if len(match) != 1:
            raise ValueError(f"Selected Core match does not exist exactly once: {core_match_id}")
        match_row = match.iloc[0]
        if not _finished(match_row["finished"]):
            raise ValueError(f"Selected Core match is not finished: {core_match_id}")
        kickoff_at = pd.to_datetime(match_row["kickoff_time"], errors="coerce", utc=True)
        if pd.isna(kickoff_at) or kickoff_at.date() != selected.played_at_parsed.date():
            raise ValueError(
                f"Selected played_at date does not match Core match {core_match_id}"
            )
        participants = pd.to_numeric(
            pd.Series([match_row["home_team"], match_row["away_team"]]), errors="coerce"
        ).dropna().astype(int)
        if int(selected.team_code) not in set(participants):
            raise ValueError(
                f"Selected club {int(selected.team_code)} is not a participant in {core_match_id}"
            )

        side = core_stats[core_stats["match_id"].eq(core_match_id)].merge(
            core_players[["player_id", "player_code", "team_code"]],
            on="player_id",
            how="inner",
            validate="many_to_one",
        )
        side = side[side["team_code"].eq(int(selected.team_code))].merge(
            live_identity, on="player_code", how="inner", validate="many_to_one"
        )
        side = side[side["live_team_code"].eq(int(selected.team_code))].copy()
        minutes = pd.to_numeric(side["minutes_played"], errors="coerce")
        side = side[minutes.gt(0) & minutes.le(MAX_PLAYER_MINUTES) & np.isfinite(minutes)].copy()
        side["minutes"] = pd.to_numeric(side["minutes_played"], errors="coerce")
        if side["player_code"].duplicated().any():
            raise ValueError(f"Core side has duplicate player codes: {core_match_id}")

        player_count = int(side["player_code"].nunique())
        minute_total = float(side["minutes"].sum())
        key = (int(selected.team_code), str(selected.evidence_match_id))
        if (
            player_count < MIN_SIDE_PLAYERS
            or minute_total < MIN_SIDE_MINUTES
            or minute_total > MAX_SIDE_MINUTES
        ):
            discarded.add(key)
            continue
        accepted.add(key)
        side["team_code"] = int(selected.team_code)
        side["evidence_match_id"] = str(selected.evidence_match_id)
        side["played_at"] = selected.played_at_parsed
        side["source_url"] = source_url
        rows.append(
            side[
                [
                    "player_code",
                    "team_code",
                    "evidence_match_id",
                    "played_at",
                    "minutes",
                    "source_url",
                ]
            ]
        )
    columns = [
        "player_code",
        "team_code",
        "evidence_match_id",
        "played_at",
        "minutes",
        "source_url",
    ]
    observations = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    return observations, accepted, discarded


def _load_supplements(
    path: Path,
    selection: pd.DataFrame,
    live_players: pd.DataFrame,
    *,
    cutoff: object,
) -> pd.DataFrame:
    supplements = pd.read_csv(path)
    _require_columns(supplements, SUPPLEMENT_COLUMNS, label="Official supplements")
    if supplements.empty:
        return pd.DataFrame(
            columns=[
                "player_code",
                "team_code",
                "evidence_match_id",
                "played_at",
                "minutes",
                "source_url",
            ]
        )
    supplements = supplements.copy()
    supplements["player_code"] = _integral(
        supplements["player_code"], label="supplement player_code"
    )
    supplements["team_name"] = _nonempty_text(
        supplements["team_name"], label="supplement team_name"
    )
    supplements["evidence_match_id"] = _nonempty_text(
        supplements["evidence_match_id"], label="supplement evidence_match_id"
    )
    minutes = pd.to_numeric(supplements["minutes"], errors="coerce")
    if (
        minutes.isna().any()
        or not np.isfinite(minutes).all()
        or minutes.mod(1).ne(0).any()
        or minutes.le(0).any()
        or minutes.gt(MAX_PLAYER_MINUTES).any()
    ):
        raise ValueError("supplement minutes must be integral and in [1, 120]")
    supplements["minutes"] = minutes.astype(int)
    supplements["played_at"] = pd.to_datetime(
        supplements["played_at"], errors="coerce", utc=True
    )
    if supplements["played_at"].isna().any():
        raise ValueError("supplement played_at must contain valid timestamps")
    cutoff_at = pd.to_datetime(cutoff, errors="coerce", utc=True)
    if pd.isna(cutoff_at):
        raise ValueError("cutoff must be a valid timestamp")
    if bool(supplements["played_at"].gt(cutoff_at).any()):
        raise ValueError("Official supplement evidence must be on or before the cutoff")
    supplements["source_url"] = supplements["source_url"].map(_safe_source_url)

    team_names = selection[["team_name", "team_code"]].drop_duplicates()
    supplements = supplements.merge(
        team_names, on="team_name", how="left", validate="many_to_one"
    )
    if supplements["team_code"].isna().any():
        unknown = sorted(supplements.loc[supplements["team_code"].isna(), "team_name"].unique())
        raise ValueError(f"supplement team_name does not match the live bootstrap: {unknown}")
    supplements["team_code"] = supplements["team_code"].astype(int)
    if supplements.duplicated(["team_code", "evidence_match_id", "player_code"]).any():
        raise ValueError("Official supplements contain duplicate player-match rows")

    selection_keys = selection[
        ["team_code", "evidence_match_id", "played_at_parsed"]
    ].rename(columns={"played_at_parsed": "selected_played_at"})
    supplements = supplements.merge(
        selection_keys,
        on=["team_code", "evidence_match_id"],
        how="left",
        validate="many_to_one",
    )
    if supplements["selected_played_at"].isna().any():
        bad = supplements.loc[
            supplements["selected_played_at"].isna(), ["team_code", "evidence_match_id"]
        ].drop_duplicates()
        raise ValueError(f"Official supplement references an unselected side: {bad.to_dict('records')}")
    date_mismatch = supplements["played_at"].dt.date.ne(
        supplements["selected_played_at"].dt.date
    )
    if date_mismatch.any():
        raise ValueError("supplement played_at date does not match the selected evidence match")

    live_identity = live_players[["player_code", "team_code"]].rename(
        columns={"team_code": "live_team_code"}
    )
    supplements = supplements.merge(
        live_identity, on="player_code", how="left", validate="many_to_one"
    )
    if supplements["live_team_code"].isna().any():
        unknown_codes = sorted(
            supplements.loc[supplements["live_team_code"].isna(), "player_code"].astype(int).unique()
        )
        raise ValueError(f"supplement player_code is absent from the live roster: {unknown_codes}")
    mismatch = supplements["live_team_code"].astype(int).ne(supplements["team_code"])
    if mismatch.any():
        bad_codes = sorted(supplements.loc[mismatch, "player_code"].astype(int).unique())
        raise ValueError(f"supplement player belongs to a different live club: {bad_codes}")
    return supplements[
        [
            "player_code",
            "team_code",
            "evidence_match_id",
            "played_at",
            "minutes",
            "source_url",
        ]
    ].copy()


def _merge_observations(
    core: pd.DataFrame,
    supplements: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    accepted_core_sides: set[tuple[int, str]],
    discarded_core_sides: set[tuple[int, str]],
) -> pd.DataFrame:
    key_columns = ["team_code", "evidence_match_id", "player_code"]
    complete_keys = selection[selection["official_complete"]][
        ["team_code", "evidence_match_id"]
    ].drop_duplicates()
    if not complete_keys.empty and not core.empty:
        core = core.merge(complete_keys.assign(_complete_official=True), on=["team_code", "evidence_match_id"], how="left")
        core = core[core["_complete_official"].isna()].drop(columns="_complete_official")
    if not supplements.empty and not core.empty:
        official_keys = supplements[key_columns].drop_duplicates().assign(_official=True)
        core = core.merge(official_keys, on=key_columns, how="left")
        core = core[core["_official"].isna()].drop(columns="_official")
    observations = pd.concat([core, supplements], ignore_index=True)
    if observations.duplicated(key_columns).any():
        raise ValueError("Merged evidence contains duplicate player-match rows")

    failures = []
    for selected in selection.itertuples(index=False):
        key = (int(selected.team_code), str(selected.evidence_match_id))
        side = observations[
            observations["team_code"].eq(key[0])
            & observations["evidence_match_id"].eq(key[1])
        ]
        player_count = int(side["player_code"].nunique())
        minute_total = float(pd.to_numeric(side["minutes"], errors="coerce").sum())
        official_side = supplements[
            supplements["team_code"].eq(key[0])
            & supplements["evidence_match_id"].eq(key[1])
        ]
        official_players = int(official_side["player_code"].nunique())
        official_minutes = float(pd.to_numeric(official_side["minutes"], errors="coerce").sum())
        if bool(selected.official_complete):
            valid = (
                official_players >= MIN_OFFICIAL_PLAYERS
                and MIN_COMPLETE_OFFICIAL_MINUTES
                <= official_minutes
                <= MAX_SIDE_MINUTES
            )
        elif str(selected.core_match_id).strip():
            valid = (
                key in accepted_core_sides
                and player_count >= MIN_SIDE_PLAYERS
                and MIN_SIDE_MINUTES <= minute_total <= MAX_SIDE_MINUTES
            )
        else:
            valid = (
                official_players >= MIN_OFFICIAL_PLAYERS
                and MIN_PARTIAL_OFFICIAL_MINUTES
                <= official_minutes
                <= MAX_SIDE_MINUTES
            )
        if not valid:
            source_detail = (
                " (Core side discarded before supplement merge)"
                if key in discarded_core_sides
                else ""
            )
            failures.append(
                f"{selected.team_name}/{key[1]}={player_count} players,{minute_total:g} minutes"
                f"{source_detail}"
            )
    if failures:
        raise ValueError(
            "selected evidence side is not credible: " + "; ".join(failures)
        )
    return observations


def _source_list(values: Iterable[object]) -> str:
    return "; ".join(sorted({str(value) for value in values if str(value).strip()}))


def _validate_meaningful_coverage(usage: pd.DataFrame, live_players: pd.DataFrame) -> None:
    derived = derive_preseason_player_priors(usage)
    matches = pd.to_numeric(derived["team_preseason_matches"], errors="coerce")
    confidence = pd.to_numeric(derived["source_confidence"], errors="coerce")
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
    coverage = live_players[["player_code", "team_code", "team_name", "can_select", "removed"]].copy()
    coverage["meaningful"] = coverage["player_code"].map(
        dict(zip(derived["player_code"].astype(int), meaningful.astype(bool)))
    ).fillna(False)
    coverage = coverage[coverage["can_select"] & ~coverage["removed"]]
    counts = coverage.groupby(["team_code", "team_name"])["meaningful"].sum().astype(int)
    minimum = int(STRICT_GW1_MIN_EVIDENCED_PLAYERS_PER_TEAM)
    shortfalls = [
        f"{name} {int(count)}/{minimum}"
        for (_, name), count in counts.items()
        if int(count) < minimum
    ]
    if shortfalls:
        raise ValueError("meaningful club coverage is below strict GW1 minimum: " + ", ".join(shortfalls))


def build_preseason_usage(
    *,
    bootstrap_path: Path,
    core_dir: Path,
    selection_path: Path,
    supplement_path: Path,
    cutoff: object,
    core_source_url: str = DEFAULT_CORE_SOURCE_URL,
) -> pd.DataFrame:
    """Return a full live-roster usage frame from explicitly selected evidence."""
    core_source_url = _safe_source_url(core_source_url)
    teams, live_players = _load_live_roster(Path(bootstrap_path))
    selection = _load_selection(Path(selection_path), teams, cutoff=cutoff)
    core, accepted_core, discarded = _load_core_observations(
        Path(core_dir), selection, live_players, source_url=core_source_url
    )
    supplements = _load_supplements(
        Path(supplement_path), selection, live_players, cutoff=cutoff
    )
    observations = _merge_observations(
        core,
        supplements,
        selection,
        accepted_core_sides=accepted_core,
        discarded_core_sides=discarded,
    )

    selected_safety = selection[
        ["team_code", "evidence_match_id", "absence_safe"]
    ].drop_duplicates()
    observations = observations.merge(
        selected_safety,
        on=["team_code", "evidence_match_id"],
        how="left",
        validate="many_to_one",
    )
    safe_match_counts = (
        selection[selection["absence_safe"]]
        .groupby("team_code")["evidence_match_id"]
        .nunique()
        .to_dict()
    )
    unsafe_positive_counts = (
        observations[~observations["absence_safe"]]
        .groupby(["player_code", "team_code"])["evidence_match_id"]
        .nunique()
        .to_dict()
    )

    aggregate = (
        observations.groupby(["player_code", "team_code"], as_index=False)
        .agg(
            preseason_appearances=("evidence_match_id", "nunique"),
            preseason_minutes=("minutes", "sum"),
            evidence_matches=("evidence_match_id", _source_list),
            evidence_sources=("source_url", _source_list),
        )
    )
    aggregate["team_preseason_matches"] = aggregate.apply(
        lambda row: min(
            2,
            int(safe_match_counts.get(int(row["team_code"]), 0))
            + int(
                unsafe_positive_counts.get(
                    (int(row["player_code"]), int(row["team_code"])), 0
                )
            ),
        ),
        axis=1,
    )
    aggregate["preseason_starts"] = pd.NA
    aggregate["preseason_last_two_minutes"] = aggregate["preseason_minutes"]
    aggregate["source_confidence"] = SOURCE_CONFIDENCE
    aggregate["notes"] = aggregate.apply(
        lambda row: (
            "Positive-minute observations only; selected matches: "
            f"{row['evidence_matches']}; sources: {row['evidence_sources']}"
        ),
        axis=1,
    )
    aggregate = aggregate.drop(columns=["evidence_matches", "evidence_sources"])

    output = live_players[
        ["player_code", "player_id", "full_name", "team_name", "team_code"]
    ].merge(
        aggregate,
        on=["player_code", "team_code"],
        how="left",
        validate="one_to_one",
    )
    output = output[
        [
            "player_code",
            "player_id",
            "full_name",
            "team_name",
            "team_code",
            "team_preseason_matches",
            "preseason_appearances",
            "preseason_starts",
            "preseason_minutes",
            "preseason_last_two_minutes",
            "source_confidence",
            "notes",
        ]
    ].sort_values(["team_name", "full_name"], ignore_index=True)
    _validate_meaningful_coverage(output, live_players)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--core-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--supplements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", required=True, help="Latest allowed evidence timestamp")
    parser.add_argument("--core-source-url", default=DEFAULT_CORE_SOURCE_URL)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    usage = build_preseason_usage(
        bootstrap_path=args.bootstrap,
        core_dir=args.core_dir,
        selection_path=args.selection,
        supplement_path=args.supplements,
        cutoff=args.cutoff,
        core_source_url=args.core_source_url,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    usage.to_csv(args.output, index=False)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
