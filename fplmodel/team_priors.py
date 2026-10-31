"""Prior-season team form features from Football-Data-style result files."""

from __future__ import annotations

from collections.abc import Mapping
import re
import unicodedata

import numpy as np
import pandas as pd


# These are explicit source-to-FPL name contracts, not fuzzy matches.  Callers can
# extend or replace individual entries through ``team_aliases``.
DEFAULT_TEAM_ALIASES: dict[str, str] = {
    "Birmingham": "Birmingham City",
    "Blackburn": "Blackburn Rovers",
    "Brighton": "Brighton and Hove Albion",
    "Cardiff": "Cardiff City",
    "Coventry": "Coventry City",
    "Derby": "Derby County",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Leeds United": "Leeds",
    "Leicester": "Leicester City",
    "Man City": "Manchester City",
    "Man United": "Man Utd",
    "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Norwich": "Norwich City",
    "Nottingham Forest": "Nott'm Forest",
    "Oxford": "Oxford United",
    "Preston": "Preston North End",
    "QPR": "Queens Park Rangers",
    "Sheffield United": "Sheffield United",
    "Sheffield Weds": "Sheffield Wednesday",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
    "Tottenham": "Spurs",
    "Tottenham Hotspur": "Spurs",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}

_RESULT_COLUMNS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
_TEAM_STATS = (
    "team_goals_for",
    "team_goals_against",
    "team_goal_diff",
    "team_clean_sheet",
    "team_conceded_two_plus",
    "team_match_points",
)


def _normalise_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _empty_team_prior_frame(window: int) -> pd.DataFrame:
    feature_columns = [f"{stat}_ma{window}" for stat in _TEAM_STATS]
    return pd.DataFrame(
        columns=[
            "team_id",
            "team_code",
            "team_name",
            *feature_columns,
            "team_prior_matches",
            "team_prior_as_of",
            "team_prior_source",
            "team_prior_league",
        ]
    )


def _current_team_lookup(
    teams: pd.DataFrame,
    aliases: Mapping[str, str | int],
) -> tuple[dict[str, dict[str, object]], dict[int, dict[str, object]], dict[str, str | int]]:
    name_column = "name" if "name" in teams.columns else "team_name"
    code_column = "code" if "code" in teams.columns else "team_code"
    if name_column not in teams.columns or code_column not in teams.columns:
        raise ValueError("teams must contain name/team_name and code/team_code")
    id_column = "team_id" if "team_id" in teams.columns else "id" if "id" in teams.columns else None

    current_by_name: dict[str, dict[str, object]] = {}
    current_by_code: dict[int, dict[str, object]] = {}
    for row in teams.to_dict(orient="records"):
        name = str(row[name_column]).strip()
        code_value = pd.to_numeric(pd.Series([row[code_column]]), errors="coerce").iloc[0]
        if not name or pd.isna(code_value):
            continue
        code = int(code_value)
        key = _normalise_name(name)
        if key in current_by_name:
            raise ValueError(f"Duplicate current team name after normalisation: {name!r}")
        if code in current_by_code:
            raise ValueError(f"Duplicate current team code: {code}")
        team_id: int | float = np.nan
        if id_column is not None:
            numeric_id = pd.to_numeric(pd.Series([row.get(id_column)]), errors="coerce").iloc[0]
            if pd.notna(numeric_id):
                team_id = int(numeric_id)
        record: dict[str, object] = {
            "team_id": team_id,
            "team_code": code,
            "team_name": name,
        }
        current_by_name[key] = record
        current_by_code[code] = record

    alias_lookup = {_normalise_name(source): target for source, target in aliases.items()}
    return current_by_name, current_by_code, alias_lookup


def _resolve_team(
    source_name: object,
    *,
    current_by_name: Mapping[str, dict[str, object]],
    current_by_code: Mapping[int, dict[str, object]],
    aliases: Mapping[str, str | int],
) -> dict[str, object] | None:
    source_key = _normalise_name(source_name)
    direct = current_by_name.get(source_key)
    if direct is not None:
        return direct
    target = aliases.get(source_key)
    if target is None:
        return None
    if isinstance(target, (int, np.integer)):
        return current_by_code.get(int(target))
    # Numeric strings are accepted as stable codes only when there is no current
    # team whose actual name is that string.
    target_key = _normalise_name(target)
    named = current_by_name.get(target_key)
    if named is not None:
        return named
    try:
        return current_by_code.get(int(str(target).strip()))
    except ValueError:
        return None


def build_team_priors_from_results(
    results: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    source_league: str,
    source_label: str,
    window: int = 5,
    team_aliases: Mapping[str, str | int] | None = None,
) -> pd.DataFrame:
    """Build last-``window`` form priors for current teams from match results.

    ``results`` follows the Football-Data CSV contract (``Date``, ``HomeTeam``,
    ``AwayTeam``, ``FTHG`` and ``FTAG``).  Only exact current names and explicit
    aliases are joined.  Other clubs remain useful as opponents in the source
    matches but are not emitted as current-team priors.
    """
    if not isinstance(results, pd.DataFrame) or not isinstance(teams, pd.DataFrame):
        raise TypeError("results and teams must be pandas DataFrames")
    missing = _RESULT_COLUMNS.difference(results.columns)
    if missing:
        raise ValueError(
            "results is missing Football-Data columns: " + ", ".join(sorted(missing))
        )
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("window must be a positive integer")
    league = str(source_league).strip()
    label = str(source_label).strip()
    if not league or not label:
        raise ValueError("source_league and source_label must not be empty")

    aliases: dict[str, str | int] = dict(DEFAULT_TEAM_ALIASES)
    if team_aliases:
        aliases.update(team_aliases)
    current_by_name, current_by_code, alias_lookup = _current_team_lookup(teams, aliases)
    if not current_by_name:
        return _empty_team_prior_frame(window)

    matches = results[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    matches["match_date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")
    matches["home_goals"] = pd.to_numeric(matches["FTHG"], errors="coerce")
    matches["away_goals"] = pd.to_numeric(matches["FTAG"], errors="coerce")
    matches["source_order"] = np.arange(len(matches), dtype=int)
    matches = matches.dropna(
        subset=["match_date", "HomeTeam", "AwayTeam", "home_goals", "away_goals"]
    )
    if matches.empty:
        return _empty_team_prior_frame(window)

    home = pd.DataFrame(
        {
            "source_team_name": matches["HomeTeam"],
            "match_date": matches["match_date"],
            "source_order": matches["source_order"],
            "team_goals_for": matches["home_goals"],
            "team_goals_against": matches["away_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "source_team_name": matches["AwayTeam"],
            "match_date": matches["match_date"],
            "source_order": matches["source_order"],
            "team_goals_for": matches["away_goals"],
            "team_goals_against": matches["home_goals"],
        }
    )
    long_results = pd.concat([home, away], ignore_index=True)
    long_results["team_goal_diff"] = (
        long_results["team_goals_for"] - long_results["team_goals_against"]
    )
    long_results["team_clean_sheet"] = (
        long_results["team_goals_against"] == 0
    ).astype(float)
    long_results["team_conceded_two_plus"] = (
        long_results["team_goals_against"] >= 2
    ).astype(float)
    long_results["team_match_points"] = np.select(
        [
            long_results["team_goals_for"] > long_results["team_goals_against"],
            long_results["team_goals_for"] == long_results["team_goals_against"],
        ],
        [3.0, 1.0],
        default=0.0,
    )
    long_results["current_team"] = long_results["source_team_name"].map(
        lambda value: _resolve_team(
            value,
            current_by_name=current_by_name,
            current_by_code=current_by_code,
            aliases=alias_lookup,
        )
    )
    long_results = long_results[long_results["current_team"].notna()].copy()
    if long_results.empty:
        return _empty_team_prior_frame(window)
    long_results["team_code"] = long_results["current_team"].map(
        lambda team: int(team["team_code"])
    )
    long_results = long_results.sort_values(
        ["team_code", "match_date", "source_order"], kind="mergesort"
    )

    rows: list[dict[str, object]] = []
    provenance = label if label.rsplit(":", 1)[-1] == league else f"{label}:{league}"
    for team_code, team_results in long_results.groupby("team_code", sort=True):
        recent = team_results.tail(window)
        current = current_by_code[int(team_code)]
        row: dict[str, object] = dict(current)
        for stat in _TEAM_STATS:
            row[f"{stat}_ma{window}"] = float(
                pd.to_numeric(recent[stat], errors="coerce").mean()
            )
        row.update(
            {
                "team_prior_matches": int(len(recent)),
                "team_prior_as_of": recent["match_date"].max().date().isoformat(),
                "team_prior_source": provenance,
                "team_prior_league": league,
            }
        )
        rows.append(row)

    columns = list(_empty_team_prior_frame(window).columns)
    return pd.DataFrame(rows).reindex(columns=columns).reset_index(drop=True)
