from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import re
from typing import Dict, Any, Tuple, Sequence, Iterable

from .config import ROLLING_WINDOWS, MIN_MATCHES_FOR_FEATURES
from .state import ModelState


SET_PIECE_ORDER_MAP = {
    "corners_and_indirect_freekicks_order": "corners",
    "direct_freekicks_order": "direct_fk",
    "penalties_order": "penalty",
}

_TEMP_SEASON_SORT_COL = "__season_sort__"
_TEMP_KICKOFF_SORT_COL = "__kickoff_sort__"

TEAM_CONTEXT_STATS = (
    "team_goals_for",
    "team_goals_against",
    "team_goal_diff",
    "team_clean_sheet",
    "team_conceded_two_plus",
    "team_match_points",
)


def _season_start_year(label: Any) -> int | None:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    if isinstance(label, (int, np.integer)):
        return int(label)
    match = re.search(r"(\d{4})", str(label))
    if match:
        return int(match.group(1))
    return None


def _default_season_year_for_month(month: int, season_start_year: int | None) -> int | None:
    if season_start_year is None:
        return None
    return season_start_year if month >= 7 else season_start_year + 1


def _infer_status_return_year(
    *,
    month: int,
    day: int,
    news_added: Any,
    season_start_year: int | None,
) -> int | None:
    news_added_ts = pd.to_datetime(news_added, errors="coerce", utc=True)
    if pd.notna(news_added_ts):
        candidate_year = int(news_added_ts.year)
        if (month, day) < (int(news_added_ts.month), int(news_added_ts.day)):
            candidate_year += 1
        if season_start_year is None:
            return candidate_year
        if candidate_year in {season_start_year, season_start_year + 1}:
            return candidate_year
    return _default_season_year_for_month(month, season_start_year)


def _parse_status_return_date(
    news: Any,
    *,
    news_added: Any,
    season_start_year: int | None,
) -> pd.Timestamp:
    if news is None or (isinstance(news, float) and np.isnan(news)):
        return pd.NaT
    text = str(news).strip()
    match = re.search(
        r"(?:^|[-,;]\s*|\b)(Expected back|Suspended until)\s+(\d{1,2})\s+([A-Za-z]{3,9})(?:$|\b)",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return pd.NaT
    day = int(match.group(2))
    month_text = match.group(3)
    month = pd.to_datetime(month_text, format="%b", errors="coerce")
    if pd.isna(month):
        month = pd.to_datetime(month_text, format="%B", errors="coerce")
    if pd.isna(month):
        return pd.NaT
    year = _infer_status_return_year(
        month=int(month.month),
        day=day,
        news_added=news_added,
        season_start_year=season_start_year,
    )
    if year is None:
        return pd.NaT
    parsed = pd.to_datetime(
        f"{year:04d}-{int(month.month):02d}-{day:02d}",
        errors="coerce",
        utc=True,
    )
    if pd.isna(parsed):
        return pd.NaT
    return parsed.tz_localize(None).normalize()


def _add_history_sort_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporary chronological sort keys that work across multi-season history."""
    out = df.copy()
    if "season_name" in out.columns:
        season_sort = out["season_name"].map(_season_start_year)
        if season_sort.notna().any():
            fallback = float(season_sort.max(skipna=True)) + 1.0
            out[_TEMP_SEASON_SORT_COL] = season_sort.fillna(fallback)
        else:
            out[_TEMP_SEASON_SORT_COL] = 0.0
    else:
        out[_TEMP_SEASON_SORT_COL] = 0.0

    if "kickoff_time" in out.columns:
        out[_TEMP_KICKOFF_SORT_COL] = pd.to_datetime(out["kickoff_time"], errors="coerce")
    else:
        out[_TEMP_KICKOFF_SORT_COL] = pd.NaT
    return out


def _history_sort_columns(df: pd.DataFrame, leading_cols: Sequence[str]) -> list[str]:
    cols = [col for col in leading_cols if col in df.columns]
    for col in (_TEMP_SEASON_SORT_COL, "round", _TEMP_KICKOFF_SORT_COL, "fixture"):
        if col in df.columns and col not in cols:
            cols.append(col)
    return cols


def _current_season_mask(df: pd.DataFrame) -> pd.Series:
    if "season_name" not in df.columns:
        return pd.Series(True, index=df.index)
    season_sort = df["season_name"].map(_season_start_year)
    if not season_sort.notna().any():
        return pd.Series(True, index=df.index)
    current_start = season_sort.max(skipna=True)
    return season_sort == current_start


def _season_mask(df: pd.DataFrame, current_season_name: str | None) -> pd.Series:
    if current_season_name is not None and "season_name" in df.columns:
        return df["season_name"].astype(str).eq(str(current_season_name))
    return _current_season_mask(df)


def _target_fixture_context(
    histories: pd.DataFrame,
    fixtures_df: pd.DataFrame | None,
    *,
    next_gw: int,
    last_finished_gw: int,
    current_season: pd.Series,
    windows: Iterable[int],
    teams_df: pd.DataFrame | None = None,
    team_priors_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build leakage-safe team/opponent form for the fixtures in ``next_gw``.

    Current-season results are used only through ``last_finished_gw``. Early in
    a season they are shrunk toward the most recent stable-code prior, sourced
    from prior-season history or an optional promoted-team snapshot.
    """
    columns = ["team_id", "fixture_count", "was_home", "fixture_difficulty", "target_fixture_kickoff"]
    if fixtures_df is None or fixtures_df.empty or "event" not in fixtures_df.columns:
        return pd.DataFrame(columns=columns)

    target = fixtures_df[pd.to_numeric(fixtures_df["event"], errors="coerce") == int(next_gw)]
    if target.empty:
        return pd.DataFrame(columns=columns)

    stats = list(TEAM_CONTEXT_STATS)
    rounds = pd.to_numeric(histories.get("round"), errors="coerce")
    completed = (
        histories[current_season & rounds.le(int(last_finished_gw))].copy()
        if len(histories)
        else histories.copy()
    )
    required = {"team", "fixture", "was_home", "team_h_score", "team_a_score"}
    team_code_by_id: dict[int, int] = {}
    if teams_df is not None and {"team_id", "code"}.issubset(teams_df.columns):
        valid_teams = teams_df[["team_id", "code"]].dropna().copy()
        team_code_by_id = {
            int(row.team_id): int(row.code)
            for row in valid_teams.itertuples(index=False)
        }

    def add_match_stats(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not required.issubset(frame.columns):
            return pd.DataFrame()
        out = frame.copy()
        was_home = out["was_home"].astype(bool)
        out["team_goals_for"] = np.where(
            was_home, out["team_h_score"], out["team_a_score"]
        )
        out["team_goals_against"] = np.where(
            was_home, out["team_a_score"], out["team_h_score"]
        )
        out["team_goal_diff"] = out["team_goals_for"] - out["team_goals_against"]
        out["team_clean_sheet"] = (out["team_goals_against"] == 0).astype(int)
        out["team_conceded_two_plus"] = (out["team_goals_against"] >= 2).astype(int)
        out["team_match_points"] = np.select(
            [
                out["team_goals_for"] > out["team_goals_against"],
                out["team_goals_for"] == out["team_goals_against"],
            ],
            [3, 1],
            default=0,
        )
        return out

    def stable_team_key(frame: pd.DataFrame, *, current: bool) -> pd.Series:
        if "team_code" in frame.columns:
            key = pd.to_numeric(frame["team_code"], errors="coerce")
        else:
            key = pd.Series(np.nan, index=frame.index, dtype=float)
        if current and team_code_by_id:
            mapped = pd.to_numeric(frame["team"], errors="coerce").map(team_code_by_id)
            key = key.fillna(mapped)
        return key.fillna(pd.to_numeric(frame["team"], errors="coerce"))

    def aggregate_latest(
        frame: pd.DataFrame,
        *,
        current: bool,
    ) -> tuple[dict[int, dict[str, float]], dict[int, dict[int, int]]]:
        values_by_team: dict[int, dict[str, float]] = {}
        counts_by_team: dict[int, dict[int, int]] = {}
        enriched = add_match_stats(frame)
        if enriched.empty:
            return values_by_team, counts_by_team
        enriched["__team_key__"] = stable_team_key(enriched, current=current)
        enriched = enriched.dropna(subset=["__team_key__"])
        if enriched.empty:
            return values_by_team, counts_by_team
        team_matches = enriched.drop_duplicates(["__team_key__", "fixture"]).copy()
        team_matches = _add_history_sort_keys(team_matches)
        team_matches = team_matches.sort_values(
            _history_sort_columns(team_matches, ["__team_key__"]), kind="mergesort"
        )
        for team_key, group in team_matches.groupby("__team_key__"):
            values: dict[str, float] = {}
            counts: dict[int, int] = {}
            for window in windows:
                tail = group.tail(int(window))
                counts[int(window)] = len(tail)
                for stat in stats:
                    values[f"{stat}_ma{window}"] = float(
                        pd.to_numeric(tail[stat], errors="coerce").mean()
                    )
            values_by_team[int(team_key)] = values
            counts_by_team[int(team_key)] = counts
        return values_by_team, counts_by_team

    current_values, current_counts = aggregate_latest(completed, current=True)

    prior_rows = histories[~current_season].copy() if len(histories) else histories.copy()
    prior_source_label = "history"
    if not prior_rows.empty and "season_name" in prior_rows.columns:
        prior_rows = _add_history_sort_keys(prior_rows)
        latest_prior_start = prior_rows[_TEMP_SEASON_SORT_COL].max(skipna=True)
        if pd.notna(latest_prior_start):
            prior_rows = prior_rows[prior_rows[_TEMP_SEASON_SORT_COL].eq(latest_prior_start)]
        season_labels = prior_rows["season_name"].dropna().astype(str).unique()
        if len(season_labels):
            prior_source_label = f"history:{season_labels[-1]}"
        prior_rows = prior_rows.drop(
            columns=[_TEMP_SEASON_SORT_COL, _TEMP_KICKOFF_SORT_COL], errors="ignore"
        )
    prior_values, _ = aggregate_latest(prior_rows, current=False)
    prior_sources = {team_key: prior_source_label for team_key in prior_values}

    if team_priors_df is not None and not team_priors_df.empty:
        snapshot = team_priors_df.copy()
        if "team_code" not in snapshot.columns:
            raise ValueError("team_priors_df must include team_code")
        for record in snapshot.to_dict(orient="records"):
            team_code = pd.to_numeric(pd.Series([record.get("team_code")]), errors="coerce").iloc[0]
            if pd.isna(team_code):
                continue
            key = int(team_code)
            # Prior-season PL history is preferred for incumbent clubs. The
            # snapshot fills promoted/new clubs that have no stable-code row.
            if key in prior_values:
                continue
            values = {
                f"{stat}_ma{window}": float(record[f"{stat}_ma{window}"])
                for window in windows
                for stat in stats
                if pd.notna(record.get(f"{stat}_ma{window}"))
            }
            if values:
                prior_values[key] = values
                prior_sources[key] = str(record.get("team_prior_source") or "snapshot")

    feature_names = [f"{stat}_ma{window}" for window in windows for stat in stats]
    neutral_values: dict[str, float] = {}
    for feature in feature_names:
        available = [values.get(feature) for values in prior_values.values()]
        available = [float(value) for value in available if value is not None and pd.notna(value)]
        neutral_values[feature] = float(np.mean(available)) if available else 0.0

    max_window = max((int(window) for window in windows), default=1)

    def context_for(team_key: int) -> tuple[dict[str, float], str, float]:
        current = current_values.get(team_key, {})
        counts = current_counts.get(team_key, {})
        prior = prior_values.get(team_key, {})
        prior_available = bool(prior) or bool(prior_values)
        values: dict[str, float] = {}
        used_prior = False
        largest_count = 0
        for window in windows:
            count = int(counts.get(int(window), 0))
            largest_count = max(largest_count, count)
            for stat in stats:
                feature = f"{stat}_ma{window}"
                current_value = current.get(feature)
                prior_value = prior.get(feature, neutral_values.get(feature, 0.0))
                if current_value is None or pd.isna(current_value):
                    values[feature] = float(prior_value)
                    used_prior = True
                elif (
                    count < int(window)
                    and prior_available
                    and prior_value is not None
                    and pd.notna(prior_value)
                ):
                    values[feature] = float(
                        (float(current_value) * count + float(prior_value) * (int(window) - count))
                        / float(int(window))
                    )
                    used_prior = True
                else:
                    values[feature] = float(current_value)
        prior_weight = max(0.0, 1.0 - (largest_count / float(max_window)))
        if largest_count >= max_window:
            source = "current_season"
        elif largest_count and used_prior:
            source = f"current+{prior_sources.get(team_key, 'neutral')}"
        elif largest_count:
            source = "current_season"
        else:
            source = prior_sources.get(team_key, "neutral")
        return values, source, prior_weight

    fixture_map: dict[int, list[tuple[bool, int, float]]] = {}
    for fixture in target.to_dict(orient="records"):
        home = int(fixture["team_h"])
        away = int(fixture["team_a"])
        home_difficulty = pd.to_numeric(
            pd.Series([fixture.get("team_h_difficulty")]),
            errors="coerce",
        ).iloc[0]
        away_difficulty = pd.to_numeric(
            pd.Series([fixture.get("team_a_difficulty")]),
            errors="coerce",
        ).iloc[0]
        fixture_map.setdefault(home, []).append(
            (True, away, float(home_difficulty) if pd.notna(home_difficulty) else 3.0)
        )
        fixture_map.setdefault(away, []).append(
            (False, home, float(away_difficulty) if pd.notna(away_difficulty) else 3.0)
        )

    rows: list[dict[str, float | int]] = []
    for team_id, fixtures in fixture_map.items():
        team_key = team_code_by_id.get(team_id, team_id)
        own, own_source, own_prior_weight = context_for(team_key)
        row: dict[str, float | int] = {
            "team_id": team_id,
            "fixture_count": len(fixtures),
            "was_home": float(
                np.mean([1.0 if is_home else 0.0 for is_home, _, _ in fixtures])
            ),
            "fixture_difficulty": float(
                np.mean([difficulty for _, _, difficulty in fixtures])
            ),
            "team_prior_source": own_source,
            "team_context_prior_weight": own_prior_weight,
        }
        fixture_rows = target[
            pd.to_numeric(target["team_h"], errors="coerce").eq(team_id)
            | pd.to_numeric(target["team_a"], errors="coerce").eq(team_id)
        ]
        if "kickoff_time" in fixture_rows.columns:
            kickoff_values = pd.to_datetime(
                fixture_rows["kickoff_time"],
                errors="coerce",
                utc=True,
            ).dt.tz_localize(None)
            row["target_fixture_kickoff"] = kickoff_values.min() if kickoff_values.notna().any() else pd.NaT
        else:
            row["target_fixture_kickoff"] = pd.NaT
        for feature in feature_names:
            row[feature] = float(own.get(feature, 0.0))
            opponent_contexts = [
                context_for(team_code_by_id.get(opponent, opponent))
                for _, opponent, _ in fixtures
            ]
            opponent_values = [context[0].get(feature, 0.0) for context in opponent_contexts]
            opponent_feature = feature.replace("team_", "opp_team_", 1)
            row[opponent_feature] = float(np.mean(opponent_values)) if opponent_values else 0.0
        opponent_sources = [context[1] for context in opponent_contexts]
        row["opponent_team_prior_source"] = " / ".join(dict.fromkeys(opponent_sources))
        opponent_weights = [context[2] for context in opponent_contexts]
        row["opponent_team_context_prior_weight"] = (
            float(np.mean(opponent_weights)) if opponent_weights else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_player_static_features(
    elements_df: pd.DataFrame,
    *,
    current_season_name: str | None = None,
) -> pd.DataFrame:
    """Enhance elements dataframe with numeric form metrics and set-piece flags."""
    df = elements_df.copy()
    numeric_cols = [
        "form",
        "points_per_game",
        "selected_by_percent",
        "value_form",
        "value_season",
        "expected_goals_per_90",
        "expected_assists_per_90",
        "expected_goal_involvements_per_90",
        "expected_goals_conceded_per_90",
        "goals_conceded_per_90",
        "saves_per_90",
        "starts_per_90",
        "clean_sheets_per_90",
        "defensive_contribution_per_90",
        "ep_next",
        "price_change_percent",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "minutes" in df.columns:
        df["season_minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    else:
        df["season_minutes"] = 0.0

    if "ep_next" in df.columns:
        df["official_ep_next"] = pd.to_numeric(df["ep_next"], errors="coerce")
    else:
        df["official_ep_next"] = np.nan
    if "price_change_percent" not in df.columns:
        df["price_change_percent"] = np.nan

    score_components = []
    for order_col, prefix in SET_PIECE_ORDER_MAP.items():
        if order_col in df.columns:
            df[order_col] = pd.to_numeric(df[order_col], errors="coerce")
            order = df[order_col]
        else:
            order = pd.Series(np.nan, index=df.index, dtype=float)
            df[order_col] = order

        df[f"has_{prefix}_duty"] = (order.fillna(0) > 0).astype(int)
        df[f"primary_{prefix}_taker"] = (order == 1).astype(int)
        inv = 1.0 / order.replace({0: np.nan})
        inv = inv.where(order > 0).fillna(0)
        score_components.append(inv)

    if score_components:
        df["set_piece_duty_score"] = sum(score_components)
    else:
        df["set_piece_duty_score"] = 0.0

    chance_cols = ("chance_of_playing_this_round", "chance_of_playing_next_round")
    for col in chance_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0.0, upper=100.0)
        else:
            df[col] = np.nan

    status_series = df.get("status")
    if status_series is not None:
        status_lower = status_series.fillna("").astype(str).str.lower()
    else:
        status_lower = pd.Series("", index=df.index, dtype=str)

    status_availability_map = {
        "a": 1.0,
        "d": 0.75,
        "i": 0.0,
        "s": 0.0,
        "u": 0.0,
        "n": 0.0,
    }
    df["status_availability"] = status_lower.map(status_availability_map).fillna(0.5)
    df["status_injury_flag"] = status_lower.isin({"d", "i"}).astype(int)
    df["status_persistent_flag"] = status_lower.isin({"u", "n"}).astype(int)
    df["status_code"] = status_lower
    season_start_year = _season_start_year(current_season_name)
    if "news" in df.columns:
        news_series = df["news"]
    else:
        news_series = pd.Series("", index=df.index, dtype=object)
    if "news_added" in df.columns:
        news_added_series = df["news_added"]
    else:
        news_added_series = pd.Series(pd.NaT, index=df.index, dtype=object)
    df["status_return_date"] = [
        _parse_status_return_date(
            news,
            news_added=news_added,
            season_start_year=season_start_year,
        )
        for news, news_added in zip(news_series, news_added_series)
    ]
    df["future_status_applied_flag"] = 0

    df["availability_this_round"] = df["chance_of_playing_this_round"] / 100.0
    df["availability_next_round"] = df["chance_of_playing_next_round"] / 100.0

    chance_risk = (df["chance_of_playing_this_round"] < 100) | (df["chance_of_playing_next_round"] < 100)
    status_risk = status_lower.isin({"d", "i", "s", "u", "n"})
    df["injury_risk_flag"] = (chance_risk | status_risk).astype(int)

    return df

def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator.replace({0: np.nan})
    return ratio.replace([np.inf, -np.inf], np.nan)


def _rolling_mean(group, window: int) -> pd.Series:
    return group.transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())


def _assign_pre_round_features(
    frame: pd.DataFrame,
    *,
    entity_cols: Sequence[str],
    stats: Sequence[str],
    windows: Sequence[int],
) -> pd.DataFrame:
    """Assign leakage-safe lag/rolling features using only prior rounds.

    Every row inside the same season+round receives the same feature values, so
    double-gameweek fixtures cannot consume each other's realized stats.
    """
    if frame.empty or "round" not in frame.columns:
        return frame

    frame = frame.copy()
    work = _add_history_sort_keys(frame)
    work["__orig_pos__"] = np.arange(len(work))
    work["__round_numeric__"] = pd.to_numeric(work["round"], errors="coerce")
    work = work.sort_values(
        _history_sort_columns(work, list(entity_cols)),
        kind="mergesort",
    )

    total_rows = len(work)
    prev_matches = np.zeros(total_rows, dtype=float)
    lag_columns: Dict[str, np.ndarray] = {
        f"{stat}_lag1": np.full(total_rows, np.nan, dtype=float) for stat in stats
    }
    mean_columns: Dict[str, np.ndarray] = {
        f"{stat}_ma{window}": np.full(total_rows, np.nan, dtype=float)
        for stat in stats
        for window in windows
    }

    def _same_round(current: float, previous: float) -> bool:
        if pd.isna(current) and pd.isna(previous):
            return True
        return current == previous

    for _, group in work.groupby(list(entity_cols), sort=False, dropna=False):
        orig_positions = group["__orig_pos__"].to_numpy(dtype=int, copy=False)
        group_size = len(group)
        if group_size == 0:
            continue

        round_values = group["__round_numeric__"].to_numpy(dtype=float, copy=False)
        round_start_mask = np.ones(group_size, dtype=bool)
        for idx in range(1, group_size):
            round_start_mask[idx] = not _same_round(round_values[idx], round_values[idx - 1])

        round_starts = np.flatnonzero(round_start_mask)
        round_lengths = np.diff(np.append(round_starts, group_size))
        prior_counts = np.cumsum(np.concatenate(([0], round_lengths[:-1]))).astype(float)
        repeated_prior_counts = np.repeat(prior_counts, round_lengths)
        prev_matches[orig_positions] = repeated_prior_counts

        for stat in stats:
            stat_values = pd.to_numeric(group[stat], errors="coerce").to_numpy(dtype=float, copy=False)
            filled = pd.Series(stat_values).ffill().to_numpy(dtype=float, copy=False)
            lag_by_round = np.full(len(round_starts), np.nan, dtype=float)
            has_prior = round_starts > 0
            lag_by_round[has_prior] = filled[round_starts[has_prior] - 1]
            lag_columns[f"{stat}_lag1"][orig_positions] = np.repeat(lag_by_round, round_lengths)

            valid_mask = ~np.isnan(stat_values)
            prefix_sum = np.concatenate(([0.0], np.cumsum(np.where(valid_mask, stat_values, 0.0), dtype=float)))
            prefix_count = np.concatenate(([0], np.cumsum(valid_mask.astype(np.int64), dtype=np.int64)))

            for window in windows:
                start_idx = np.maximum(round_starts - int(window), 0)
                window_sum = prefix_sum[round_starts] - prefix_sum[start_idx]
                window_count = prefix_count[round_starts] - prefix_count[start_idx]
                means = np.divide(
                    window_sum,
                    window_count,
                    out=np.full(len(round_starts), np.nan, dtype=float),
                    where=window_count > 0,
                )
                mean_columns[f"{stat}_ma{window}"][orig_positions] = np.repeat(means, round_lengths)

    frame["prev_matches"] = prev_matches
    for column, values in lag_columns.items():
        frame[column] = values
    for column, values in mean_columns.items():
        frame[column] = values
    return frame


def _add_team_context_features(hist: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    """Create rolling team form features and merge opponent versions."""
    required_cols = {"team_h_score", "team_a_score", "was_home", "team", "fixture"}
    if not required_cols.issubset(hist.columns):
        return hist

    hist = hist.copy()
    was_home = hist["was_home"].astype(bool)
    hist["team_goals_for"] = np.where(was_home, hist["team_h_score"], hist["team_a_score"])
    hist["team_goals_against"] = np.where(was_home, hist["team_a_score"], hist["team_h_score"])
    hist["team_goal_diff"] = hist["team_goals_for"] - hist["team_goals_against"]
    hist["team_clean_sheet"] = (hist["team_goals_against"] == 0).astype(int)
    hist["team_conceded_two_plus"] = (hist["team_goals_against"] >= 2).astype(int)
    hist["team_match_points"] = np.select(
        [hist["team_goals_for"] > hist["team_goals_against"], hist["team_goals_for"] == hist["team_goals_against"]],
        [3, 1],
        default=0,
    )

    team_stats = [
        "team_goals_for",
        "team_goals_against",
        "team_goal_diff",
        "team_clean_sheet",
        "team_conceded_two_plus",
        "team_match_points",
    ]

    match_keys = ["fixture", "team"]
    if "season_name" in hist.columns:
        match_keys.insert(0, "season_name")

    team_match_cols = list(dict.fromkeys(match_keys + ["round", "kickoff_time"] + team_stats))
    team_match_cols = [col for col in team_match_cols if col in hist.columns]
    team_matches = hist[team_match_cols].drop_duplicates(subset=match_keys).copy()
    team_matches = _add_history_sort_keys(team_matches)
    team_matches = team_matches.sort_values(
        _history_sort_columns(team_matches, ["team"]),
        kind="mergesort",
    )
    group_cols = ["team"]
    if "season_name" in team_matches.columns:
        group_cols.append("season_name")
    team_matches = _assign_pre_round_features(
        team_matches,
        entity_cols=group_cols,
        stats=team_stats,
        windows=tuple(windows),
    )

    team_feature_cols = [f"{stat}_ma{w}" for stat in team_stats for w in windows]
    team_feature_cols = [c for c in team_feature_cols if c in team_matches.columns]
    if team_feature_cols:
        team_frame = team_matches[match_keys + team_feature_cols]
        hist = hist.merge(team_frame, on=match_keys, how="left")
        rename_map = {
            col: col.replace("team_", "opp_team_", 1) if col.startswith("team_") else f"opp_{col}"
            for col in team_feature_cols
        }
        opponent_frame = team_frame.rename(columns={"team": "opponent_team", **rename_map})
        opponent_keys = ["fixture", "opponent_team"]
        if "season_name" in opponent_frame.columns and "season_name" in hist.columns:
            opponent_keys.insert(0, "season_name")
        hist = hist.merge(opponent_frame, on=opponent_keys, how="left")

    return hist


def _rolling_feats(hist: pd.DataFrame, windows=(3, 5)) -> pd.DataFrame:
    """Create rolling means for key stats, grouped by player, ordered by round."""
    hist = _add_history_sort_keys(hist)
    hist = hist.sort_values(
        _history_sort_columns(hist, ["player_id"]),
        kind="mergesort",
    )
    group_cols = ["player_id"]
    if "season_name" in hist.columns:
        group_cols.append("season_name")
    group = hist.groupby(group_cols, group_keys=False)

    base_stats = [
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "goals_conceded",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "saves",
        "bonus",
        "bps",
        "clearances_blocks_interceptions",
        "recoveries",
        "tackles",
        "defensive_contribution",
        "starts",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "value",
        "transfers_balance",
        "selected",
        "transfers_in",
        "transfers_out",
    ]

    available_cols = set(hist.columns)
    for col in base_stats:
        if col in available_cols:
            hist[col] = _safe_numeric(hist[col])

    derived_features = []
    if {"goals_scored", "assists"}.issubset(available_cols):
        hist["attacking_returns"] = hist["goals_scored"].fillna(0) + hist["assists"].fillna(0)
        derived_features.append("attacking_returns")
    if {"goals_scored", "expected_goals"}.issubset(available_cols):
        hist["finishing_plus_minus"] = hist["goals_scored"].fillna(0) - hist["expected_goals"].fillna(0)
        hist["finishing_ratio"] = _safe_ratio(hist["goals_scored"].fillna(0), hist["expected_goals"].fillna(0))
        derived_features.extend(["finishing_plus_minus", "finishing_ratio"])
    if {"assists", "expected_assists"}.issubset(available_cols):
        hist["creation_plus_minus"] = hist["assists"].fillna(0) - hist["expected_assists"].fillna(0)
        hist["creation_ratio"] = _safe_ratio(hist["assists"].fillna(0), hist["expected_assists"].fillna(0))
        derived_features.extend(["creation_plus_minus", "creation_ratio"])
    if {"goals_scored", "assists", "expected_goal_involvements"}.issubset(available_cols):
        hist["xgi_plus_minus"] = (
            hist["goals_scored"].fillna(0) + hist["assists"].fillna(0) - hist["expected_goal_involvements"].fillna(0)
        )
        derived_features.append("xgi_plus_minus")
    if {"minutes"}.issubset(available_cols):
        hist["minutes_share"] = hist["minutes"].fillna(0) / 90.0
        derived_features.append("minutes_share")
    if {"minutes", "total_points"}.issubset(available_cols):
        minutes = hist["minutes"].replace({0: np.nan})
        hist["points_per_90"] = (hist["total_points"].fillna(0) * 90.0) / minutes
        hist["points_per_90"] = hist["points_per_90"].replace([np.inf, -np.inf], np.nan)
        derived_features.append("points_per_90")
    if {"tackles", "clearances_blocks_interceptions"}.issubset(available_cols):
        hist["tackles_plus_interceptions"] = hist["tackles"].fillna(0) + hist["clearances_blocks_interceptions"].fillna(0)
        derived_features.append("tackles_plus_interceptions")
    if {"tackles", "recoveries", "clearances_blocks_interceptions"}.issubset(available_cols):
        hist["defensive_actions"] = (
            hist["tackles"].fillna(0)
            + hist["recoveries"].fillna(0)
            + hist["clearances_blocks_interceptions"].fillna(0)
        )
        derived_features.append("defensive_actions")

    stats = [col for col in base_stats + derived_features if col in hist.columns]

    hist = _assign_pre_round_features(
        hist,
        entity_cols=group_cols,
        stats=stats,
        windows=tuple(windows),
    )

    if "was_home" in hist.columns:
        hist["was_home"] = hist["was_home"].astype(int)

    hist["enough_prev"] = hist["prev_matches"] >= MIN_MATCHES_FOR_FEATURES

    hist = _add_team_context_features(hist, windows)
    return hist.drop(columns=[_TEMP_SEASON_SORT_COL, _TEMP_KICKOFF_SORT_COL], errors="ignore")

def _merge_team_strength(hist: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Add team and opponent base strength and recent form features."""
    base_cols = [
        "team_id",
        "strength",
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
        "form",
        "points",
        "position",
        "played",
        "win",
        "draw",
        "loss",
    ]
    available_cols = [c for c in base_cols if c in teams_df.columns]
    teams = teams_df[available_cols].copy()

    rename_map = {
        "strength": "team_strength_overall",
        "strength_overall_home": "team_strength_home",
        "strength_overall_away": "team_strength_away",
        "strength_attack_home": "team_attack_home",
        "strength_attack_away": "team_attack_away",
        "strength_defence_home": "team_def_home",
        "strength_defence_away": "team_def_away",
        "form": "team_form_rating",
        "points": "team_points_table",
        "position": "team_league_position",
        "played": "team_matches_played",
        "win": "team_wins",
        "draw": "team_draws",
        "loss": "team_losses",
    }
    teams = teams.rename(columns=rename_map)
    hist = hist.merge(teams, left_on="team", right_on="team_id", how="left")

    opp_cols = {
        "team_strength_overall": "opp_strength_overall",
        "team_strength_home": "opp_strength_home",
        "team_strength_away": "opp_strength_away",
        "team_attack_home": "opp_attack_home",
        "team_attack_away": "opp_attack_away",
        "team_def_home": "opp_def_home",
        "team_def_away": "opp_def_away",
        "team_form_rating": "opp_form_rating",
        "team_points_table": "opp_points_table",
        "team_league_position": "opp_league_position",
        "team_matches_played": "opp_matches_played",
        "team_wins": "opp_wins",
        "team_draws": "opp_draws",
        "team_losses": "opp_losses",
    }
    opponent = teams.rename(columns=opp_cols)
    hist = hist.merge(opponent, left_on="opponent_team", right_on="team_id", how="left", suffixes=("", "_opp"))
    hist = hist.drop(columns=["team_id_opp"], errors="ignore")
    return hist


def _prediction_feature_snapshot(
    base: pd.DataFrame,
    last_rows: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Create a true one-step-forward player snapshot from completed raw rows.

    Training rows intentionally shift rolling values by one match because each
    row's target is that same match. A next-gameweek prediction has no target
    row, so its lag and moving-average features must be recomputed as of the
    latest completed match rather than copied from that shifted row.
    """
    if last_rows.empty:
        return pd.DataFrame(columns=["player_id", *feature_cols])

    snapshot = last_rows[["player_id", *feature_cols]].copy()
    snapshot = snapshot.drop_duplicates("player_id", keep="last").set_index("player_id")
    sorted_base = _add_history_sort_keys(base)
    sorted_base = sorted_base.sort_values(
        _history_sort_columns(sorted_base, ["player_id"]), kind="mergesort"
    )

    selected_seasons: dict[int, Any] = {}
    if "season_name" in last_rows.columns:
        selected_seasons = (
            last_rows.drop_duplicates("player_id", keep="last")
            .set_index("player_id")["season_name"]
            .to_dict()
        )

    grouped_rows: dict[int, pd.DataFrame] = {}
    for player_id, group in sorted_base.groupby("player_id", sort=False):
        selected = group
        if "season_name" in group.columns and int(player_id) in selected_seasons:
            season = selected_seasons[int(player_id)]
            if pd.isna(season):
                selected = group[group["season_name"].isna()]
            else:
                selected = group[group["season_name"].astype(str).eq(str(season))]
        if not selected.empty:
            grouped_rows[int(player_id)] = selected

    for feature in feature_cols:
        if feature.startswith("team_") or feature.startswith("opp_team_"):
            continue
        raw_col: str | None = None
        window: int | None = None
        if feature.endswith("_lag1"):
            raw_col = feature.removesuffix("_lag1")
        else:
            match = re.fullmatch(r"(.+)_ma(\d+)", feature)
            if match:
                raw_col = match.group(1)
                window = int(match.group(2))
        if raw_col is None or raw_col not in base.columns:
            continue

        values: dict[int, float] = {}
        for player_id, group in grouped_rows.items():
            raw = pd.to_numeric(group[raw_col], errors="coerce")
            if window is None:
                eligible = raw.dropna()
                if not eligible.empty:
                    values[player_id] = float(eligible.iloc[-1])
            else:
                eligible = raw.tail(window)
                value = eligible.mean()
                if pd.notna(value):
                    values[player_id] = float(value)
        if values:
            snapshot[feature] = pd.Series(values)

    return snapshot.reset_index()

def build_training_and_pred_frames(
    elements_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    histories_df: pd.DataFrame,
    next_gw: int,
    last_finished_gw: int,
    state: ModelState,
    fixtures_df: pd.DataFrame | None = None,
    current_season_name: str | None = None,
    official_next_gw: int | None = None,
    preseason_player_df: pd.DataFrame | None = None,
    team_priors_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      X_train (DataFrame), y_train (Series of total_points), X_pred (DataFrame for next_gw),
      train_metadata (season/round info aligned with X_train)
    """
    elements_enhanced = _prepare_player_static_features(
        elements_df,
        current_season_name=current_season_name,
    )
    if "code" in elements_enhanced.columns and "player_code" not in elements_enhanced.columns:
        elements_enhanced["player_code"] = pd.to_numeric(
            elements_enhanced["code"], errors="coerce"
        )
    if preseason_player_df is not None and not preseason_player_df.empty:
        from .preseason import derive_preseason_player_priors

        preseason = derive_preseason_player_priors(preseason_player_df)
        if "player_code" in preseason.columns and "player_code" in elements_enhanced.columns:
            preseason["player_code"] = pd.to_numeric(preseason["player_code"], errors="coerce")
            preseason = preseason.dropna(subset=["player_code"]).drop_duplicates(
                "player_code", keep="last"
            )
            elements_enhanced = elements_enhanced.merge(
                preseason,
                on="player_code",
                how="left",
                suffixes=("", "_preseason"),
            )
        elif "player_id" in preseason.columns:
            elements_enhanced = elements_enhanced.merge(
                preseason.drop_duplicates("player_id", keep="last"),
                on="player_id",
                how="left",
                suffixes=("", "_preseason"),
            )
        else:
            raise ValueError("preseason_player_df must include player_code or player_id")

    if "preseason_prior_quality" not in elements_enhanced.columns:
        elements_enhanced["preseason_prior_quality"] = 0.0
    elements_enhanced["preseason_prior_quality"] = pd.to_numeric(
        elements_enhanced["preseason_prior_quality"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    if "readiness_multiplier" not in elements_enhanced.columns:
        elements_enhanced["readiness_multiplier"] = 1.0
    elements_enhanced["readiness_multiplier"] = pd.to_numeric(
        elements_enhanced["readiness_multiplier"], errors="coerce"
    ).fillna(1.0).clip(0.0, 1.0)

    team_labels = teams_df[["team_id", "name"]].rename(columns={"name": "team_name"})
    # Enrichment snapshots retain team_name as an auditable identity label.
    # Use the current bootstrap team table as the canonical prediction label;
    # otherwise pandas suffixes the two copies and no plain team_name survives.
    elements_with_team = elements_enhanced.drop(
        columns=["team_name"], errors="ignore"
    ).merge(team_labels, on="team_id", how="left")

    # Merge element team ids into histories (histories has 'team')
    static_feature_cols = [
        "player_id",
        "team_id",
        "element_type",
    ]
    merge_cols = [c for c in static_feature_cols if c in elements_enhanced.columns]
    base = histories_df.merge(
        elements_enhanced[merge_cols].rename(columns={"team_id": "team"}),
        on="player_id", how="left",
        suffixes=("", "_current"),
    )

    # Prefer historical team ids when available, but fall back to current squad assignment.
    if "team" not in base.columns and "team_current" in base.columns:
        base = base.rename(columns={"team_current": "team"})
    elif "team_current" in base.columns:
        base["team"] = base["team"].fillna(base["team_current"])
        base = base.drop(columns=["team_current"])

    # External rows retain the player's position in that historical season.
    # Current API rows have no position column, so fill only those gaps from the
    # current bootstrap rather than relabelling old points after a reclassification.
    if "element_type_current" in base.columns:
        if "element_type" not in base.columns:
            base = base.rename(columns={"element_type_current": "element_type"})
        else:
            base["element_type"] = pd.to_numeric(
                base["element_type"],
                errors="coerce",
            ).fillna(
                pd.to_numeric(base["element_type_current"], errors="coerce")
            )
            base = base.drop(columns=["element_type_current"])

    if "fixture_difficulty" not in base.columns and "difficulty" in base.columns:
        base["fixture_difficulty"] = pd.to_numeric(base["difficulty"], errors="coerce")
    elif "fixture_difficulty" in base.columns:
        base["fixture_difficulty"] = pd.to_numeric(
            base["fixture_difficulty"],
            errors="coerce",
        )
    else:
        base["fixture_difficulty"] = np.nan
    base["fixture_difficulty"] = base["fixture_difficulty"].fillna(3.0)

    element_type_numeric = pd.to_numeric(base.get("element_type"), errors="coerce")
    for code, label in ((1, "gk"), (2, "def"), (3, "mid"), (4, "fwd")):
        base[f"position_{label}"] = (element_type_numeric == code).astype(float)

    base = _rolling_feats(base, windows=tuple(ROLLING_WINDOWS))

    rolling_feature_cols = [
        c
        for c in base.columns
        if any(c.endswith(f"_ma{w}") for w in ROLLING_WINDOWS) or c.endswith("_lag1")
    ]
    # Snapshot fields from bootstrap are current information and must not be
    # copied onto old training targets. Keep the model inputs historical.
    manual_features = [
        "was_home",
        "fixture_difficulty",
        "position_gk",
        "position_def",
        "position_mid",
        "position_fwd",
    ]
    manual_feature_cols = [c for c in manual_features if c in base.columns]
    feature_cols = rolling_feature_cols + manual_feature_cols

    # TRAIN: current season only uses completed GWs; prior completed seasons remain fully available.
    current_season = _season_mask(base, current_season_name)
    round_numbers = pd.to_numeric(base["round"], errors="coerce")
    completed_current_rows = (~current_season) | round_numbers.le(int(last_finished_gw))
    eligible_base = base[completed_current_rows].copy()
    train_rows = base[(base["enough_prev"]) & completed_current_rows].copy()
    X_train = train_rows[feature_cols].fillna(0.0)
    y_train = train_rows["total_points"].astype(float)
    metadata_cols = [
        c
        for c in (
            "player_id",
            "current_player_id",
            "fixture",
            "season_name",
            "round",
            "kickoff_time",
            "minutes",
            "element_type",
            "total_points",
        )
        if c in train_rows.columns
    ]
    train_metadata = train_rows[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=train_rows.index)

    # PRED: prefer each player's latest current-season row up to the last finished GW,
    # falling back to their latest historical row if current-season history is absent.
    base_with_sort = _add_history_sort_keys(eligible_base)
    sort_cols = _history_sort_columns(base_with_sort, ["player_id"])
    eligible_current_season = _season_mask(base_with_sort, current_season_name)
    current_rows = base_with_sort[eligible_current_season]
    last_rows = current_rows.sort_values(sort_cols, kind="mergesort").groupby("player_id").tail(1)
    historical_fallback = pd.DataFrame(columns=base_with_sort.columns)
    if len(last_rows) < elements_with_team["player_id"].nunique():
        missing_ids = set(elements_with_team["player_id"]) - set(last_rows["player_id"])
        historical_fallback = (
            base_with_sort[base_with_sort["player_id"].isin(missing_ids)]
            .sort_values(sort_cols, kind="mergesort")
            .groupby("player_id")
            .tail(1)
        )
    if not historical_fallback.empty:
        last_rows = pd.concat([last_rows, historical_fallback], ignore_index=True, sort=False)
    last_rows = last_rows.drop(columns=[_TEMP_SEASON_SORT_COL, _TEMP_KICKOFF_SORT_COL], errors="ignore")
    historical_features = _prediction_feature_snapshot(
        eligible_base,
        last_rows,
        feature_cols,
    )

    meta_cols = [
        "player_id",
        "player_code",
        "full_name",
        "now_cost_millions",
        "team_id",
        "element_type",
        "team_name",
        "season_minutes",
        "availability_this_round",
        "availability_next_round",
        "status_code",
        "status_availability",
        "status_injury_flag",
        "status_persistent_flag",
        "status_return_date",
        "future_status_applied_flag",
        "target_fixture_kickoff",
        "injury_risk_flag",
        "official_ep_next",
        "price_change_percent",
        "set_piece_duty_score",
        "has_corners_duty",
        "primary_corners_taker",
        "has_direct_fk_duty",
        "primary_direct_fk_taker",
        "has_penalty_duty",
        "primary_penalty_taker",
        "preseason_minutes",
        "preseason_starts",
        "preseason_appearances",
        "preseason_last_two_minutes",
        "prior_p_start",
        "prior_p_appearance",
        "preseason_prior_quality",
        "preseason_data_source",
        "source_confidence",
        "preseason_appearance_share",
        "preseason_start_share",
        "preseason_minutes_share",
        "preseason_recent_minutes_share",
        "preseason_evidence_components",
        "preseason_evidence_depth",
        "role_target_gameweek",
        "attacking_role_delta",
        "penalty_role_delta",
        "corner_role_delta",
        "direct_fk_role_delta",
        "role_evidence_matches",
        "role_evidence_confidence",
        "role_last_observed_at",
        "role_change_score",
        "role_evidence_components",
        "role_evidence_quality",
        "role_data_source",
        "role_notes",
        "tournament_minutes",
        "tournament_matches",
        "tournament_starts",
        "last_tournament_appearance_at",
        "late_return_flag",
        "readiness_multiplier",
        "workload_evidence_confidence",
        "workload_valid_through_gameweek",
        "workload_notes",
        "workload_data_source",
        "provider_expected_points",
        "provider_expected_points_weight",
        "provider_expected_points_source",
    ]
    meta_cols = [col for col in meta_cols if col in elements_with_team.columns]
    X_pred = elements_with_team[meta_cols].drop_duplicates("player_id").merge(
        historical_features,
        on="player_id",
        how="left",
    )

    use_official_next = official_next_gw is None or int(next_gw) == int(official_next_gw)
    if "official_ep_next" in X_pred.columns and not use_official_next:
        X_pred["official_ep_next"] = np.nan

    prediction_element_type = pd.to_numeric(X_pred["element_type"], errors="coerce")
    for code, label in ((1, "gk"), (2, "def"), (3, "mid"), (4, "fwd")):
        X_pred[f"position_{label}"] = (prediction_element_type == code).astype(float)

    history_counts = eligible_base.groupby("player_id").size().rename("history_match_count")
    X_pred = X_pred.merge(history_counts, on="player_id", how="left")
    X_pred["history_match_count"] = X_pred["history_match_count"].fillna(0.0)
    current_season_counts = (
        current_rows.groupby("player_id").size().rename("current_season_match_count")
    )
    X_pred = X_pred.merge(current_season_counts, on="player_id", how="left")
    X_pred["current_season_match_count"] = X_pred[
        "current_season_match_count"
    ].fillna(0.0)

    # Historical replay must use the price available in the target gameweek,
    # not the final bootstrap price captured when the replay is run.
    if "value" in base.columns:
        target_prices = base[
            current_season & (pd.to_numeric(base["round"], errors="coerce") == int(next_gw))
        ][["player_id", "value"]].copy()
        if not target_prices.empty:
            target_prices["historical_price"] = (
                pd.to_numeric(target_prices["value"], errors="coerce") / 10.0
            )
            target_prices = target_prices.dropna(subset=["historical_price"]).drop_duplicates(
                "player_id", keep="last"
            )
            price_map = target_prices.set_index("player_id")["historical_price"]
            mapped_prices = X_pred["player_id"].map(price_map)
            X_pred["now_cost_millions"] = mapped_prices.fillna(X_pred["now_cost_millions"])

    fixture_context = _target_fixture_context(
        base,
        fixtures_df,
        next_gw=next_gw,
        last_finished_gw=last_finished_gw,
        current_season=current_season,
        windows=tuple(ROLLING_WINDOWS),
        teams_df=teams_df,
        team_priors_df=team_priors_df,
    )
    if not fixture_context.empty:
        X_pred = X_pred.merge(fixture_context, on="team_id", how="left", suffixes=("", "_target"))
        target_columns = ["was_home", "fixture_difficulty"] + [
            col for col in feature_cols if col.startswith("team_") or col.startswith("opp_team_")
        ]
        target_columns.extend(
            [
                "fixture_count",
                "target_fixture_kickoff",
                "team_prior_source",
                "opponent_team_prior_source",
                "team_context_prior_weight",
                "opponent_team_context_prior_weight",
            ]
        )
        for col in target_columns:
            target_col = f"{col}_target"
            if target_col in X_pred.columns:
                X_pred[col] = X_pred[target_col].fillna(X_pred.get(col, 0.0))
                X_pred = X_pred.drop(columns=[target_col])

    if not use_official_next:
        persistent_status = (
            X_pred["status_persistent_flag"].eq(1.0)
            if "status_persistent_flag" in X_pred.columns
            else pd.Series(False, index=X_pred.index)
        )
        target_fixture_kickoff = (
            pd.to_datetime(X_pred["target_fixture_kickoff"], errors="coerce", utc=True).dt.tz_localize(None)
            if "target_fixture_kickoff" in X_pred.columns
            else pd.Series(pd.NaT, index=X_pred.index)
        )
        status_return_date = (
            pd.to_datetime(X_pred["status_return_date"], errors="coerce", utc=True).dt.tz_localize(None)
            if "status_return_date" in X_pred.columns
            else pd.Series(pd.NaT, index=X_pred.index)
        )
        status_code = (
            X_pred["status_code"].fillna("").astype(str).str.lower()
            if "status_code" in X_pred.columns
            else pd.Series("", index=X_pred.index, dtype=str)
        )
        dated_future_status = (
            status_code.isin({"i", "s"})
            & target_fixture_kickoff.notna()
            & status_return_date.notna()
            & target_fixture_kickoff.lt(status_return_date)
        )
        future_status = persistent_status | dated_future_status
        if "future_status_applied_flag" in X_pred.columns:
            X_pred["future_status_applied_flag"] = future_status.astype(int)
        if "availability_this_round" in X_pred.columns:
            X_pred["availability_this_round"] = np.where(
                future_status,
                X_pred["availability_this_round"],
                1.0,
            )
        if "availability_next_round" in X_pred.columns:
            X_pred["availability_next_round"] = np.where(
                future_status,
                X_pred["availability_next_round"],
                1.0,
            )
        if "status_availability" in X_pred.columns:
            X_pred["status_availability"] = np.where(
                future_status,
                X_pred["status_availability"],
                1.0,
            )
        if "status_injury_flag" in X_pred.columns:
            X_pred["status_injury_flag"] = np.where(
                future_status,
                X_pred["status_injury_flag"],
                0,
            )
        if "status_persistent_flag" in X_pred.columns:
            X_pred["status_persistent_flag"] = np.where(
                future_status,
                X_pred["status_persistent_flag"],
                0,
            )
        if "injury_risk_flag" in X_pred.columns:
            X_pred["injury_risk_flag"] = np.where(
                future_status,
                X_pred["injury_risk_flag"],
                0,
            )
    if "status_code" in X_pred.columns:
        X_pred = X_pred.drop(columns=["status_code"])

    X_pred[feature_cols] = X_pred[feature_cols].fillna(0.0)
    X_pred = X_pred.reset_index(drop=True)
    return X_train, y_train, X_pred, train_metadata

def expand_for_double_gw(pred_df: pd.DataFrame, fixtures_df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
    """
    If a player has multiple fixtures in next_gw, scale EP by number of fixtures.
    fixtures_df: all fixtures
    """
    # Count fixtures per team in next_gw
    gw_fx = fixtures_df[fixtures_df["event"] == next_gw]
    if gw_fx.empty:
        pred_df["fixture_multiplier"] = 1.0
        return pred_df
    team_counts = {}
    for _, row in gw_fx.iterrows():
        team_counts[row["team_h"]] = team_counts.get(row["team_h"], 0) + 1
        team_counts[row["team_a"]] = team_counts.get(row["team_a"], 0) + 1
    pred_df["fixture_multiplier"] = pred_df["team_id"].map(team_counts).fillna(0).astype(float)
    return pred_df


def log_model_feature_weights(
    logger: logging.Logger,
    feature_names: Sequence[str],
    model: object,
    model_label: str = "model",
    top_n: int = 15,
) -> None:
    """
    Log the most influential features (by absolute weight) for a fitted model.
    """
    if feature_names is None:
        logger.info("No feature names available to log for %s", model_label)
        return

    feature_names = list(feature_names)
    if not feature_names:
        logger.info("No feature names available to log for %s", model_label)
        return

    selector = None
    estimator = None
    if hasattr(model, "named_steps"):
        selector = model.named_steps.get("feature_selector")
        estimator = model.named_steps.get("est")
    if selector is not None and hasattr(selector, "features_to_keep_") and selector.features_to_keep_:
        feature_names = list(selector.features_to_keep_)
    if estimator is None and hasattr(model, "feature_importances_"):
        estimator = model
    if estimator is None:
        logger.info("Skipping feature weight logging for %s; estimator unavailable.", model_label)
        return

    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        logger.info(
            "Estimator %s for %s does not expose feature_importances_; skipping logging.",
            type(estimator).__name__,
            model_label,
        )
        return

    if len(importances) != len(feature_names):
        logger.warning(
            "Feature name count (%d) does not match weights (%d) for %s; logging skipped.",
            len(feature_names),
            len(importances),
            model_label,
        )
        return

    pairs = sorted(
        zip(feature_names, importances),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    top_pairs = [(name, weight) for name, weight in pairs if abs(weight) > 0][:top_n]

    if not top_pairs:
        logger.info("All feature importances are zero for %s.", model_label)
        return

    formatted = ", ".join(f"{name}: {weight:.4f}" for name, weight in top_pairs)
    logger.info("Top feature weights for %s: %s", model_label, formatted)
