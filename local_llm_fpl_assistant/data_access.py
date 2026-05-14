"""Data loading helpers for the separate local-LLM Streamlit app."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from fplmodel.config import DATA_DIR, OUTPUTS_DIR
from fplmodel.utils import get_current_and_last_finished_gw


FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def extract_gw_from_path(path: Path) -> Optional[int]:
    """Extract a gameweek number from an artifact filename."""

    match = re.search(r"_gw(\d+)", path.stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def discover_prediction_files() -> Dict[int, Path]:
    """Map available prediction gameweeks to their CSV files."""

    files: Dict[int, Path] = {}
    for path in OUTPUTS_DIR.glob("predictions_gw*.csv"):
        gw = extract_gw_from_path(path)
        if gw is not None:
            files[gw] = path
    return files


def load_predictions(path: Path) -> pd.DataFrame:
    """Load a prediction CSV artifact."""

    predictions = pd.read_csv(path)
    if "player_id" not in predictions.columns:
        raise ValueError(f"Prediction file missing player_id: {path}")
    predictions["player_id"] = predictions["player_id"].astype(int)
    return predictions


def load_bootstrap_data() -> Dict[str, object]:
    """Load cached bootstrap data from disk."""

    path = DATA_DIR / "raw" / "bootstrap-static.json"
    if not path.exists():
        raise FileNotFoundError(
            "bootstrap-static.json not found in data/raw. Run the pipeline first."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_live_bootstrap_data() -> Dict[str, object]:
    """Attempt to load current bootstrap metadata from the public FPL API."""

    response = requests.get(FPL_BOOTSTRAP_URL, timeout=10)
    response.raise_for_status()
    data = response.json()
    if not data.get("events"):
        raise ValueError("Live bootstrap response did not include events.")
    if not data.get("elements"):
        raise ValueError("Live bootstrap response did not include elements.")
    return data


def load_bootstrap_data_with_fallback() -> Dict[str, object]:
    """Load bootstrap metadata, preferring live data and falling back to cache."""

    try:
        return load_live_bootstrap_data()
    except (requests.RequestException, ValueError):
        return load_bootstrap_data()


def load_bootstrap_events() -> pd.DataFrame:
    """Load event metadata, preferring live data and falling back to cached data."""

    data = load_bootstrap_data_with_fallback()
    events = data.get("events", [])
    if not events:
        raise ValueError("No events data found in bootstrap metadata.")
    return pd.DataFrame(events)


def select_prediction_gameweek(
    available_gameweeks: List[int],
    events_df: Optional[pd.DataFrame],
) -> int:
    """Select the most relevant prediction gameweek."""

    if not available_gameweeks:
        raise FileNotFoundError(
            "No prediction files found in outputs/. Run the pipeline first."
        )

    candidate_gws = sorted(available_gameweeks)
    if events_df is not None and not events_df.empty:
        try:
            next_gw, last_finished = get_current_and_last_finished_gw(events_df)
        except (KeyError, TypeError, ValueError):
            next_gw, last_finished = None, None

        if next_gw in candidate_gws:
            return int(next_gw)

        if last_finished is not None:
            future_gws = [gw for gw in candidate_gws if gw > int(last_finished)]
            if future_gws:
                return future_gws[0]

    return candidate_gws[-1]


def load_next_predictions() -> Tuple[int, Path, pd.DataFrame]:
    """Load the selected next-gameweek prediction file."""

    files = discover_prediction_files()
    if not files:
        raise FileNotFoundError(
            "No prediction files found in outputs/. Run the pipeline first."
        )

    try:
        events_df = load_bootstrap_events()
    except (FileNotFoundError, ValueError):
        events_df = None

    next_gw = select_prediction_gameweek(sorted(files), events_df)
    predictions_path = files[next_gw]
    return next_gw, predictions_path, load_predictions(predictions_path)


def load_predictions_for_horizon(
    horizon: int,
) -> Tuple[List[int], Dict[int, pd.DataFrame], List[int]]:
    """Load prediction frames across a requested horizon."""

    files = discover_prediction_files()
    if not files:
        raise FileNotFoundError(
            "No prediction files found in outputs/. Run the pipeline first."
        )

    try:
        events_df = load_bootstrap_events()
    except (FileNotFoundError, ValueError):
        events_df = None
    start_gw = select_prediction_gameweek(sorted(files), events_df)

    target_gws = [start_gw + offset for offset in range(horizon)]
    predictions_by_gw: Dict[int, pd.DataFrame] = {}
    missing: List[int] = []

    for gw in target_gws:
        path = files.get(gw)
        if path is None:
            missing.append(gw)
            continue
        predictions_by_gw[gw] = load_predictions(path)

    if not predictions_by_gw:
        raise FileNotFoundError(
            "No prediction files found for the requested horizon starting at "
            f"GW{start_gw}."
        )

    return sorted(predictions_by_gw), predictions_by_gw, missing


def fetch_fpl_team_from_api(fpl_id: int, event: int) -> pd.DataFrame:
    """Fetch a user's squad for a specific gameweek via the public FPL API."""

    if fpl_id <= 0:
        raise ValueError("FPL ID must be a positive integer.")

    url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/event/{event}/picks/"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch FPL data (status {response.status_code}). "
            "Double-check the FPL ID and gameweek."
        )

    payload = response.json()
    picks = payload.get("picks", [])
    if not picks:
        raise ValueError("No picks returned for the provided FPL ID and gameweek.")

    records = []
    for pick in picks:
        position = int(pick.get("position", 0))
        records.append(
            {
                "player_id": int(pick.get("element", 0)),
                "starting": int(position <= 11),
                "bench": int(position > 11),
                "captain": int(bool(pick.get("is_captain", False))),
                "vice_captain": int(bool(pick.get("is_vice_captain", False))),
                "fpl_position": position,
            }
        )

    squad = pd.DataFrame(records)
    if squad.empty:
        raise ValueError("Failed to parse squad picks for the provided FPL ID.")
    squad["player_id"] = squad["player_id"].astype(int)
    return squad


def load_team_from_csv(uploaded_file) -> pd.DataFrame:
    """Load a team CSV from a Streamlit upload object."""

    team_df = pd.read_csv(uploaded_file)
    if "player_id" not in team_df.columns:
        raise ValueError("Uploaded CSV must include a player_id column.")
    team_df["player_id"] = pd.to_numeric(team_df["player_id"], errors="raise").astype(int)
    for col in ("starting", "bench", "captain"):
        if col in team_df.columns:
            team_df[col] = pd.to_numeric(team_df[col], errors="coerce").fillna(0).astype(int)
    return team_df


def infer_captain_id(team_df: pd.DataFrame) -> Optional[int]:
    """Infer the current captain from a team dataframe."""

    if "captain" not in team_df.columns:
        return None
    captain_rows = team_df.loc[team_df["captain"] == 1, "player_id"]
    if captain_rows.empty:
        return None
    return int(captain_rows.iloc[0])


def validate_user_team(team_df: pd.DataFrame) -> None:
    """Validate basic squad shape before analytics are run."""

    if team_df.empty:
        raise ValueError("The provided team is empty.")

    if "player_id" not in team_df.columns:
        raise ValueError("The provided team is missing player_id.")

    if team_df["player_id"].duplicated().any():
        raise ValueError("The provided team contains duplicate player_id values.")

    if len(team_df) != 15:
        raise ValueError(
            f"Expected a full 15-player squad, but received {len(team_df)} players."
        )


def enrich_user_team(
    user_team: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach prediction metadata and player status fields to a user squad."""

    enriched = user_team.copy()
    prediction_cols = [
        "player_id",
        "full_name",
        "team_name",
        "team_id",
        "element_type",
        "now_cost_millions",
        "expected_points",
        "start_probability",
        "confidence_score",
        "confidence_level",
        "expected_points_lower_80",
        "expected_points_upper_80",
    ]
    prediction_cols = [col for col in prediction_cols if col in predictions.columns]
    predictions_meta = predictions[prediction_cols].drop_duplicates("player_id")
    enriched = enriched.merge(predictions_meta, on="player_id", how="left", suffixes=("", "_pred"))

    for col in prediction_cols:
        if col == "player_id":
            continue
        pred_col = f"{col}_pred"
        if col in enriched.columns and pred_col in enriched.columns:
            enriched[col] = enriched[col].fillna(enriched[pred_col])
            enriched = enriched.drop(columns=pred_col)
        elif pred_col in enriched.columns:
            enriched[col] = enriched[pred_col]
            enriched = enriched.drop(columns=pred_col)
        elif col not in enriched.columns:
            enriched[col] = pd.NA

    if "starting" not in enriched.columns:
        enriched["starting"] = 0
        if len(enriched) >= 11:
            enriched.loc[enriched.index[:11], "starting"] = 1
    if "bench" not in enriched.columns:
        enriched["bench"] = 1 - enriched["starting"]
    if "captain" not in enriched.columns:
        enriched["captain"] = 0

    bootstrap_data = load_bootstrap_data_with_fallback()
    elements = pd.DataFrame(bootstrap_data.get("elements", []))
    if not elements.empty:
        element_meta = elements[
            [
                "id",
                "status",
                "news",
                "chance_of_playing_this_round",
                "chance_of_playing_next_round",
            ]
        ].rename(columns={"id": "player_id"})
        enriched = enriched.merge(element_meta, on="player_id", how="left")

    enriched["player_id"] = enriched["player_id"].astype(int)
    for col in ("starting", "bench", "captain"):
        enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(0).astype(int)

    if "chance_of_playing_this_round" in enriched.columns:
        enriched["chance_of_playing_this_round"] = pd.to_numeric(
            enriched["chance_of_playing_this_round"], errors="coerce"
        )
    if "chance_of_playing_next_round" in enriched.columns:
        enriched["chance_of_playing_next_round"] = pd.to_numeric(
            enriched["chance_of_playing_next_round"], errors="coerce"
        )

    return enriched
