from __future__ import annotations

import pandas as pd


_GENERATED_COLUMNS = {
    "historical_player_id",
    "historical_element_type",
    "historical_team",
    "current_player_id",
    "current_element_type",
    "is_current_player_mapped",
    "training_player_key",
    "position_changed",
}


def _positive_integer_codes(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & numeric.gt(0) & numeric.mod(1).eq(0)
    if not bool(valid.all()):
        invalid_count = int((~valid).sum())
        raise ValueError(
            f"{label} contains {invalid_count} invalid stable player code(s); "
            "codes must be positive integers."
        )
    return numeric.astype("int64")


def _current_player_mapping(current_players: pd.DataFrame) -> pd.DataFrame:
    if "code" not in current_players.columns or "player_id" not in current_players.columns:
        return pd.DataFrame(
            {
                "player_code": pd.Series(dtype="int64"),
                "current_player_id": pd.Series(dtype="Int64"),
                "current_element_type": pd.Series(dtype="Int64"),
            }
        )

    columns = ["code", "player_id"]
    if "element_type" in current_players.columns:
        columns.append("element_type")
    current = current_players[columns].copy()

    numeric_codes = pd.to_numeric(current["code"], errors="coerce")
    numeric_ids = pd.to_numeric(current["player_id"], errors="coerce")
    valid = (
        numeric_codes.notna()
        & numeric_codes.gt(0)
        & numeric_codes.mod(1).eq(0)
        & numeric_ids.notna()
        & numeric_ids.gt(0)
        & numeric_ids.mod(1).eq(0)
    )
    current = current.loc[valid].copy()
    current["player_code"] = numeric_codes.loc[valid].astype("int64")
    current["current_player_id"] = numeric_ids.loc[valid].astype("int64")

    if "element_type" in current.columns:
        current["current_element_type"] = pd.to_numeric(
            current["element_type"], errors="coerce"
        ).astype("Int64")
    else:
        current["current_element_type"] = pd.Series(
            pd.NA, index=current.index, dtype="Int64"
        )

    mapping = current[
        ["player_code", "current_player_id", "current_element_type"]
    ].drop_duplicates()
    conflicting_codes = mapping.groupby("player_code")["current_player_id"].nunique()
    if bool(conflicting_codes.gt(1).any()):
        raise ValueError("Current player data maps one stable code to multiple player IDs.")
    conflicting_ids = mapping.groupby("current_player_id")["player_code"].nunique()
    if bool(conflicting_ids.gt(1).any()):
        raise ValueError("Current player data maps one player ID to multiple stable codes.")

    return mapping.drop_duplicates("player_code").reset_index(drop=True)


def build_history_universe(
    external_histories: pd.DataFrame,
    current_players: pd.DataFrame,
) -> pd.DataFrame:
    """Retain the full training universe while mapping active players to live IDs.

    Historical FPL element IDs are season-local, so stable ``player_code`` values
    define player identity across seasons. Active players use their positive live
    ID. Historical-only players use the negative of their stable code, keeping the
    two numeric namespaces disjoint for the existing feature pipeline.
    """
    if external_histories is None or external_histories.empty:
        return pd.DataFrame()
    required = {"player_id", "player_code"}
    missing = required.difference(external_histories.columns)
    if missing:
        missing_labels = ", ".join(sorted(missing))
        raise ValueError(f"External history is missing required column(s): {missing_labels}.")

    histories = external_histories.drop(
        columns=_GENERATED_COLUMNS.intersection(external_histories.columns),
        errors="ignore",
    ).copy()
    histories["player_code"] = _positive_integer_codes(
        histories["player_code"], "External history"
    )
    histories["historical_player_id"] = pd.to_numeric(
        histories["player_id"], errors="coerce"
    ).astype("Int64")
    if "element_type" in histories.columns:
        histories["historical_element_type"] = histories["element_type"]
    if "team" in histories.columns:
        histories["historical_team"] = histories["team"]
    histories = histories.drop(columns=["player_id"])

    current = _current_player_mapping(current_players)
    universe = histories.merge(
        current,
        on="player_code",
        how="left",
        validate="m:1",
        sort=False,
    )
    universe["current_player_id"] = universe["current_player_id"].astype("Int64")
    universe["current_element_type"] = universe["current_element_type"].astype("Int64")
    universe["is_current_player_mapped"] = universe["current_player_id"].notna()

    historical_surrogate = -universe["player_code"]
    universe["player_id"] = universe["current_player_id"].fillna(
        historical_surrogate
    ).astype("int64")
    universe["training_player_key"] = (
        "player-code:" + universe["player_code"].astype(str)
    ).astype("string")

    universe["position_changed"] = pd.Series(
        pd.NA, index=universe.index, dtype="boolean"
    )
    if "element_type" in universe.columns:
        historical_position = pd.to_numeric(universe["element_type"], errors="coerce")
        comparable = (
            universe["is_current_player_mapped"]
            & historical_position.notna()
            & universe["current_element_type"].notna()
        )
        universe.loc[comparable, "position_changed"] = historical_position.loc[
            comparable
        ].ne(universe.loc[comparable, "current_element_type"])

    player_code_counts = universe.groupby("player_id")["player_code"].nunique()
    if bool(player_code_counts.gt(1).any()):
        raise RuntimeError("Training player IDs collided across stable player codes.")

    return universe
