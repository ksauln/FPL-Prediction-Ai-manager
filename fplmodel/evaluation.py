from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

from .state import ModelState


def prequential_training_mask(
    metadata: pd.DataFrame,
    *,
    evaluation_gw: int,
    current_season_name: str,
) -> pd.Series:
    """Select rows that were knowable before ``evaluation_gw`` kicked off.

    Prior-season rows remain eligible, while the current season is cut strictly
    before the Gameweek being evaluated.  This mask is used to fit the residual
    calibration model without training on its own evaluation targets.
    """
    if metadata is None or metadata.empty:
        return pd.Series(False, index=getattr(metadata, "index", None), dtype=bool)
    required = {"season_name", "round"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            "Prequential calibration metadata is missing: "
            + ", ".join(sorted(missing))
        )
    seasons = metadata["season_name"].astype("string")
    rounds = pd.to_numeric(metadata["round"], errors="coerce")
    current = seasons.eq(str(current_season_name))
    eligible = seasons.notna() & (~current | rounds.lt(int(evaluation_gw)))
    return eligible.fillna(False).astype(bool)


def evaluate_last_finished_gw_and_update_state(
    start_clf,
    appearance_clf,
    reg,
    cameo_points_by_position: dict[int, float],
    X_train_like: pd.DataFrame,
    histories_df: pd.DataFrame,
    last_finished_gw: int,
    state: ModelState,
    current_season_name: str | None = None,
) -> pd.DataFrame:
    """
    Score last_finished_gw with a caller-supplied prequential model, compare the
    prediction with actual points, and update EMA biases in state.
    Returns residuals df used for update.
    """
    if int(last_finished_gw) <= 0 or state.last_evaluated_gw >= int(last_finished_gw):
        return pd.DataFrame()

    rich_metadata = {"player_id", "element_type", "round", "total_points"}
    if rich_metadata.issubset(X_train_like.columns):
        evaluation_rows = X_train_like[
            pd.to_numeric(X_train_like["round"], errors="coerce").eq(
                int(last_finished_gw)
            )
        ].copy()
        if current_season_name is not None and "season_name" in evaluation_rows.columns:
            evaluation_rows = evaluation_rows[
                evaluation_rows["season_name"].astype(str).eq(str(current_season_name))
            ]
        if evaluation_rows.empty:
            return pd.DataFrame()

        metadata_cols = {
            "player_id",
            "current_player_id",
            "fixture",
            "season_name",
            "round",
            "kickoff_time",
            "minutes",
            "element_type",
            "total_points",
        }
        feats_cols = [
            column for column in X_train_like.columns if column not in metadata_cols
        ]
        merged = evaluation_rows.dropna(subset=feats_cols, how="all")
        if merged.empty:
            return pd.DataFrame()
    else:
        merged = pd.DataFrame()
        feats_cols: list[str] = []

    # We need rows where 'round' == last_finished_gw; features must be from previous match (already lagged in build).
    if merged.empty:
        eval_mask = pd.to_numeric(
            histories_df["round"],
            errors="coerce",
        ).eq(int(last_finished_gw))
        if current_season_name is not None:
            if "season_name" not in histories_df.columns:
                return pd.DataFrame()
            eval_mask &= histories_df["season_name"].astype(str).eq(
                str(current_season_name)
            )
        eval_rows = histories_df[eval_mask].copy()
        if eval_rows.empty:
            return pd.DataFrame()

        # Ensure we have element_type metadata; if histories lack it, merge from provided features frame.
        if "element_type" not in eval_rows.columns:
            meta_map = X_train_like[["player_id", "element_type"]].drop_duplicates()
            eval_rows = eval_rows.merge(meta_map, on="player_id", how="left")

        meta_cols = ["player_id", "element_type"]
        feats_cols = [c for c in X_train_like.columns if c not in meta_cols]
        merged = eval_rows[["player_id", "element_type", "total_points"]].merge(
            X_train_like,
            on=["player_id", "element_type"],
            how="left",
            suffixes=("", ""),
        ).dropna(subset=feats_cols, how="all")

        if merged.empty:
            return pd.DataFrame()

    # Predict
    feats = merged[feats_cols].fillna(0.0)
    start_probability = start_clf.predict_proba(feats)[:, 1]
    appearance_probability = appearance_clf.predict_proba(feats)[:, 1]
    appearance_probability = np.maximum(appearance_probability, start_probability)
    pts_hat = reg.predict(feats)
    cameo_probability = np.clip(appearance_probability - start_probability, 0.0, 1.0)
    cameo_points = merged["element_type"].map(cameo_points_by_position).fillna(1.0)
    ep = start_probability * pts_hat + cameo_probability * cameo_points.to_numpy()

    merged["predicted_points"] = ep
    merged["residual"] = merged["total_points"].astype(float) - merged["predicted_points"].astype(float)
    merged["gw"] = int(last_finished_gw)

    # Update state EMA
    result_cols = ["player_id", "element_type", "residual", "gw"]
    if "fixture" in merged.columns:
        result_cols.insert(1, "fixture")
    res_short = merged[result_cols].copy()
    if "current_player_id" in merged.columns:
        current_ids = pd.to_numeric(merged["current_player_id"], errors="coerce")
        res_short["player_id"] = current_ids.fillna(
            pd.to_numeric(res_short["player_id"], errors="coerce")
        ).astype(int)
    state.update_from_residuals(res_short)
    return res_short
