"""Structured analytics context for the separate local-LLM assistant app."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from fplmodel.config import FORMATION_OPTIONS
from fplmodel.team_analysis import compare_team_to_optimal, summarise_team
from fplmodel.team_picker import pick_best_xi
from fplmodel.transfer_recommender import aggregate_expected_points, recommend_transfers

from .data_access import infer_captain_id


def _round_float(value: object, digits: int = 2) -> Optional[float]:
    """Round numeric values for prompt-friendly output."""

    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _player_snapshot(record: Dict[str, Any]) -> Dict[str, Any]:
    """Trim player records down to high-signal fields."""

    return {
        "player_id": int(record["player_id"]),
        "full_name": record.get("full_name"),
        "team_name": record.get("team_name"),
        "position": int(record.get("element_type")) if pd.notna(record.get("element_type")) else None,
        "expected_points": _round_float(record.get("expected_points")),
        "cost_m": _round_float(record.get("now_cost_millions")),
        "confidence_level": record.get("confidence_level"),
        "confidence_score": _round_float(record.get("confidence_score"), 1),
        "start_probability": _round_float(record.get("start_probability"), 3),
        "range_80": {
            "lower": _round_float(record.get("expected_points_lower_80")),
            "upper": _round_float(record.get("expected_points_upper_80")),
        },
    }


def _build_risk_flags(team_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Collect injury, availability, and low-confidence warnings for the squad."""

    risk_rows: List[Dict[str, Any]] = []
    for _, row in team_df.iterrows():
        issues: List[str] = []
        status = str(row.get("status") or "").strip().lower()
        if status in {"d", "i", "s", "u", "n"}:
            issues.append(f"status={status}")

        next_round = row.get("chance_of_playing_next_round")
        if next_round is not None and not pd.isna(next_round) and float(next_round) < 100.0:
            issues.append(f"chance_next_round={int(float(next_round))}%")

        confidence_level = str(row.get("confidence_level") or "").strip()
        if confidence_level.lower() == "low":
            issues.append("low_model_confidence")

        if issues:
            risk_rows.append(
                {
                    "player": row.get("full_name"),
                    "issues": issues,
                    "news": str(row.get("news") or "").strip(),
                }
            )
    return risk_rows


def _build_benching_advice(optimized_team: Dict[str, Any]) -> Dict[str, Any]:
    """Extract lineup advice from the optimized squad selection."""

    starters = [_player_snapshot(player) for player in optimized_team.get("squad", [])]
    bench = [_player_snapshot(player) for player in optimized_team.get("bench", [])]
    return {
        "suggested_formation": optimized_team.get("formation_name"),
        "captain": optimized_team.get("captain"),
        "starting_xi": starters,
        "bench": bench,
        "projected_points_with_captain": _round_float(
            optimized_team.get("total_expected_points_with_captain")
        ),
    }


def _build_transfer_summary(transfers_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert transfer recommendations into a prompt-friendly summary."""

    suggestions = []
    for item in transfers_payload.get("recommended_transfers", []):
        suggestions.append(
            {
                "out": _player_snapshot(item["out_player"]),
                "in": _player_snapshot(item["in_player"]),
                "expected_points_delta": _round_float(item.get("expected_points_delta")),
            }
        )

    metadata = transfers_payload.get("metadata", {})
    return {
        "recommendations": suggestions,
        "free_transfers_used": int(metadata.get("free_transfers_used", 0)),
        "additional_transfers": int(metadata.get("additional_transfers", 0)),
        "projected_points_current": _round_float(metadata.get("total_expected_points_current")),
        "projected_points_optimal": _round_float(metadata.get("total_expected_points_optimal")),
    }


def _build_comparison_summary(comparison_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize the user's gap versus the best available team."""

    comparison = comparison_payload.get("comparison", {})
    return {
        "user_expected_points": _round_float(comparison.get("user_expected_points")),
        "optimal_expected_points": _round_float(comparison.get("optimal_expected_points")),
        "points_gap": _round_float(comparison.get("points_gap")),
        "rating_pct": _round_float(comparison.get("rating")),
    }


def _build_horizon_leaders(aggregated_predictions: pd.DataFrame, limit: int = 8) -> List[Dict[str, Any]]:
    """List the strongest projected players over the selected horizon."""

    leaders = aggregated_predictions.sort_values("expected_points", ascending=False).head(limit)
    return [_player_snapshot(record) for record in leaders.to_dict(orient="records")]


def build_analysis_context(
    *,
    user_team: pd.DataFrame,
    current_predictions: pd.DataFrame,
    predictions_by_gw: Dict[int, pd.DataFrame],
    loaded_gws: List[int],
    missing_gws: List[int],
    free_transfers: int,
    max_transfers: int,
) -> Dict[str, Any]:
    """Build a structured context object for the local LLM."""

    captain_id = infer_captain_id(user_team)
    user_summary = summarise_team(user_team, captain_id=captain_id)
    optimized_current_squad = pick_best_xi(
        user_team,
        budget_m=float(user_summary.total_cost),
        formations=FORMATION_OPTIONS,
    )
    comparison_payload = compare_team_to_optimal(
        current_predictions,
        user_team,
        captain_id=captain_id,
        budget_m=float(user_summary.total_cost),
        formations=FORMATION_OPTIONS,
    )
    transfers_payload = recommend_transfers(
        user_team,
        predictions_by_gw,
        gameweeks=loaded_gws,
        free_transfers=free_transfers,
        max_transfers=max_transfers,
        budget_m=float(user_summary.total_cost),
        formations=FORMATION_OPTIONS,
    )
    aggregated_predictions = aggregate_expected_points(predictions_by_gw, gameweeks=loaded_gws)

    top_captains = (
        user_team.sort_values("expected_points", ascending=False)
        .head(3)
        .to_dict(orient="records")
    )

    context = {
        "analysis_scope": {
            "current_gameweek": int(loaded_gws[0]),
            "horizon_gameweeks": [int(gw) for gw in loaded_gws],
            "missing_prediction_gameweeks": [int(gw) for gw in missing_gws],
        },
        "user_squad": {
            "total_cost_m": _round_float(user_summary.total_cost),
            "current_captain": user_summary.captain,
            "projected_points_without_captain": _round_float(
                user_summary.expected_points_without_captain
            ),
            "projected_points_with_captain": _round_float(
                user_summary.total_expected_points_with_captain
            ),
            "bench_expected_points": _round_float(user_summary.bench_expected_points),
            "players": [_player_snapshot(player) for player in user_team.to_dict(orient="records")],
        },
        "lineup_advice": _build_benching_advice(optimized_current_squad),
        "captaincy_options": [_player_snapshot(player) for player in top_captains],
        "optimal_comparison": _build_comparison_summary(comparison_payload),
        "transfer_plan": _build_transfer_summary(transfers_payload),
        "risk_flags": _build_risk_flags(user_team),
        "horizon_top_targets": _build_horizon_leaders(aggregated_predictions),
    }
    return context
