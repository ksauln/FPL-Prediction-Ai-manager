"""Stateful Fantasy Premier League season manager simulation.

The one-gameweek optimizer is useful, but a season manager has to preserve
state: squad, bank, free transfers, chips, captaincy, and decision history.
This module keeps that state explicit so the same engine can replay a finished
season and later drive live next-gameweek recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    BUDGET_MILLIONS,
    FORMATION_OPTIONS,
    MAX_PER_TEAM,
    OUTPUTS_DIR,
    SQUAD_POSITION_LIMITS,
)
from .team_analysis import summarise_team
from .team_picker import pick_best_xi
from .transfer_recommender import aggregate_expected_points


MANAGER_PRINCIPLE = (
    "A real FPL manager needs state, transfer costs, chip timing, bench value, "
    "future fixtures, and uncertainty. This is a stateful manager, not another "
    "one-week optimizer."
)

CHIP_NAMES = ("wildcard", "free_hit", "bench_boost", "triple_captain")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@dataclass(frozen=True)
class SeasonRules:
    """FPL rule settings that can change by season."""

    budget_m: float = BUDGET_MILLIONS
    free_transfer_per_gameweek: int = 1
    max_free_transfers: int = 5
    transfer_hit_cost: float = 4.0
    first_half_end_gw: int = 19
    chips_by_half: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "first": {
                "wildcard": 1,
                "free_hit": 1,
                "bench_boost": 1,
                "triple_captain": 1,
            },
            "second": {
                "wildcard": 1,
                "free_hit": 1,
                "bench_boost": 1,
                "triple_captain": 1,
            },
        }
    )


@dataclass(frozen=True)
class SeasonManagerConfig:
    """Decision knobs for the season manager."""

    rules: SeasonRules = field(default_factory=SeasonRules)
    initial_horizon: int = 4
    transfer_horizon: int = 4
    chip_lookahead: int = 4
    transfer_gain_threshold: float = 1.5
    max_transfers_per_gw: int = 2
    enable_chips: bool = True
    formations: Sequence[Dict[str, int]] = field(default_factory=lambda: tuple(FORMATION_OPTIONS))
    min_start_probability_for_captain: float = 0.65
    captain_start_probability_weight: float = 0.35
    captain_confidence_weight: float = 0.15
    bench_boost_gain_threshold: float = 8.0
    triple_captain_gain_threshold: float = 5.0
    free_hit_gain_threshold: float = 10.0
    wildcard_gain_threshold: float = 16.0
    monte_carlo_noise_scale: float = 1.0
    strategic_chip_gameweeks: Optional[Sequence[int]] = None


@dataclass
class ManagerState:
    """Mutable season state carried from one gameweek to the next."""

    squad_player_ids: List[int]
    bank_m: float
    free_transfers: int
    purchase_price_by_player_id: Dict[int, float] = field(default_factory=dict)
    used_chips: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {"first": {}, "second": {}}
    )
    history: List[Dict[str, object]] = field(default_factory=list)


def _normalise_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "player_id",
        "full_name",
        "team_name",
        "team_id",
        "element_type",
        "now_cost_millions",
        "expected_points",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")

    out = predictions.copy()
    out["player_id"] = out["player_id"].astype(int)
    out["team_id"] = out["team_id"].astype(int)
    out["element_type"] = out["element_type"].astype(int)
    out["now_cost_millions"] = pd.to_numeric(out["now_cost_millions"], errors="coerce")
    out["expected_points"] = pd.to_numeric(out["expected_points"], errors="coerce").fillna(0.0)
    if "start_probability" not in out.columns:
        out["start_probability"] = 1.0
    out["start_probability"] = pd.to_numeric(out["start_probability"], errors="coerce").fillna(0.75)
    if "confidence_score" not in out.columns:
        out["confidence_score"] = 70.0
    out["confidence_score"] = pd.to_numeric(out["confidence_score"], errors="coerce").fillna(70.0)
    if "expected_points_lower_80" not in out.columns:
        out["expected_points_lower_80"] = (out["expected_points"] - 1.0).clip(lower=0.0)
    if "expected_points_upper_80" not in out.columns:
        out["expected_points_upper_80"] = out["expected_points"] + 1.0
    if out["now_cost_millions"].isna().any():
        bad_ids = out.loc[out["now_cost_millions"].isna(), "player_id"].tolist()
        raise ValueError(f"Missing costs for players: {bad_ids}")
    return out


def _normalise_prediction_map(
    predictions_by_gw: Dict[int, pd.DataFrame],
) -> Dict[int, pd.DataFrame]:
    if not predictions_by_gw:
        raise ValueError("predictions_by_gw cannot be empty")
    return {int(gw): _normalise_predictions(df) for gw, df in predictions_by_gw.items()}


def _available_horizon(
    predictions_by_gw: Dict[int, pd.DataFrame],
    gameweek: int,
    horizon: int,
) -> list[int]:
    available = sorted(gw for gw in predictions_by_gw if gw >= gameweek)
    selected = available[: max(1, int(horizon))]
    if not selected:
        raise KeyError(f"No prediction files available from GW{gameweek}")
    return selected


def _aggregate_projection_for_gameweek(
    predictions_by_gw: Dict[int, pd.DataFrame],
    gameweeks: Iterable[int],
    current_gw: int,
) -> pd.DataFrame:
    """Aggregate horizon points while keeping current-GW player metadata."""

    aggregated = aggregate_expected_points(predictions_by_gw, gameweeks)
    current_meta_cols = [
        "player_id",
        "full_name",
        "team_name",
        "team_id",
        "element_type",
        "now_cost_millions",
    ]
    current_meta = predictions_by_gw[current_gw][current_meta_cols].copy()
    point_cols = ["player_id"] + [
        col
        for col in aggregated.columns
        if col.startswith("expected_points_gw") or col == "expected_points"
    ]
    return current_meta.merge(
        aggregated[point_cols],
        on="player_id",
        how="inner",
    )


def _records_to_squad_ids(team: Dict[str, object]) -> list[int]:
    records = list(team.get("squad", [])) + list(team.get("bench", []))
    return [int(player["player_id"]) for player in records]


def _team_to_frame(team: Dict[str, object]) -> pd.DataFrame:
    records = []
    for player in team.get("squad", []):
        row = dict(player)
        row["starting"] = 1
        row["bench"] = 0
        records.append(row)
    for player in team.get("bench", []):
        row = dict(player)
        row["starting"] = 0
        row["bench"] = 1
        records.append(row)
    return pd.DataFrame(records)


def _squad_frame_for_gw(
    player_ids: Iterable[int],
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    ids = {int(pid) for pid in player_ids}
    frame = predictions[predictions["player_id"].isin(ids)].copy()
    missing = sorted(ids - set(frame["player_id"].astype(int)))
    if missing:
        raise ValueError(f"Missing predictions for squad players: {missing}")
    return frame


def _cost_for_ids(player_ids: Iterable[int], predictions: pd.DataFrame) -> float:
    frame = _squad_frame_for_gw(player_ids, predictions)
    return float(frame["now_cost_millions"].sum())


def _sale_value_millions(purchase_price: float, current_price: float) -> float:
    """Return FPL sale value for a player bought at purchase_price."""

    purchase_price = round(float(purchase_price), 1)
    current_price = round(float(current_price), 1)
    if current_price <= purchase_price:
        return current_price
    profit_tenths = int(round((current_price - purchase_price) * 10))
    sale_profit_tenths = profit_tenths // 2
    return round(purchase_price + (sale_profit_tenths / 10.0), 1)


def _current_price_lookup(predictions: pd.DataFrame) -> dict[int, float]:
    return {
        int(row["player_id"]): float(row["now_cost_millions"])
        for _, row in predictions.iterrows()
    }


def _squad_sale_value(
    player_ids: Iterable[int],
    predictions: pd.DataFrame,
    purchase_price_by_player_id: Dict[int, float],
) -> float:
    current_prices = _current_price_lookup(predictions)
    total = 0.0
    for player_id in player_ids:
        player_id = int(player_id)
        current_price = current_prices[player_id]
        purchase_price = purchase_price_by_player_id.get(player_id, current_price)
        total += _sale_value_millions(purchase_price, current_price)
    return round(total, 1)


def _legal_squad_structure(
    player_ids: Iterable[int],
    predictions: pd.DataFrame,
) -> bool:
    ids = [int(pid) for pid in player_ids]
    if len(ids) != 15 or len(set(ids)) != 15:
        return False
    frame = _squad_frame_for_gw(ids, predictions)
    position_counts = frame["element_type"].value_counts().to_dict()
    position_key = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    for element_type, position_name in position_key.items():
        if int(position_counts.get(element_type, 0)) != int(SQUAD_POSITION_LIMITS[position_name]):
            return False
    if frame.groupby("team_id").size().max() > MAX_PER_TEAM:
        return False
    return True


def _is_legal_squad_ids(
    player_ids: Iterable[int],
    predictions: pd.DataFrame,
    budget_m: float,
) -> bool:
    if not _legal_squad_structure(player_ids, predictions):
        return False
    ids = [int(pid) for pid in player_ids]
    frame = _squad_frame_for_gw(ids, predictions)
    if float(frame["now_cost_millions"].sum()) > budget_m + 1e-6:
        return False
    return True


def _optimise_team(
    predictions: pd.DataFrame,
    budget_m: float,
    config: SeasonManagerConfig,
) -> Dict[str, object]:
    return pick_best_xi(
        predictions,
        budget_m=budget_m,
        formations=config.formations,
    )


def _team_result_from_lineup(
    frame: pd.DataFrame,
    starters: pd.DataFrame,
    bench: pd.DataFrame,
    formation: Dict[str, int],
) -> Dict[str, object]:
    bench_outfield = bench[bench["element_type"] != 1].sort_values(
        "expected_points", ascending=False
    )
    bench_gk = bench[bench["element_type"] == 1].sort_values(
        "expected_points", ascending=False
    )
    bench_ordered = pd.concat([bench_outfield, bench_gk], ignore_index=True)
    bench_ordered["bench_order"] = bench_ordered.index + 1

    starters = starters.sort_values(
        ["element_type", "expected_points"], ascending=[True, False]
    )
    starting_cost = float(starters["now_cost_millions"].sum())
    bench_cost = float(bench_ordered["now_cost_millions"].sum())
    base_ep = float(starters["expected_points"].sum())

    return {
        "squad": starters.to_dict(orient="records"),
        "bench": bench_ordered.to_dict(orient="records"),
        "total_cost": float(frame["now_cost_millions"].sum()),
        "starting_cost": starting_cost,
        "bench_cost": bench_cost,
        "expected_points_without_captain": base_ep,
        "total_expected_points_with_captain": base_ep,
        "bench_expected_points": float(bench_ordered["expected_points"].sum()),
        "captain": None,
        "formation": formation.copy(),
        "formation_name": f"{formation.get('DEF', 0)}-{formation.get('MID', 0)}-{formation.get('FWD', 0)}",
    }


def _pick_lineup_from_owned_squad(
    squad_frame: pd.DataFrame,
    config: SeasonManagerConfig,
) -> Dict[str, object]:
    """Pick starters and bench from an already-owned squad.

    This does not re-apply full purchase constraints such as budget and max
    three per club. Those constraints matter when buying a squad; lineup
    selection should still work for the players already held.
    """

    frame = squad_frame.copy()
    frame["pos_name"] = frame["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    base_columns = [
        "player_id",
        "full_name",
        "team_name",
        "team_id",
        "element_type",
        "now_cost_millions",
        "expected_points",
    ]
    optional_columns = [
        "start_probability",
        "confidence_score",
        "confidence_level",
        "expected_points_lower_80",
        "expected_points_upper_80",
    ]
    selected_columns = base_columns + [col for col in optional_columns if col in frame.columns]
    frame = frame[selected_columns + ["pos_name"]].copy()

    best_result: Optional[Dict[str, object]] = None
    best_points = float("-inf")
    for formation in config.formations:
        starter_parts = []
        possible = True
        for position, count in formation.items():
            candidates = frame[frame["pos_name"] == position].sort_values(
                "expected_points", ascending=False
            )
            if len(candidates) < count:
                possible = False
                break
            starter_parts.append(candidates.head(count))
        if not possible:
            continue

        starters = pd.concat(starter_parts, ignore_index=True)
        starter_ids = set(starters["player_id"].astype(int))
        bench = frame[~frame["player_id"].astype(int).isin(starter_ids)].copy()
        if len(starters) != 11 or len(bench) != 4:
            continue

        result = _team_result_from_lineup(frame, starters, bench, formation)
        points = float(result["expected_points_without_captain"])
        if points > best_points:
            best_points = points
            best_result = result

    if best_result is None:
        position_counts = frame["pos_name"].value_counts().to_dict()
        raise RuntimeError(
            "Unable to pick a valid starting XI from the owned squad. "
            f"Position counts: {position_counts}"
        )
    return best_result


def _captain_score(row: pd.Series, config: SeasonManagerConfig) -> float:
    start_probability = float(row.get("start_probability", 0.75))
    confidence = float(row.get("confidence_score", 70.0)) / 100.0
    expected_points = float(row["expected_points"])
    reliability = (
        (1.0 - config.captain_start_probability_weight - config.captain_confidence_weight)
        + config.captain_start_probability_weight * start_probability
        + config.captain_confidence_weight * confidence
    )
    if start_probability < config.min_start_probability_for_captain:
        reliability *= 0.5
    return expected_points * reliability


def _select_captains(
    starters: pd.DataFrame,
    config: SeasonManagerConfig,
) -> tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    if starters.empty:
        return None, None, None, None
    ranked = starters.copy()
    ranked["captain_score"] = ranked.apply(lambda row: _captain_score(row, config), axis=1)
    ranked = ranked.sort_values(
        ["captain_score", "expected_points"],
        ascending=[False, False],
    )
    captain = ranked.iloc[0]
    vice = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
    return (
        int(captain["player_id"]),
        int(vice["player_id"]),
        str(captain["full_name"]),
        str(vice["full_name"]),
    )


def _apply_captain_flags(
    team: Dict[str, object],
    config: SeasonManagerConfig,
) -> Dict[str, object]:
    out = {
        key: [dict(player) for player in value] if key in {"squad", "bench"} else value
        for key, value in team.items()
    }
    starters = pd.DataFrame(out.get("squad", []))
    captain_id, vice_id, captain_name, vice_name = _select_captains(starters, config)
    for section in ("squad", "bench"):
        for player in out.get(section, []):
            pid = int(player["player_id"])
            player["captain"] = int(captain_id is not None and pid == captain_id)
            player["vice_captain"] = int(vice_id is not None and pid == vice_id)
    captain_points = 0.0
    if captain_id is not None and not starters.empty:
        captain_points = float(starters.loc[starters["player_id"] == captain_id, "expected_points"].iloc[0])
    base_points = float(starters["expected_points"].sum()) if not starters.empty else 0.0
    out["captain"] = captain_name
    out["captain_id"] = captain_id
    out["vice_captain"] = vice_name
    out["vice_captain_id"] = vice_id
    out["expected_points_without_captain"] = base_points
    out["total_expected_points_with_captain"] = base_points + captain_points
    return out


def _summarise_fixed_squad(
    player_ids: Iterable[int],
    predictions: pd.DataFrame,
    config: SeasonManagerConfig,
) -> Dict[str, object]:
    frame = _squad_frame_for_gw(player_ids, predictions)
    team = _pick_lineup_from_owned_squad(frame, config)
    return _apply_captain_flags(team, config)


def _half_for_gameweek(gameweek: int, rules: SeasonRules) -> str:
    return "first" if int(gameweek) <= int(rules.first_half_end_gw) else "second"


def _chip_remaining(
    state: ManagerState,
    gameweek: int,
    chip: str,
    rules: SeasonRules,
) -> int:
    half = _half_for_gameweek(gameweek, rules)
    allowed = int(rules.chips_by_half.get(half, {}).get(chip, 0))
    used = int(state.used_chips.get(half, {}).get(chip, 0))
    return max(0, allowed - used)


def _mark_chip_used(
    state: ManagerState,
    gameweek: int,
    chip: str,
    rules: SeasonRules,
) -> None:
    half = _half_for_gameweek(gameweek, rules)
    state.used_chips.setdefault(half, {})
    state.used_chips[half][chip] = int(state.used_chips[half].get(chip, 0)) + 1


def _can_use_free_hit(state: ManagerState, gameweek: int) -> bool:
    if int(gameweek) == 1:
        return False
    if state.history and state.history[-1].get("chip") == "free_hit":
        return False
    return True


def _transfer_candidate(
    state: ManagerState,
    predictions_by_gw: Dict[int, pd.DataFrame],
    gameweek: int,
    config: SeasonManagerConfig,
) -> dict[str, object]:
    horizon_gws = _available_horizon(predictions_by_gw, gameweek, config.transfer_horizon)
    aggregated = _aggregate_projection_for_gameweek(predictions_by_gw, horizon_gws, gameweek)
    current_ids = set(state.squad_player_ids)
    current_agg = _squad_frame_for_gw(current_ids, aggregated)
    current_total = float(current_agg["expected_points"].sum())
    resale_budget = (
        _squad_sale_value(
            state.squad_player_ids,
            aggregated,
            state.purchase_price_by_player_id,
        )
        + state.bank_m
    )

    optimal = _optimise_team(aggregated, budget_m=resale_budget, config=config)
    optimal_ids = set(_records_to_squad_ids(optimal))
    outgoing = list(current_ids - optimal_ids)
    incoming = list(optimal_ids - current_ids)

    if not outgoing or not incoming:
        return {
            "transfers": [],
            "gain": 0.0,
            "projected_total_after": current_total,
            "projected_total_before": current_total,
            "horizon_gameweeks": horizon_gws,
        }

    lookup = aggregated.set_index("player_id")
    outgoing_by_pos: dict[int, list[int]] = {}
    incoming_by_pos: dict[int, list[int]] = {}
    for pid in outgoing:
        outgoing_by_pos.setdefault(int(lookup.loc[pid, "element_type"]), []).append(pid)
    for pid in incoming:
        incoming_by_pos.setdefault(int(lookup.loc[pid, "element_type"]), []).append(pid)

    suggestions: list[dict[str, object]] = []
    trial_ids = list(state.squad_player_ids)
    trial_bank = float(state.bank_m)
    max_transfers = min(config.max_transfers_per_gw, len(outgoing), len(incoming))
    for pos in sorted(set(outgoing_by_pos) | set(incoming_by_pos)):
        outs = sorted(outgoing_by_pos.get(pos, []), key=lambda pid: float(lookup.loc[pid, "expected_points"]))
        ins = sorted(
            incoming_by_pos.get(pos, []),
            key=lambda pid: float(lookup.loc[pid, "expected_points"]),
            reverse=True,
        )
        for out_pid, in_pid in zip(outs, ins):
            if len(suggestions) >= max_transfers:
                break
            candidate_ids = [in_pid if pid == out_pid else pid for pid in trial_ids]
            if not _legal_squad_structure(candidate_ids, aggregated):
                continue
            out_current_price = float(lookup.loc[out_pid, "now_cost_millions"])
            in_current_price = float(lookup.loc[in_pid, "now_cost_millions"])
            out_purchase_price = state.purchase_price_by_player_id.get(out_pid, out_current_price)
            sale_value = _sale_value_millions(out_purchase_price, out_current_price)
            if trial_bank + sale_value + 1e-6 < in_current_price:
                continue
            gain = float(lookup.loc[in_pid, "expected_points"] - lookup.loc[out_pid, "expected_points"])
            suggestions.append(
                {
                    "out_player": _player_record(lookup, out_pid),
                    "in_player": _player_record(lookup, in_pid),
                    "expected_points_delta": gain,
                    "out_purchase_price": float(out_purchase_price),
                    "out_sale_value": float(sale_value),
                    "in_purchase_price": float(in_current_price),
                }
            )
            trial_ids = candidate_ids
            trial_bank = round(trial_bank + sale_value - in_current_price, 2)
        if len(suggestions) >= max_transfers:
            break

    gross_gain = float(sum(item["expected_points_delta"] for item in suggestions))
    paid_transfers = max(0, len(suggestions) - state.free_transfers)
    hit_cost = paid_transfers * config.rules.transfer_hit_cost
    net_gain = gross_gain - hit_cost
    if net_gain < config.transfer_gain_threshold:
        suggestions = []
        net_gain = 0.0

    return {
        "transfers": suggestions,
        "gain": float(net_gain),
        "gross_gain": float(gross_gain),
        "hit_cost": float(hit_cost),
        "projected_total_after": float(current_total + net_gain),
        "projected_total_before": current_total,
        "horizon_gameweeks": horizon_gws,
    }


def _player_record(player_lookup: pd.DataFrame, player_id: int) -> dict[str, object]:
    row = player_lookup.loc[int(player_id)]
    return {
        "player_id": int(player_id),
        "full_name": str(row["full_name"]),
        "team_name": str(row["team_name"]),
        "team_id": int(row["team_id"]),
        "element_type": int(row["element_type"]),
        "now_cost_millions": float(row["now_cost_millions"]),
        "expected_points": float(row["expected_points"]),
    }


def _apply_transfers(state: ManagerState, transfers: list[dict[str, object]]) -> None:
    if not transfers:
        return
    squad = list(state.squad_player_ids)
    for transfer in transfers:
        out_id = int(transfer["out_player"]["player_id"])
        in_id = int(transfer["in_player"]["player_id"])
        if out_id in squad:
            squad[squad.index(out_id)] = in_id
            state.bank_m += float(transfer["out_sale_value"])
            state.bank_m -= float(transfer["in_purchase_price"])
            state.purchase_price_by_player_id.pop(out_id, None)
            state.purchase_price_by_player_id[in_id] = float(transfer["in_purchase_price"])
    state.squad_player_ids = squad
    state.bank_m = round(float(state.bank_m), 2)


def _update_free_transfers_after_gameweek(
    state: ManagerState,
    transfers_made: int,
    rules: SeasonRules,
) -> None:
    spent = min(transfers_made, state.free_transfers)
    remaining = max(0, state.free_transfers - spent)
    state.free_transfers = min(
        rules.max_free_transfers,
        remaining + rules.free_transfer_per_gameweek,
    )


def _chip_candidates(
    state: ManagerState,
    predictions_by_gw: Dict[int, pd.DataFrame],
    gameweek: int,
    weekly_team: Dict[str, object],
    config: SeasonManagerConfig,
    allowed_chips: Optional[set[str]] = None,
) -> list[dict[str, object]]:
    if not config.enable_chips:
        return []

    allowed_chips = set(CHIP_NAMES if allowed_chips is None else allowed_chips)
    rules = config.rules
    current_predictions = predictions_by_gw[gameweek]
    current_points = float(weekly_team["total_expected_points_with_captain"])
    candidates: list[dict[str, object]] = []

    if "bench_boost" in allowed_chips and _chip_remaining(state, gameweek, "bench_boost", rules):
        bench_gain = float(weekly_team.get("bench_expected_points", 0.0))
        if bench_gain >= config.bench_boost_gain_threshold:
            candidates.append({"chip": "bench_boost", "gain": bench_gain})

    if "triple_captain" in allowed_chips and _chip_remaining(state, gameweek, "triple_captain", rules):
        captain_id = weekly_team.get("captain_id")
        if captain_id is not None:
            starters = pd.DataFrame(weekly_team.get("squad", []))
            captain_row = starters[starters["player_id"] == int(captain_id)]
            captain_gain = float(captain_row["expected_points"].iloc[0]) if not captain_row.empty else 0.0
            if captain_gain >= config.triple_captain_gain_threshold:
                candidates.append({"chip": "triple_captain", "gain": captain_gain})

    if (
        "free_hit" in allowed_chips
        and _can_use_free_hit(state, gameweek)
        and _chip_remaining(state, gameweek, "free_hit", rules)
    ):
        free_hit_budget = (
            _squad_sale_value(
                state.squad_player_ids,
                current_predictions,
                state.purchase_price_by_player_id,
            )
            + state.bank_m
        )
        free_hit_team = _optimise_team(
            current_predictions,
            budget_m=free_hit_budget,
            config=config,
        )
        free_hit_team = _apply_captain_flags(free_hit_team, config)
        gain = float(free_hit_team["total_expected_points_with_captain"] - current_points)
        if gain >= config.free_hit_gain_threshold:
            candidates.append({"chip": "free_hit", "gain": gain, "team": free_hit_team})

    if "wildcard" in allowed_chips and _chip_remaining(state, gameweek, "wildcard", rules):
        horizon = _available_horizon(predictions_by_gw, gameweek, config.chip_lookahead)
        aggregated = _aggregate_projection_for_gameweek(predictions_by_gw, horizon, gameweek)
        wildcard_budget = (
            _squad_sale_value(
                state.squad_player_ids,
                current_predictions,
                state.purchase_price_by_player_id,
            )
            + state.bank_m
        )
        wildcard_team = _optimise_team(
            aggregated,
            budget_m=wildcard_budget,
            config=config,
        )
        current_agg = _squad_frame_for_gw(state.squad_player_ids, aggregated)
        gain = float(wildcard_team["expected_points_without_captain"] - current_agg["expected_points"].sum())
        if gain >= config.wildcard_gain_threshold:
            candidates.append({"chip": "wildcard", "gain": gain, "team": wildcard_team})

    return sorted(candidates, key=lambda item: float(item["gain"]), reverse=True)


def _decision_record(
    gameweek: int,
    state: ManagerState,
    weekly_team: Dict[str, object],
    transfers: list[dict[str, object]],
    transfer_gain: float,
    free_transfers_before: int,
    free_transfers_after: int,
    chip: Optional[str],
    chip_gain: float,
    transfer_hit_cost: float,
    team_value_m: float,
    squad_sale_value_m: float,
) -> dict[str, object]:
    team_frame = _team_to_frame(weekly_team)
    summary = summarise_team(team_frame).as_dict()
    expected_points_before_hits = float(weekly_team["total_expected_points_with_captain"])
    expected_points = expected_points_before_hits - float(transfer_hit_cost)
    return {
        "gameweek": int(gameweek),
        "squad_player_ids": list(state.squad_player_ids),
        "starting_player_ids": [int(player["player_id"]) for player in weekly_team.get("squad", [])],
        "bench_player_ids": [int(player["player_id"]) for player in weekly_team.get("bench", [])],
        "captain": weekly_team.get("captain"),
        "captain_id": weekly_team.get("captain_id"),
        "vice_captain": weekly_team.get("vice_captain"),
        "vice_captain_id": weekly_team.get("vice_captain_id"),
        "transfers": transfers,
        "transfer_gain": float(transfer_gain),
        "free_transfers_before": int(free_transfers_before),
        "free_transfers_after": int(free_transfers_after),
        "bank_m": round(float(state.bank_m), 2),
        "team_value_m": round(float(team_value_m), 2),
        "squad_sale_value_m": round(float(squad_sale_value_m), 2),
        "chip": chip,
        "chip_gain": float(chip_gain),
        "transfer_hit_cost": float(transfer_hit_cost),
        "expected_points_before_transfer_hits": expected_points_before_hits,
        "expected_points": expected_points,
        "bench_expected_points": float(weekly_team.get("bench_expected_points", 0.0)),
        "team": summary,
    }


def _apply_scoring_chip(
    chip: str,
    weekly_team: Dict[str, object],
) -> Dict[str, object]:
    if chip == "bench_boost":
        out = dict(weekly_team)
        out["total_expected_points_with_captain"] = (
            float(out["total_expected_points_with_captain"])
            + float(out.get("bench_expected_points", 0.0))
        )
        return out
    if chip == "triple_captain":
        captain_id = weekly_team.get("captain_id")
        starters = pd.DataFrame(weekly_team.get("squad", []))
        captain_row = starters[starters["player_id"] == int(captain_id)] if captain_id else pd.DataFrame()
        captain_gain = float(captain_row["expected_points"].iloc[0]) if not captain_row.empty else 0.0
        out = dict(weekly_team)
        out["total_expected_points_with_captain"] = (
            float(out["total_expected_points_with_captain"]) + captain_gain
        )
        return out
    return weekly_team


def simulate_season(
    predictions_by_gw: Dict[int, pd.DataFrame],
    *,
    gameweeks: Optional[Iterable[int]] = None,
    config: Optional[SeasonManagerConfig] = None,
    progress_callback: Optional[Callable[[dict[str, object]], None]] = None,
    progress_context: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    """Simulate a season or partial season from prediction frames.

    The simulation starts by optimizing the initial squad across the configured
    opening horizon, then carries the same state through each gameweek.
    """

    config = config or SeasonManagerConfig()
    predictions = _normalise_prediction_map(predictions_by_gw)
    selected_gws = sorted(int(gw) for gw in (gameweeks if gameweeks is not None else predictions))
    if not selected_gws:
        raise ValueError("At least one gameweek is required")
    missing = [gw for gw in selected_gws if gw not in predictions]
    if missing:
        raise KeyError(f"Missing predictions for gameweeks: {missing}")

    first_gw = selected_gws[0]
    initial_gws = _available_horizon(predictions, first_gw, config.initial_horizon)
    initial_projection = _aggregate_projection_for_gameweek(predictions, initial_gws, first_gw)
    initial_team = _optimise_team(
        initial_projection,
        budget_m=config.rules.budget_m,
        config=config,
    )
    initial_ids = _records_to_squad_ids(initial_team)
    initial_cost = _cost_for_ids(initial_ids, predictions[first_gw])
    initial_current_prices = _current_price_lookup(predictions[first_gw])
    state = ManagerState(
        squad_player_ids=initial_ids,
        bank_m=round(config.rules.budget_m - initial_cost, 2),
        free_transfers=0,
        purchase_price_by_player_id={
            player_id: float(initial_current_prices[player_id])
            for player_id in initial_ids
        },
    )

    decisions: list[dict[str, object]] = []
    progress_context = progress_context or {}
    gameweek_count = len(selected_gws)
    for gameweek_index, gameweek in enumerate(selected_gws, start=1):
        free_transfers_before = state.free_transfers
        transfer_result = {
            "transfers": [],
            "gain": 0.0,
            "projected_total_after": 0.0,
            "projected_total_before": 0.0,
            "horizon_gameweeks": [gameweek],
        }
        transfers: list[dict[str, object]] = []
        weekly_team = _summarise_fixed_squad(state.squad_player_ids, predictions[gameweek], config)
        chip = None
        chip_gain = 0.0

        evaluate_strategic_chips = (
            config.strategic_chip_gameweeks is None
            or int(gameweek) in {int(gw) for gw in config.strategic_chip_gameweeks}
        )
        strategic_chip_options = (
            _chip_candidates(
                state,
                predictions,
                gameweek,
                weekly_team,
                config,
                allowed_chips={"free_hit", "wildcard"},
            )
            if evaluate_strategic_chips
            else []
        )
        if strategic_chip_options:
            chosen = strategic_chip_options[0]
            chip = str(chosen["chip"])
            chip_gain = float(chosen["gain"])
            _mark_chip_used(state, gameweek, chip, config.rules)
            if chip == "wildcard":
                available_budget = (
                    _squad_sale_value(
                        state.squad_player_ids,
                        predictions[gameweek],
                        state.purchase_price_by_player_id,
                    )
                    + state.bank_m
                )
                wildcard_team = chosen["team"]
                state.squad_player_ids = _records_to_squad_ids(wildcard_team)
                current_prices = _current_price_lookup(predictions[gameweek])
                state.purchase_price_by_player_id = {
                    player_id: float(current_prices[player_id])
                    for player_id in state.squad_player_ids
                }
                state.bank_m = round(
                    available_budget - _cost_for_ids(state.squad_player_ids, predictions[gameweek]),
                    2,
                )
                weekly_team = _summarise_fixed_squad(state.squad_player_ids, predictions[gameweek], config)
            elif chip == "free_hit":
                weekly_team = chosen["team"]
        else:
            if gameweek != first_gw:
                transfer_result = _transfer_candidate(state, predictions, gameweek, config)
                transfers = transfer_result["transfers"]
                _apply_transfers(state, transfers)
                weekly_team = _summarise_fixed_squad(state.squad_player_ids, predictions[gameweek], config)

            scoring_chip_options = _chip_candidates(
                state,
                predictions,
                gameweek,
                weekly_team,
                config,
                allowed_chips={"bench_boost", "triple_captain"},
            )
            if scoring_chip_options:
                chosen = scoring_chip_options[0]
                chip = str(chosen["chip"])
                chip_gain = float(chosen["gain"])
                _mark_chip_used(state, gameweek, chip, config.rules)
                weekly_team = _apply_scoring_chip(chip, weekly_team)

        if chip in {"free_hit", "wildcard"}:
            state.free_transfers = free_transfers_before
        else:
            _update_free_transfers_after_gameweek(state, len(transfers), config.rules)
        decision = _decision_record(
            gameweek=gameweek,
            state=state,
            weekly_team=weekly_team,
            transfers=transfers,
            transfer_gain=float(transfer_result["gain"]),
            free_transfers_before=free_transfers_before,
            free_transfers_after=state.free_transfers,
            chip=chip,
            chip_gain=chip_gain,
            transfer_hit_cost=float(transfer_result.get("hit_cost", 0.0)),
            team_value_m=(
                _cost_for_ids(state.squad_player_ids, predictions[gameweek])
                + state.bank_m
            ),
            squad_sale_value_m=_squad_sale_value(
                state.squad_player_ids,
                predictions[gameweek],
                state.purchase_price_by_player_id,
            ),
        )
        state.history.append(decision)
        decisions.append(decision)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "gameweek_complete",
                    "gameweek": int(gameweek),
                    "gameweek_index": gameweek_index,
                    "gameweeks_total": gameweek_count,
                    "decision": decision,
                    **progress_context,
                }
            )

    total_expected = float(sum(decision["expected_points"] for decision in decisions))
    return {
        "manager_principle": MANAGER_PRINCIPLE,
        "initial_squad": initial_ids,
        "decisions": decisions,
        "used_chips": state.used_chips,
        "summary": {
            "gameweeks": len(selected_gws),
            "start_gameweek": selected_gws[0],
            "end_gameweek": selected_gws[-1],
            "total_expected_points": total_expected,
            "transfers_made": int(sum(len(decision["transfers"]) for decision in decisions)),
            "chips_used": [
                decision["chip"]
                for decision in decisions
                if decision.get("chip")
            ],
        },
    }


def _sample_prediction_frame(
    predictions: pd.DataFrame,
    rng: np.random.Generator,
    noise_scale: float,
) -> pd.DataFrame:
    sampled = _normalise_predictions(predictions)
    lower = pd.to_numeric(sampled["expected_points_lower_80"], errors="coerce").fillna(
        sampled["expected_points"].clip(lower=0.0)
    )
    upper = pd.to_numeric(sampled["expected_points_upper_80"], errors="coerce").fillna(
        sampled["expected_points"] + 1.0
    )
    sigma = ((upper - lower).clip(lower=0.1) / (2 * 1.2815515655446004)) * noise_scale
    draws = rng.normal(sampled["expected_points"].to_numpy(dtype=float), sigma.to_numpy(dtype=float))
    sampled["expected_points"] = np.clip(draws, 0.0, None)
    return sampled


def _sample_player_points(
    player: dict[str, object],
    rng: np.random.Generator,
    noise_scale: float,
) -> float:
    mean = float(player.get("expected_points", 0.0))
    lower = float(player.get("expected_points_lower_80", max(0.0, mean - 1.0)))
    upper = float(player.get("expected_points_upper_80", mean + 1.0))
    sigma = max(0.05, ((upper - lower) / (2 * 1.2815515655446004)) * noise_scale)
    return float(max(0.0, rng.normal(mean, sigma)))


def _sample_decision_points(
    decision: dict[str, object],
    rng: np.random.Generator,
    noise_scale: float,
) -> float:
    starters = list(decision["team"].get("squad", []))
    bench = list(decision["team"].get("bench", []))
    starter_draws = {
        int(player["player_id"]): _sample_player_points(player, rng, noise_scale)
        for player in starters
    }
    total = float(sum(starter_draws.values()))

    captain_id = decision.get("captain_id")
    if captain_id is not None:
        captain_points = starter_draws.get(int(captain_id), 0.0)
        total += captain_points
        if decision.get("chip") == "triple_captain":
            total += captain_points

    if decision.get("chip") == "bench_boost":
        total += sum(_sample_player_points(player, rng, noise_scale) for player in bench)

    total -= float(decision.get("transfer_hit_cost", 0.0))
    return float(total)


def _run_fixed_policy_point_simulations(
    policy_run: dict[str, object],
    *,
    simulations: int,
    rng: np.random.Generator,
    config: SeasonManagerConfig,
    show_progress: bool,
    season_label: str,
    simulation_id_start: int = 1,
) -> dict[str, object]:
    started_at = perf_counter()
    runs: list[dict[str, object]] = []
    best_total = float("-inf")
    best_run: Optional[dict[str, object]] = None

    for idx in range(simulations):
        simulation_id = simulation_id_start + idx
        weekly_points = [
            _sample_decision_points(decision, rng, config.monte_carlo_noise_scale)
            for decision in policy_run["decisions"]
        ]
        total_points = float(sum(weekly_points))
        run_summary = {
            **policy_run["summary"],
            "total_expected_points": total_points,
            "simulation_mode": "fixed_policy",
        }
        compact_run = {
            "simulation_id": simulation_id,
            "summary": run_summary,
            "weekly_points": weekly_points,
        }
        runs.append(compact_run)

        if total_points > best_total:
            best_total = total_points
            best_run = {
                **policy_run,
                "simulation_id": simulation_id,
                "summary": run_summary,
                "weekly_points": weekly_points,
            }

        if show_progress and (
            simulation_id == 1
            or simulation_id == simulations
            or simulation_id % max(1, simulations // 20) == 0
        ):
            elapsed = perf_counter() - started_at
            average = elapsed / simulation_id
            eta = average * (simulations - simulation_id)
            print(
                f"{season_label} - simulation #{simulation_id} complete | "
                f"elapsed {_format_duration(elapsed)} | "
                f"average {average:.3f}s per simulation | "
                f"ETA {_format_duration(eta)}",
                flush=True,
            )

    totals = [float(run["summary"]["total_expected_points"]) for run in runs]
    transfers = [int(run["summary"]["transfers_made"]) for run in runs]
    return {
        "manager_principle": MANAGER_PRINCIPLE,
        "simulation_mode": "fixed_policy",
        "policy_run": policy_run,
        "runs": runs,
        "best_run": best_run,
        "summary": {
            "simulations": simulations,
            "simulation_mode": "fixed_policy",
            "average_total_expected_points": float(np.mean(totals)),
            "median_total_expected_points": float(np.median(totals)),
            "min_total_expected_points": float(np.min(totals)),
            "max_total_expected_points": float(np.max(totals)),
            "average_transfers_made": float(np.mean(transfers)),
        },
    }


def _summarize_runs(
    runs: list[dict[str, object]],
    *,
    simulations: int,
    simulation_mode: str,
    extra_summary: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    totals = [float(run["summary"]["total_expected_points"]) for run in runs]
    transfers = [int(run["summary"]["transfers_made"]) for run in runs]
    summary = {
        "simulations": simulations,
        "simulation_mode": simulation_mode,
        "average_total_expected_points": float(np.mean(totals)),
        "median_total_expected_points": float(np.median(totals)),
        "min_total_expected_points": float(np.min(totals)),
        "max_total_expected_points": float(np.max(totals)),
        "average_transfers_made": float(np.mean(transfers)),
    }
    if extra_summary:
        summary.update(extra_summary)
    return summary


def _run_periodic_reoptimization_simulations(
    base_predictions: Dict[int, pd.DataFrame],
    *,
    gameweeks: Optional[Iterable[int]],
    simulations: int,
    config: SeasonManagerConfig,
    rng: np.random.Generator,
    show_progress: bool,
    season_label: str,
    policy_refresh_interval: int,
) -> dict[str, object]:
    if policy_refresh_interval < 1:
        raise ValueError("policy_refresh_interval must be >= 1")

    started_at = perf_counter()
    runs: list[dict[str, object]] = []
    policy_runs: list[dict[str, object]] = []
    best_total = float("-inf")
    best_run: Optional[dict[str, object]] = None
    simulation_id = 1
    block_index = 0

    while simulation_id <= simulations:
        block_index += 1
        block_size = min(policy_refresh_interval, simulations - simulation_id + 1)
        if show_progress:
            print(
                f"{season_label} - optimizing AI manager policy block #{block_index} "
                f"for simulations {simulation_id}-{simulation_id + block_size - 1}",
                flush=True,
            )

        sampled_predictions = {
            gw: _sample_prediction_frame(df, rng, config.monte_carlo_noise_scale)
            for gw, df in base_predictions.items()
        }
        policy_run = simulate_season(sampled_predictions, gameweeks=gameweeks, config=config)
        policy_run["policy_block"] = block_index
        policy_run["simulation_id_start"] = simulation_id
        policy_run["simulation_id_end"] = simulation_id + block_size - 1
        policy_runs.append(policy_run)

        block_result = _run_fixed_policy_point_simulations(
            policy_run,
            simulations=block_size,
            rng=rng,
            config=config,
            show_progress=False,
            season_label=season_label,
            simulation_id_start=simulation_id,
        )
        for run in block_result["runs"]:
            run["policy_block"] = block_index
            runs.append(run)
            total_points = float(run["summary"]["total_expected_points"])
            if total_points > best_total:
                best_total = total_points
                best_run = {
                    **policy_run,
                    "simulation_id": run["simulation_id"],
                    "summary": run["summary"],
                    "weekly_points": run["weekly_points"],
                }

        simulation_id += block_size
        if show_progress:
            completed = len(runs)
            elapsed = perf_counter() - started_at
            average = elapsed / completed
            eta = average * (simulations - completed)
            print(
                f"{season_label} - completed {completed}/{simulations} simulations "
                f"after policy block #{block_index} | elapsed {_format_duration(elapsed)} | "
                f"average {average:.3f}s per simulation | ETA {_format_duration(eta)}",
                flush=True,
            )

    return {
        "manager_principle": MANAGER_PRINCIPLE,
        "simulation_mode": "periodic_reoptimization",
        "policy_runs": policy_runs,
        "runs": runs,
        "best_run": best_run,
        "summary": _summarize_runs(
            runs,
            simulations=simulations,
            simulation_mode="periodic_reoptimization",
            extra_summary={
                "policy_refresh_interval": policy_refresh_interval,
                "policy_reoptimizations": len(policy_runs),
            },
        ),
    }


def run_repeated_season_simulations(
    predictions_by_gw: Dict[int, pd.DataFrame],
    *,
    gameweeks: Optional[Iterable[int]] = None,
    simulations: int = 50,
    config: Optional[SeasonManagerConfig] = None,
    random_seed: Optional[int] = None,
    show_progress: bool = False,
    season_label: str = "FPL season",
    progress_gameweek_interval: int = 5,
    simulation_mode: str = "full_reoptimization",
    policy_refresh_interval: int = 1000,
) -> dict[str, object]:
    """Run repeated stateful season simulations with prediction uncertainty."""

    if simulations < 1:
        raise ValueError("simulations must be >= 1")
    valid_modes = {"full_reoptimization", "fixed_policy", "periodic_reoptimization"}
    if simulation_mode not in valid_modes:
        raise ValueError(f"simulation_mode must be one of {sorted(valid_modes)}")
    config = config or SeasonManagerConfig()
    base_predictions = _normalise_prediction_map(predictions_by_gw)
    rng = np.random.default_rng(random_seed)

    if simulation_mode == "fixed_policy":
        if show_progress:
            print(f"{season_label} - optimizing fixed manager policy", flush=True)
        policy_run = simulate_season(base_predictions, gameweeks=gameweeks, config=config)
        if show_progress:
            print(
                f"{season_label} - fixed policy optimized across "
                f"{policy_run['summary']['gameweeks']} gameweeks; running {simulations} point simulations",
                flush=True,
            )
        return _run_fixed_policy_point_simulations(
            policy_run,
            simulations=simulations,
            rng=rng,
            config=config,
            show_progress=show_progress,
            season_label=season_label,
        )

    if simulation_mode == "periodic_reoptimization":
        return _run_periodic_reoptimization_simulations(
            base_predictions,
            gameweeks=gameweeks,
            simulations=simulations,
            config=config,
            rng=rng,
            show_progress=show_progress,
            season_label=season_label,
            policy_refresh_interval=policy_refresh_interval,
        )

    runs: list[dict[str, object]] = []
    total_started_at = perf_counter()

    def progress_callback(event: dict[str, object]) -> None:
        if not show_progress or event.get("event") != "gameweek_complete":
            return
        gameweek_index = int(event["gameweek_index"])
        gameweeks_total = int(event["gameweeks_total"])
        if (
            gameweek_index != gameweeks_total
            and progress_gameweek_interval > 0
            and gameweek_index % progress_gameweek_interval != 0
        ):
            return
        sim_id = int(event["simulation_id"])
        gameweek = int(event["gameweek"])
        elapsed = perf_counter() - float(event["simulation_started_at"])
        print(
            f"{season_label} - simulation #{sim_id}/{simulations}: "
            f"GW{gameweek} complete ({gameweek_index}/{gameweeks_total}) "
            f"elapsed {_format_duration(elapsed)}",
            flush=True,
        )

    for idx in range(simulations):
        simulation_id = idx + 1
        simulation_started_at = perf_counter()
        if show_progress:
            print(
                f"{season_label} - simulation #{simulation_id}/{simulations} starting",
                flush=True,
            )
        sampled_predictions = {
            gw: _sample_prediction_frame(df, rng, config.monte_carlo_noise_scale)
            for gw, df in base_predictions.items()
        }
        run = simulate_season(
            sampled_predictions,
            gameweeks=gameweeks,
            config=config,
            progress_callback=progress_callback if show_progress else None,
            progress_context={
                "simulation_id": simulation_id,
                "simulation_started_at": simulation_started_at,
            },
        )
        run["simulation_id"] = simulation_id
        runs.append(run)
        if show_progress:
            elapsed_total = perf_counter() - total_started_at
            completed = idx + 1
            average_per_simulation = elapsed_total / completed
            remaining = simulations - completed
            eta = average_per_simulation * remaining
            print(
                f"{season_label} - simulation #{simulation_id}/{simulations} complete "
                f"in {_format_duration(perf_counter() - simulation_started_at)} | "
                f"average {_format_duration(average_per_simulation)} per simulation | "
                f"ETA {_format_duration(eta)}",
                flush=True,
            )

    best_run = max(runs, key=lambda run: float(run["summary"]["total_expected_points"]))
    return {
        "manager_principle": MANAGER_PRINCIPLE,
        "simulation_mode": "full_reoptimization",
        "runs": runs,
        "best_run": best_run,
        "summary": _summarize_runs(
            runs,
            simulations=simulations,
            simulation_mode="full_reoptimization",
        ),
    }


def load_prediction_files(
    output_dir: Path = OUTPUTS_DIR,
    *,
    start_gw: Optional[int] = None,
    end_gw: Optional[int] = None,
) -> dict[int, pd.DataFrame]:
    """Load ``outputs/predictions_gw<N>.csv`` files into a gameweek map."""

    frames: dict[int, pd.DataFrame] = {}
    for path in sorted(Path(output_dir).glob("predictions_gw*.csv")):
        suffix = path.stem.replace("predictions_gw", "")
        if not suffix.isdigit():
            continue
        gw = int(suffix)
        if start_gw is not None and gw < start_gw:
            continue
        if end_gw is not None and gw > end_gw:
            continue
        frames[gw] = pd.read_csv(path)
    if not frames:
        raise FileNotFoundError(f"No prediction files found in {output_dir}")
    return frames


def save_season_simulation_artifact(
    result: dict[str, object],
    output_path: Path,
) -> Path:
    """Persist a simulation result as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return output_path


__all__ = [
    "MANAGER_PRINCIPLE",
    "ManagerState",
    "SeasonManagerConfig",
    "SeasonRules",
    "load_prediction_files",
    "run_repeated_season_simulations",
    "save_season_simulation_artifact",
    "simulate_season",
]
