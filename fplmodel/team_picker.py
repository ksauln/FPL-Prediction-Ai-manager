from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd
try:
    from pulp import (
        PULP_CBC_CMD,
        LpBinary,
        LpInteger,
        LpMaximize,
        LpProblem,
        LpStatusOptimal,
        LpVariable,
        lpSum,
    )
except ImportError:  # pragma: no cover - exercised in environments without PuLP
    PULP_CBC_CMD = None
    LpBinary = None
    LpInteger = None
    LpMaximize = None
    LpProblem = None
    LpStatusOptimal = None
    LpVariable = None
    lpSum = None

from .config import (
    BENCH_GK_COUNT,
    BENCH_SIZE,
    BUDGET_MILLIONS,
    FORMATION,
    MAX_PER_TEAM,
    SQUAD_POSITION_LIMITS,
    BENCH_EP_WEIGHT,
)

POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _solve_for_formation(
    df: pd.DataFrame,
    budget_m: float,
    formation: Dict[str, int],
    *,
    current_player_ids: Optional[set[int]] = None,
    bank_m: float = 0.0,
    sale_value_by_player_id: Optional[Dict[int, float]] = None,
    max_transfers: Optional[int] = None,
    free_transfers: int = 0,
    transfer_hit_cost: float = 4.0,
) -> Dict[str, object]:
    """Solve the ILP for a specific formation."""
    if LpProblem is None:
        raise ImportError("PuLP is required for squad optimization. Install the 'pulp' package.")

    # Decision variables
    start_vars = {
        pid: LpVariable(f"x_{pid}", lowBound=0, upBound=1, cat=LpBinary)
        for pid in df["player_id"]
    }
    bench_vars = {
        pid: LpVariable(f"b_{pid}", lowBound=0, upBound=1, cat=LpBinary)
        for pid in df["player_id"]
    }
    captain_vars = {
        pid: LpVariable(f"c_{pid}", lowBound=0, upBound=1, cat=LpBinary)
        for pid in df["player_id"]
    }
    prob = LpProblem("FPL_Best_Squad", LpMaximize)

    # Objective: maximise starters (with captain double) and weighted bench EP
    ep_map = df.set_index("player_id")["expected_points"].to_dict()
    starter_points = lpSum(start_vars[pid] * ep_map[pid] for pid in ep_map)
    captain_points = lpSum(captain_vars[pid] * ep_map[pid] for pid in ep_map)
    bench_points = lpSum(bench_vars[pid] * ep_map[pid] for pid in ep_map)
    objective = starter_points + captain_points + BENCH_EP_WEIGHT * bench_points

    # Exactly 11 starters based on formation
    desired_xi = sum(formation.values())
    total_slots = sum(SQUAD_POSITION_LIMITS.values())
    if total_slots != desired_xi + BENCH_SIZE:
        raise ValueError(
            "Formation and squad constraints misaligned: "
            f"XI requires {desired_xi} slots but squad totals {total_slots} "
            f"with bench size {BENCH_SIZE}."
        )
    prob += lpSum(start_vars[pid] for pid in ep_map) == desired_xi

    # Bench size (includes reserve GK)
    prob += lpSum(bench_vars[pid] for pid in ep_map) == BENCH_SIZE

    # Captain must be a starter; exactly one captain
    prob += lpSum(captain_vars[pid] for pid in ep_map) == 1
    for pid in ep_map:
        prob += captain_vars[pid] <= start_vars[pid]
        prob += bench_vars[pid] + start_vars[pid] <= 1

    # Formation constraints for the XI
    for pos, need in formation.items():
        pids = df[df["pos_name"] == pos]["player_id"].tolist()
        prob += lpSum(start_vars[pid] for pid in pids) == need

    # Total squad composition (XI + bench)
    for pos, limit in SQUAD_POSITION_LIMITS.items():
        pids = df[df["pos_name"] == pos]["player_id"].tolist()
        if not pids:
            continue
        prob += lpSum(start_vars[pid] + bench_vars[pid] for pid in pids) == limit

    # Bench composition: enforce GK count and outfield balance
    gk_pids = df[df["pos_name"] == "GK"]["player_id"].tolist()
    if gk_pids:
        prob += lpSum(bench_vars[pid] for pid in gk_pids) == BENCH_GK_COUNT
    outfield_slots = BENCH_SIZE - BENCH_GK_COUNT
    if outfield_slots > 0:
        outfield_pids = df[df["pos_name"] != "GK"]["player_id"].tolist()
        prob += lpSum(bench_vars[pid] for pid in outfield_pids) == outfield_slots

    # Max players per team across full squad
    for team_id, grp in df.groupby("team_id"):
        pids = grp["player_id"].tolist()
        prob += lpSum(start_vars[pid] + bench_vars[pid] for pid in pids) <= MAX_PER_TEAM

    # Budget constraint for the full squad or a transition from an owned squad.
    cost_map = df.set_index("player_id")["now_cost_millions"].to_dict()
    transfers_made_expr = None
    if current_player_ids is None:
        prob += lpSum(
            (start_vars[pid] + bench_vars[pid]) * cost_map[pid] for pid in ep_map
        ) <= budget_m
    else:
        current_player_ids = {int(player_id) for player_id in current_player_ids}
        missing_current = current_player_ids - set(int(player_id) for player_id in ep_map)
        if missing_current:
            raise ValueError(f"Current squad players missing from predictions: {sorted(missing_current)}")
        sale_values = sale_value_by_player_id or {}
        selected = {pid: start_vars[pid] + bench_vars[pid] for pid in ep_map}
        incoming_ids = [pid for pid in ep_map if int(pid) not in current_player_ids]
        transfers_made_expr = lpSum(selected[pid] for pid in incoming_ids)
        if max_transfers is not None:
            prob += transfers_made_expr <= int(max_transfers)
        sale_proceeds = lpSum(
            (1 - selected[pid]) * float(sale_values.get(int(pid), cost_map[pid]))
            for pid in ep_map
            if int(pid) in current_player_ids
        )
        purchase_cost = lpSum(selected[pid] * cost_map[pid] for pid in incoming_ids)
        prob += purchase_cost <= float(bank_m) + sale_proceeds

        paid_transfers = LpVariable("paid_transfers", lowBound=0, cat=LpInteger)
        prob += paid_transfers >= transfers_made_expr - int(free_transfers)
        objective -= float(transfer_hit_cost) * paid_transfers

    prob += objective

    status = prob.solve(PULP_CBC_CMD(msg=False))
    if status != LpStatusOptimal:
        raise RuntimeError("No optimal XI found. Try adjusting budget or formation.")

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
        "appearance_probability",
        "availability_this_round",
        "availability_next_round",
        "status_availability",
        "fixture_multiplier",
        "cameo_points",
        "confidence_score",
        "confidence_level",
        "expected_points_lower_80",
        "expected_points_upper_80",
        "official_ep_next",
        "price_change_percent",
    ]
    base_columns.extend(col for col in optional_columns if col in df.columns)
    selected = df[base_columns].copy()
    selected["starting"] = selected["player_id"].apply(
        lambda pid: int(start_vars[pid].value() or 0)
    )
    selected["bench"] = selected["player_id"].apply(
        lambda pid: int(bench_vars[pid].value() or 0)
    )
    selected["captain"] = selected["player_id"].apply(
        lambda pid: int(captain_vars[pid].value() or 0)
    )

    squad = selected[selected["starting"] == 1].sort_values(
        ["element_type", "expected_points"], ascending=[True, False]
    )

    bench = selected[selected["bench"] == 1].copy()
    bench_outfield = bench[bench["element_type"] != 1].sort_values(
        "expected_points", ascending=False
    )
    bench_gk = bench[bench["element_type"] == 1].sort_values(
        "expected_points", ascending=False
    )
    bench_ordered = pd.concat([bench_outfield, bench_gk], ignore_index=True)
    bench_ordered["bench_order"] = bench_ordered.index + 1

    starting_cost = squad["now_cost_millions"].sum()
    bench_cost = bench_ordered["now_cost_millions"].sum()
    total_cost = starting_cost + bench_cost

    base_ep = squad["expected_points"].sum()
    cap_ep = (
        squad.loc[squad["captain"] == 1, "expected_points"].sum()
        if (squad["captain"] == 1).any()
        else 0.0
    )
    total_ep = base_ep + cap_ep

    result = {
        "squad": squad.to_dict(orient="records"),
        "bench": bench_ordered.to_dict(orient="records"),
        "total_cost": float(total_cost),
        "starting_cost": float(starting_cost),
        "bench_cost": float(bench_cost),
        "expected_points_without_captain": float(base_ep),
        "total_expected_points_with_captain": float(total_ep),
        "bench_expected_points": float(bench_ordered["expected_points"].sum()),
        "captain": squad.loc[squad["captain"] == 1, "full_name"].iloc[0]
        if (squad["captain"] == 1).any()
        else None,
    }
    if current_player_ids is not None:
        selected_ids = set(selected.loc[selected["starting"].eq(1) | selected["bench"].eq(1), "player_id"])
        transfers_made = len(selected_ids - current_player_ids)
        paid_transfers = max(0, transfers_made - int(free_transfers))
        result["transfers_made"] = transfers_made
        result["optimization_score"] = float(
            total_ep
            + BENCH_EP_WEIGHT * float(bench_ordered["expected_points"].sum())
            - float(transfer_hit_cost) * paid_transfers
        )
    else:
        result["optimization_score"] = float(
            total_ep + BENCH_EP_WEIGHT * float(bench_ordered["expected_points"].sum())
        )
    return result


def pick_best_xi(
    pred_df: pd.DataFrame,
    budget_m: float = BUDGET_MILLIONS,
    formation: Optional[Dict[str, int]] = None,
    formations: Optional[Iterable[Dict[str, int]]] = None,
    current_player_ids: Optional[Iterable[int]] = None,
    bank_m: float = 0.0,
    sale_value_by_player_id: Optional[Dict[int, float]] = None,
    max_transfers: Optional[int] = None,
    free_transfers: int = 0,
    transfer_hit_cost: float = 4.0,
) -> Dict[str, object]:
    """
    ILP: pick best XI with bench under budget, formation, per-team, and squad constraints.

    Also selects a captain (one of the XI) to maximise EP (captain doubles).
    `pred_df` must include: player_id, team_id, team_name, element_type,
    now_cost_millions, expected_points, full_name.
    """
    if formation is not None and formations is not None:
        raise ValueError("Provide either formation or formations, not both.")

    if formations is None:
        formations = [formation or FORMATION]
    else:
        formations = list(formations)
        if not formations:
            raise ValueError("formations must be a non-empty iterable of dicts.")

    df = pred_df.copy()
    df["pos_name"] = df["element_type"].map(POS_MAP)

    best_result: Optional[Dict[str, object]] = None
    best_points = float("-inf")
    last_error: Optional[Exception] = None
    current_ids = (
        {int(player_id) for player_id in current_player_ids}
        if current_player_ids is not None
        else None
    )

    for current in formations:
        try:
            result = _solve_for_formation(
                df,
                budget_m,
                current,
                current_player_ids=current_ids,
                bank_m=bank_m,
                sale_value_by_player_id=sale_value_by_player_id,
                max_transfers=max_transfers,
                free_transfers=free_transfers,
                transfer_hit_cost=transfer_hit_cost,
            )
        except RuntimeError as exc:
            last_error = exc
            continue

        current_points = result["optimization_score"]
        if current_points > best_points:
            best_points = current_points
            result["formation"] = current.copy()
            result["formation_name"] = f"{current.get('DEF', 0)}-{current.get('MID', 0)}-{current.get('FWD', 0)}"
            best_result = result

    if best_result is None:
        if last_error is not None:
            raise RuntimeError(
                f"Unable to find optimal XI for provided formations: {last_error}"
            ) from last_error
        raise RuntimeError("No formation produced a valid XI.")

    return best_result


__all__ = ["pick_best_xi"]
