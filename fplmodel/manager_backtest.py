"""Deadline-safe, read-only manager competition using actual scored outcomes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import json
from itertools import combinations

import numpy as np
import pandas as pd

from . import season_manager as legacy
from .manager_v2 import _json_safe, run_v2_simulations


def score_actual_decision(decision, outcomes):
    """Score actual outcomes with bench priority and legal lineup slots.

    Unreplaced DNP starters still occupy their formation slots. Counting only
    players who appeared would incorrectly suppress valid partial autosubs.
    This evaluator does not modify either manager's planning/simulation logic.
    """
    starters = list(decision["team"]["squad"])
    bench = sorted(decision["team"].get("bench", []), key=lambda p: int(p.get("bench_order", 99)))
    played = lambda p: bool(outcomes[int(p["player_id"])][0])
    scoring = [p for p in starters if played(p)]
    if decision.get("chip") == "bench_boost":
        scoring.extend(p for p in bench if played(p))
    else:
        if not any(int(p["element_type"]) == 1 and played(p) for p in starters):
            reserve = next((p for p in bench if int(p["element_type"]) == 1 and played(p)), None)
            if reserve is not None:
                scoring.append(reserve)
        missing = [p for p in starters if int(p["element_type"]) != 1 and not played(p)]
        reserves = [p for p in bench if int(p["element_type"]) != 1 and played(p)]
        chosen = None
        for count in range(min(len(missing), len(reserves)), 0, -1):
            for incoming in combinations(reserves, count):
                for outgoing in combinations(missing, count):
                    removed = {int(p["player_id"]) for p in outgoing}
                    slots = [p for p in starters if int(p["player_id"]) not in removed] + list(incoming)
                    positions = {code: sum(int(p["element_type"]) == code for p in slots) for code in (1, 2, 3, 4)}
                    if positions[1] == 1 and positions[2] >= 3 and positions[3] >= 2 and positions[4] >= 1:
                        chosen = incoming
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                scoring.extend(chosen)
                break
    total = sum(float(outcomes[int(p["player_id"])][1]) for p in scoring)
    for captain in (decision.get("captain_id"), decision.get("vice_captain_id")):
        if captain is not None and outcomes[int(captain)][0]:
            total += float(outcomes[int(captain)][1]) * (2 if decision.get("chip") == "triple_captain" else 1)
            break
    return float(total - float(decision.get("transfer_hit_cost", 0.)))


@dataclass(frozen=True)
class ForecastVintage:
    origin_gw: int
    season_name: str
    created_at: str
    inputs_as_of: str
    trained_through_gw: int
    predictions: dict
    deadline: str

    def validate(self):
        def timestamp(value):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("Forecast provenance timestamps require a timezone")
            return parsed
        deadline = timestamp(self.deadline)
        created, inputs = timestamp(self.created_at), timestamp(self.inputs_as_of)
        if created >= deadline or inputs > created:
            raise ValueError("Forecast or inputs are not verifiably before the deadline")
        if self.trained_through_gw >= self.origin_gw:
            raise ValueError("Forecast model was trained on the evaluation gameweek")
        if self.origin_gw not in self.predictions:
            raise ValueError("Vintage is missing its origin gameweek")
        base = self.predictions[self.origin_gw].set_index("player_id")
        if not base.index.is_unique:
            raise ValueError("Duplicate vintage player IDs")
        for gw, frame in self.predictions.items():
            if gw < self.origin_gw:
                raise ValueError("Vintage includes past forecasts")
            if "season_name" not in frame or set(frame.season_name.dropna().astype(str)) != {self.season_name}:
                raise ValueError("Vintage prediction season does not match manifest")
            if "gameweek" not in frame or not pd.to_numeric(frame.gameweek).eq(gw).all():
                raise ValueError("Vintage gameweek metadata does not match manifest")
            future = frame.set_index("player_id")
            if not future.index.is_unique or set(future.index) != set(base.index):
                raise ValueError("Vintage player universe changes across future gameweeks")
            for column in ("now_cost_millions", "team_id", "element_type"):
                if not future[column].reindex(base.index).equals(base[column]):
                    raise ValueError("Future price/club/position metadata must reflect the origin snapshot")


@dataclass(frozen=True)
class ReconstructedVintage(ForecastVintage):
    """Origin forecast is original; future horizons were rebuilt afterward.

    ``created_at`` records the original origin forecast's generation, while
    ``reconstructed_at`` explicitly records when the additional forecasts were
    made. These must never be reported as retained pre-deadline vintages.
    """

    reconstructed_at: str = ""
    history_through_gw: int = 0

    def validate(self):
        super().validate()
        rebuilt = datetime.fromisoformat(self.reconstructed_at.replace("Z", "+00:00"))
        if rebuilt.tzinfo is None or self.history_through_gw >= self.origin_gw:
            raise ValueError("Reconstruction must record its time and exclude evaluation-week history")


def load_vintage(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    data = {}
    for gw, name in payload["predictions"].items():
        source = (path.parent / name).resolve()
        if not source.is_relative_to(path.parent.resolve()):
            raise ValueError("Vintage prediction paths must stay inside their directory")
        data[int(gw)] = pd.read_csv(source)
    vintage = ForecastVintage(**{k: payload[k] for k in (
        "origin_gw", "season_name", "created_at", "inputs_as_of", "trained_through_gw", "deadline")}, predictions=data)
    vintage.validate()
    return vintage


def historical_competition(vintages, actuals, *, initial_state, upgraded_config=None,
                           legacy_config=None, simulations=128, seed=90,
                           allow_reconstructed=False, progress_callback=None,
                           v2_runner=None):
    """Replan independently each deadline, reveal actual outcomes only afterward.

    Vintages must be produced from inputs available at their origin deadline.
    Predicted future prices stay frozen at that origin; realized future prices
    appear only when the next vintage is revealed. This function writes nothing.
    """
    config = upgraded_config or legacy.SeasonManagerConfig(manager_version="v2", max_transfers_per_gw=5)
    runner = v2_runner or run_v2_simulations
    baseline_config = legacy_config or legacy.SeasonManagerConfig(rules=config.rules)
    if baseline_config.rules != config.rules:
        raise ValueError("Both managers must use identical season rules")
    weeks = sorted(vintages)
    if not weeks or weeks != list(range(weeks[0], weeks[-1] + 1)):
        raise ValueError("Historical competition requires consecutive vintages")
    if initial_state.last_processed_gameweek + 1 != weeks[0]:
        raise ValueError("Competition must start immediately after initial_state")
    for gw in weeks:
        vintage = vintages[gw]
        if isinstance(vintage, ReconstructedVintage) and not allow_reconstructed:
            raise ValueError("Reconstructed horizons require explicit allow_reconstructed=True")
        vintage.validate()
        if vintage.origin_gw != gw or vintage.season_name != initial_state.season_name:
            raise ValueError("Vintage origin/season does not match competition")
        needed = set(vintage.predictions[gw].player_id.astype(int))
        if needed - set(actuals.get(gw, {})):
            raise ValueError(f"Missing explicit actual outcomes, including DNPs, for GW{gw}")
        if any(not np.isfinite(float(points)) for _, points in actuals[gw].values()):
            raise ValueError("Actual points must be finite")
        if any(not isinstance(played, (bool, np.bool_)) or (not played and float(points) != 0.)
               for played, points in actuals[gw].values()):
            raise ValueError("Actual appearances must be boolean and DNP points must be zero")
    states = {name: legacy.ManagerState.from_dict(initial_state.to_dict()) for name in ("legacy", "v2", "hold")}
    reports = {name: [] for name in states}
    for gw in weeks:
        data = vintages[gw].predictions
        for name, state in states.items():
            if progress_callback:
                progress_callback({"gameweek": gw, "manager": name, "phase": "planning"})
            # All managers share a preselected initial squad. It cannot incur
            # transfers/hits before GW1; only lineup/captain/chip decisions apply.
            v2_config = replace(config, max_transfers_per_gw=0) if gw == 1 else config
            if name == "v2":
                result = runner(data, gameweeks=[gw], config=v2_config,
                    initial_state=state, simulations=simulations, random_seed=seed + gw)
                decision = result["recommended_policy"]["decisions"][0]
            else:
                old_config = replace(baseline_config, manager_version="legacy",
                                     max_transfers_per_gw=(0 if name == "hold" or gw == 1 else baseline_config.max_transfers_per_gw),
                                     enable_chips=(False if name == "hold" else baseline_config.enable_chips))
                result = legacy.simulate_season(data, gameweeks=[gw], config=old_config, initial_state=state)
                decision = result["decisions"][0]
            score = score_actual_decision(decision, actuals[gw])
            plain = deepcopy(decision)
            plain["captain_id"] = plain["vice_captain_id"] = None
            captain_bonus = score - score_actual_decision(plain, actuals[gw])
            no_chip = deepcopy(decision)
            no_chip["chip"] = None
            chip_gain = score - score_actual_decision(no_chip, actuals[gw])
            counterfactual_team = legacy._summarise_fixed_squad(state.squad_player_ids, data[gw], replace(config, manager_version="legacy"))
            counterfactual = {"team": {"squad": counterfactual_team["squad"], "bench": counterfactual_team["bench"]},
                              "captain_id": counterfactual_team["captain_id"], "vice_captain_id": counterfactual_team["vice_captain_id"],
                              "chip": None, "transfer_hit_cost": 0.}
            hold_score = score_actual_decision(counterfactual, actuals[gw])
            if decision["chip"] in {"wildcard", "free_hit"}:
                chip_gain = score - hold_score
            reports[name].append({"gameweek": gw, "net_actual_points": score,
                "captain_bonus": captain_bonus, "chip": decision["chip"], "chip_week_gain": chip_gain,
                "transfer_hit_cost": decision["transfer_hit_cost"],
                "transfers": len(decision["transfers"]),
                "gain_vs_holding_this_week": score - hold_score,
                "decision": decision})
            states[name] = legacy.ManagerState.from_dict(decision["manager_state_after"])
            if progress_callback:
                progress_callback({"gameweek": gw, "manager": name, "phase": "scored",
                                   "actual_points": score, "chip": decision["chip"],
                                   "transfers": len(decision["transfers"])})
    managers = {name: {"gameweeks": weeks, "weekly": rows,
        "net_actual_points": sum(r["net_actual_points"] for r in rows),
        "captain_bonus": sum(r["captain_bonus"] for r in rows),
        "chip_week_gain": sum(r["chip_week_gain"] for r in rows),
        "transfer_hit_cost": sum(r["transfer_hit_cost"] for r in rows),
        "transfers": sum(r["transfers"] for r in rows),
        "final_state": states[name].to_dict()} for name, rows in reports.items()}
    reconstructed = any(isinstance(v, ReconstructedVintage) for v in vintages.values())
    return _json_safe({"deadline_safe": not reconstructed,
            "as_of_reconstruction": reconstructed,
            "initial_state": initial_state.to_dict(), "managers": managers,
            "v2_gain_over_legacy": managers["v2"]["net_actual_points"] - managers["legacy"]["net_actual_points"],
            "v2_gain_over_hold": managers["v2"]["net_actual_points"] - managers["hold"]["net_actual_points"],
            "metric_note": "Chip and holding comparisons are same-week counterfactuals, not full-season causal attribution."})
