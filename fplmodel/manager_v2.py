"""Opt-in rolling-horizon manager. No live-state writes occur in this module."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math

import numpy as np
import pandas as pd

from . import season_manager as legacy
from .manager_scenarios import compare_shared_futures, normalise_forecasts


def validate_config(config):
    if config.planning_beam_width < 1:
        raise ValueError("planning beam width must be positive")
    if not 2 <= config.planning_candidates <= 30:
        raise ValueError("planning_candidates must be between 2 and 30")
    for name in ("transfer_horizon", "initial_horizon", "chip_lookahead"):
        if not 1 <= getattr(config, name) <= 38:
            raise ValueError(f"{name} must be between 1 and 38")
    if not 0 <= config.max_transfers_per_gw <= 15:
        raise ValueError("max_transfers_per_gw must be between 0 and 15")
    if not math.isfinite(config.free_transfer_future_value) or config.free_transfer_future_value < 0:
        raise ValueError("free_transfer_future_value must be nonnegative and finite")
    if not 0 <= config.scenario_injury_probability <= 1:
        raise ValueError("scenario_injury_probability must be between 0 and 1")
    if not 0 <= config.scenario_team_correlation < 1:
        raise ValueError("scenario_team_correlation must be between 0 and 1 (exclusive)")
    if not math.isfinite(config.paid_transfer_gain_buffer) or config.paid_transfer_gain_buffer < 0:
        raise ValueError("paid_transfer_gain_buffer must be nonnegative and finite")
    if not 0 <= config.paid_transfer_min_probability <= 1:
        raise ValueError("paid_transfer_min_probability must be between 0 and 1")


def _paid_transfers(decision, config):
    cost = float(decision.get("transfer_hit_cost", 0.))
    price = float(config.rules.transfer_hit_cost)
    return int(round(cost / price)) if cost > 0 and price > 0 else 0


def _planning_rank(plan, mean):
    return mean + plan.get("terminal_transfer_value", 0.) - plan.get("risk_adjustment", 0.)


def select_finalists(alternatives, selection, config):
    """Always validate the no-hit alternatives, even below the top paid plans."""
    ranked = sorted(range(len(alternatives)),
                    key=lambda i: _planning_rank(alternatives[i], selection[i]["average_total_points"]), reverse=True)
    required = {i for i, plan in enumerate(alternatives)
                if not _paid_transfers(plan["decisions"][0], config)}
    required.update(ranked[:3])
    # Hold stays first so existing paired hold diagnostics remain meaningful.
    return [i for i in range(len(alternatives)) if i in required]


def assess_paid_transfers(alternatives, metrics, config):
    """Paid moves need marginal net benefit over the best no-hit continuation.

    Scenario totals already deduct transfer hits. The buffer is a decision
    margin, not another points charge. Terminal FT estimates do not count as
    evidence that a hit will produce more scored points.
    """
    if not alternatives or len(alternatives) != len(metrics):
        raise ValueError("A matching result is required for every alternative")
    free = [i for i, plan in enumerate(alternatives)
            if not _paid_transfers(plan["decisions"][0], config)]
    if not free:
        raise ValueError("A no-hit alternative is required")
    rank = lambda i: _planning_rank(alternatives[i], metrics[i]["average_total_points"])
    reference = max(free, key=lambda i: (metrics[i]["average_total_points"], rank(i)))
    baseline = np.asarray(metrics[reference]["scores"], dtype=float)
    diagnostics, eligible = [], []
    for i, (plan, result) in enumerate(zip(alternatives, metrics)):
        difference = np.asarray(result["scores"], dtype=float) - baseline
        paid = _paid_transfers(plan["decisions"][0], config)
        required = config.transfer_gain_threshold + paid * config.paid_transfer_gain_buffer if paid else 0.
        gain, probability = float(difference.mean()), float(np.mean(difference > 0))
        allowed = not paid or (gain >= required and probability >= config.paid_transfer_min_probability)
        reasons = []
        if paid and gain < required:
            reasons.append("Marginal net gain over the free alternative is too small")
        if paid and probability < config.paid_transfer_min_probability:
            reasons.append("Paid move is too uncertain against the free alternative")
        diagnostics.append({"action": plan["action"], "free_reference_action": alternatives[reference]["action"],
            "paid_transfers_now": paid, "hit_cost_now": float(plan["decisions"][0].get("transfer_hit_cost", 0.)),
            "gain_over_free": gain, "probability_beats_free": probability,
            "required_paid_gain": required, "paid_transfer_eligible": bool(allowed),
            "rejection_reasons": reasons})
        if allowed:
            eligible.append(i)
    winner = max(eligible, key=rank)
    hold = next((i for i, plan in enumerate(alternatives) if plan["action"] == "hold"), None)
    if hold is not None and winner != hold and rank(winner) - rank(hold) < config.transfer_gain_threshold:
        winner = hold
    return {"winner": winner, "free_reference": reference, "diagnostics": diagnostics}


def choose_captains(starters):
    """Maximize captain bonus including vice fallback; EP already includes DNPs.

    This deterministic approximation assumes independent appearances. Shared
    scenario evaluation subsequently compares the full policy with dependence.
    """
    rows = starters.to_dict("records")
    if not rows:
        return None, None, 0.
    best = None
    for captain in rows:
        appearance = float(captain.get("appearance_probability", 1.))
        if float(captain.get("fixture_multiplier", 1.)) == 0:
            appearance = 0.
        for vice in rows:
            if captain["player_id"] == vice["player_id"] and len(rows) > 1:
                continue
            bonus = float(captain["expected_points"]) + (1 - np.clip(appearance, 0, 1)) * float(vice["expected_points"])
            key = (bonus, float(captain["expected_points"]), -int(captain["player_id"]), -int(vice["player_id"]))
            if best is None or key > best[0]:
                best = (key, int(captain["player_id"]), int(vice["player_id"]), bonus)
    return best[1:]


class _Search:
    def __init__(self, data, config):
        self.data, self.config = data, config
        self.lineups, self.actions = {}, {}
        self.packages, self.fresh_squads = {}, {}

    def lineup(self, ids, gw, chip=None):
        key = (tuple(sorted(ids)), gw, chip)
        if key in self.lineups:
            return deepcopy(self.lineups[key])
        frame = legacy._squad_frame_for_gw(ids, self.data[gw])
        best, best_value = None, -math.inf
        for formation in self.config.formations:
            team = legacy._pick_lineup_from_owned_squad(frame, replace(self.config, formations=(formation,)))
            captain, vice, bonus = choose_captains(pd.DataFrame(team["squad"]))
            names = frame.set_index("player_id")["full_name"].to_dict()
            for section in ("squad", "bench"):
                for player in team[section]:
                    player["captain"] = int(player["player_id"] == captain)
                    player["vice_captain"] = int(player["player_id"] == vice)
            team.update(captain_id=captain, vice_captain_id=vice,
                        captain=names.get(captain), vice_captain=names.get(vice),
                        captain_expected_bonus=float(bonus))
            value = float(team["expected_points_without_captain"]) + bonus
            if chip == "bench_boost":
                value += float(team["bench_expected_points"])
            elif chip == "triple_captain":
                value += bonus
            team["total_expected_points_with_captain"] = value
            if value > best_value:
                best, best_value = team, value
        self.lineups[key] = best
        return deepcopy(best)

    def transfers(self, state, target, gw):
        lookup = self.data[gw].set_index("player_id")
        old, new = set(state.squad_player_ids), set(target)
        swaps = []
        for position in (1, 2, 3, 4):
            outs = sorted(pid for pid in old - new if int(lookup.loc[pid, "element_type"]) == position)
            ins = sorted(pid for pid in new - old if int(lookup.loc[pid, "element_type"]) == position)
            if len(outs) != len(ins):
                raise ValueError("Transfer package changes squad composition")
            for outgoing, incoming in zip(outs, ins):
                out = legacy._player_record(lookup, outgoing)
                buy = legacy._player_record(lookup, incoming)
                purchase = state.purchase_price_by_player_id.get(outgoing, float(out["now_cost_millions"]))
                swaps.append({"out_player": out, "in_player": buy,
                              "out_purchase_price": purchase,
                              "out_sale_value": legacy._sale_value_millions(purchase, out["now_cost_millions"]),
                              "in_purchase_price": buy["now_cost_millions"],
                              "expected_points_delta": buy["expected_points"] - out["expected_points"]})
        return legacy._order_transfers_for_execution(swaps, state.bank_m)

    def candidates(self, state, gw, remaining, *, allow_transfers=True):
        key = (gw, tuple(sorted(state.squad_player_ids)), state.bank_m, state.free_transfers,
               tuple(sorted(state.purchase_price_by_player_id.items())),
               repr(state.used_chips), state.history[-1].get("chip") if state.history else None,
               tuple(remaining), allow_transfers)
        if key in self.actions:
            return self.actions[key]
        owned = set(state.squad_player_ids)
        candidates = [{"action": "hold", "target": sorted(owned), "chip": None}]
        current = self.data[gw]
        prices = legacy._current_price_lookup(current)
        sale = {pid: legacy._sale_value_millions(state.purchase_price_by_player_id.get(pid, prices[pid]), prices[pid])
                for pid in owned}
        horizons = [[gw]]
        if len(remaining) > 1:
            horizons.append(remaining)
        for horizon in horizons:
            aggregate = legacy._aggregate_projection_for_gameweek(self.data, horizon, gw)
            if allow_transfers and self.config.max_transfers_per_gw:
                limit = self.config.max_transfers_per_gw
                counts = sorted({1, min(2, limit), min(max(1, state.free_transfers), limit), limit})
                for count in counts:
                    package_key = (gw, tuple(horizon), tuple(sorted(owned)), state.bank_m,
                                   state.free_transfers, tuple(sorted(sale.items())), count)
                    try:
                        if package_key not in self.packages:
                            team = legacy.pick_best_xi(
                                aggregate, formations=self.config.formations, joint_formations=True,
                                current_player_ids=owned, bank_m=state.bank_m,
                                sale_value_by_player_id=sale, max_transfers=count,
                                free_transfers=state.free_transfers,
                                transfer_hit_cost=self.config.rules.transfer_hit_cost,
                            )
                            self.packages[package_key] = sorted(legacy._records_to_squad_ids(team))
                    except RuntimeError:
                        continue
                    target = self.packages[package_key]
                    if set(target) != owned:
                        candidates.append({"action": f"transfer_{len(set(target) - owned)}", "target": target, "chip": None})
            if not self.config.enable_chips:
                continue
            if self.config.strategic_chip_gameweeks is not None and gw not in self.config.strategic_chip_gameweeks:
                continue
            budget = sum(sale.values()) + state.bank_m
            for chip in ("wildcard", "free_hit"):
                if gw == 1 or not legacy._chip_remaining(state, gw, chip, self.config.rules):
                    continue
                if chip == "free_hit" and (not legacy._can_use_free_hit(state, gw) or horizon != [gw]):
                    continue
                try:
                    fresh_key = (gw, tuple(horizon), round(budget, 6))
                    if fresh_key not in self.fresh_squads:
                        team = legacy.pick_best_xi(aggregate, budget_m=budget,
                            formations=self.config.formations, joint_formations=True)
                        self.fresh_squads[fresh_key] = sorted(legacy._records_to_squad_ids(team))
                except RuntimeError:
                    continue
                candidates.append({"action": chip, "target": self.fresh_squads[fresh_key], "chip": chip})
        unique = {}
        for action in candidates:
            unique[(tuple(action["target"]), action["chip"])] = action
        candidates = list(unique.values())
        # Score transfer candidates by separate weekly lineups, not summed XI.
        def estimate(action):
            cost = max(0, len(set(action["target"]) - owned) - state.free_transfers) * self.config.rules.transfer_hit_cost
            if cost > 0:
                cost += cost / self.config.rules.transfer_hit_cost * self.config.paid_transfer_gain_buffer
            if action["chip"] in {"wildcard", "free_hit"}:
                cost = 0.
            return sum(float(self.lineup(action["target"], week)["total_expected_points_with_captain"])
                       for week in (remaining if action["chip"] != "free_hit" else [gw])) - cost
        regular = [a for a in candidates if a["action"] != "hold" and a["chip"] is None]
        regular.sort(key=estimate, reverse=True)
        strategic = [a for a in candidates if a["chip"]]
        shortlisted = regular[:self.config.planning_candidates - 1]
        no_hit = next((a for a in regular if len(set(a["target"]) - owned) <= state.free_transfers
                       or self.config.rules.transfer_hit_cost == 0), None)
        if no_hit is not None and no_hit not in shortlisted:
            shortlisted[-1] = no_hit
        candidates = [candidates[0]] + shortlisted + strategic
        # Scoring chips can accompany any regular transfer package.
        for base in list(candidates):
            if base["chip"] is not None or not self.config.enable_chips:
                continue
            for chip in ("bench_boost", "triple_captain"):
                if legacy._chip_remaining(state, gw, chip, self.config.rules):
                    candidates.append({**base, "action": f"{base['action']}+{chip}", "chip": chip})
        self.actions[key] = candidates
        return candidates

    def transition(self, state, action, gw):
        updated = legacy.ManagerState.from_dict(state.to_dict(include_history=False))
        before_ids, ft_before = list(state.squad_player_ids), state.free_transfers
        chip = action["chip"]
        swaps = self.transfers(state, action["target"], gw) if chip != "free_hit" else []
        if swaps:
            legacy._apply_transfers(updated, swaps)
        if chip:
            legacy._mark_chip_used(updated, gw, chip, self.config.rules)
        if chip in {"wildcard", "free_hit"}:
            legacy._apply_free_transfer_topup(updated, gw + 1, self.config.rules)
            hit = 0.
        else:
            hit = max(0, len(swaps) - ft_before) * self.config.rules.transfer_hit_cost
            legacy._update_free_transfers_after_gameweek(updated, len(swaps), self.config.rules, gw)
        updated.last_processed_gameweek = gw
        ids = action["target"] if chip == "free_hit" else updated.squad_player_ids
        team = self.lineup(ids, gw, chip)
        baseline = self.lineup(before_ids, gw)
        value = float(team["total_expected_points_with_captain"]) - hit
        decision = legacy._decision_record(
            gameweek=gw, state=updated, weekly_team=team, transfers=swaps,
            transfer_gain=value - float(baseline["total_expected_points_with_captain"]),
            free_transfers_before=ft_before, free_transfers_after=updated.free_transfers,
            chip=chip, chip_gain=(value - float(baseline["total_expected_points_with_captain"])) if chip else 0.,
            transfer_hit_cost=hit,
            team_value_m=legacy._cost_for_ids(updated.squad_player_ids, self.data[gw]) + updated.bank_m,
            squad_sale_value_m=legacy._squad_sale_value(updated.squad_player_ids, self.data[gw], updated.purchase_price_by_player_id),
            squad_player_ids_before_decision=before_ids,
            non_chip_baseline_player_ids=before_ids, non_chip_baseline_hit_cost=0.,
        )
        decision["action"] = action["action"]
        decision["paid_transfers_now"] = _paid_transfers(decision, self.config)
        decision["paid_transfer_buffer"] = decision["paid_transfers_now"] * self.config.paid_transfer_gain_buffer
        decision["planned_action"] = deepcopy(action)
        # summarise_team recalculates a simple captain total; publish the v2 pair value.
        decision["team"]["captain_expected_bonus"] = team["captain_expected_bonus"]
        # Keep the previous chip during beam search (including across the half
        # boundary) without repeatedly copying the entire committed history.
        updated.history = [{"gameweek": gw, "chip": chip}]
        return updated, decision


@dataclass
class _Node:
    state: legacy.ManagerState
    decisions: list
    points: float = 0.
    risk_adjustment: float = 0.


def _terminal_value(state, last_gw, config):
    if last_gw >= 38:
        return 0.
    # Diminishing value near the carry cap; no value after season end.
    return config.free_transfer_future_value * sum(1 / (1 + .25 * i) for i in range(state.free_transfers))


def plan_gameweek(predictions_by_gw, state, gameweek, config, *, allow_transfers=True, search=None):
    """Keep a bounded beam per first action so hold/delayed moves survive pruning."""
    validate_config(config)
    data = normalise_forecasts(predictions_by_gw) if search is None else search.data
    search = search or _Search(data, config)
    horizon = max(config.transfer_horizon, config.chip_lookahead if config.enable_chips else 1)
    weeks = legacy._available_horizon(data, gameweek, horizon)
    if weeks != list(range(gameweek, gameweek + len(weeks))):
        raise ValueError("V2 planning requires consecutive gameweeks")
    roots = search.candidates(state, gameweek, weeks, allow_transfers=allow_transfers)
    alternatives = []
    for root in roots:
        next_state, decision = search.transition(state, root, gameweek)
        beam = [_Node(next_state, [decision], float(decision["expected_points"]), decision["paid_transfer_buffer"])]
        for offset, gw in enumerate(weeks[1:], 1):
            expanded = []
            for node in beam:
                for action in search.candidates(node.state, gw, weeks[offset:]):
                    candidate_state, candidate_decision = search.transition(node.state, action, gw)
                    expanded.append(_Node(candidate_state, node.decisions + [candidate_decision],
                                          node.points + float(candidate_decision["expected_points"]),
                                          node.risk_adjustment + candidate_decision["paid_transfer_buffer"]))
            expanded.sort(key=lambda n: n.points - n.risk_adjustment + _terminal_value(n.state, gw, config), reverse=True)
            safe = next((n for n in expanded if not any(d["transfer_hit_cost"] for d in n.decisions[1:])), None)
            seen = set()
            beam = []
            for node in expanded:
                signature = (tuple(sorted(node.state.squad_player_ids)), node.state.bank_m,
                             node.state.free_transfers, repr(node.state.used_chips),
                             node.state.history[-1].get("chip") if node.state.history else None,
                             tuple(sorted(node.state.purchase_price_by_player_id.items())))
                if signature not in seen:
                    seen.add(signature)
                    beam.append(node)
                if len(beam) == config.planning_beam_width:
                    break
            if safe is not None and not any(node is safe for node in beam):
                beam[-1] = safe
        best = max(beam, key=lambda n: n.points - n.risk_adjustment + _terminal_value(n.state, weeks[-1], config))
        terminal = _terminal_value(best.state, weeks[-1], config)
        alternatives.append({"action": root["action"], "decisions": best.decisions,
                             "root_action": root,
                             "risk_adjustment": best.risk_adjustment,
                             "expected_points": best.points, "terminal_transfer_value": terminal,
                             "objective": best.points - best.risk_adjustment + terminal})
    reference = next(a for a in alternatives if a["action"] == "hold")
    free = max((a for a in alternatives if not _paid_transfers(a["decisions"][0], config)),
               key=lambda a: (a["expected_points"], a["objective"]))
    eligible = [a for a in alternatives if not _paid_transfers(a["decisions"][0], config)
                or a["expected_points"] - free["expected_points"] >= config.transfer_gain_threshold
                + _paid_transfers(a["decisions"][0], config) * config.paid_transfer_gain_buffer]
    best = max(eligible, key=lambda a: a["objective"])
    if best["action"] != "hold" and best["objective"] - reference["objective"] < config.transfer_gain_threshold:
        best = reference
    winning_state, winning_decision = search.transition(state, best["root_action"], gameweek)
    return {"decision": winning_decision, "state": winning_state,
            "alternatives": alternatives, "terminal_transfer_value": best["terminal_transfer_value"]}


def _prepare(predictions_by_gw, gameweeks, config, initial_state):
    validate_config(config)
    data = normalise_forecasts(predictions_by_gw)
    weeks = sorted(int(gw) for gw in (gameweeks if gameweeks is not None else data))
    if not weeks or weeks != list(range(weeks[0], weeks[-1] + 1)):
        raise ValueError("V2 requires consecutive gameweeks")
    if any(gw not in data for gw in weeks):
        raise ValueError("Missing V2 prediction gameweeks")
    if any(gw < 1 or gw > 38 for gw in weeks):
        raise ValueError("Gameweeks must be between 1 and 38")
    season = legacy._prediction_map_season_name(data)
    if initial_state is not None:
        state = legacy.ManagerState.from_dict(initial_state.to_dict())
        if state.last_processed_gameweek + 1 != weeks[0]:
            raise ValueError("V2 must start immediately after the saved manager state")
        if state.season_name and season and state.season_name != season:
            raise ValueError("V2 state and prediction season differ")
        # Real club transfers may grandfather an owned squad over the club cap.
        # Holding and choosing a lineup remain legal; every new purchase package
        # is still solved with the usual three-per-club constraints.
        owned = legacy._squad_frame_for_gw(state.squad_player_ids, data[weeks[0]])
        expected_counts = {code: legacy.SQUAD_POSITION_LIMITS[label]
                           for code, label in ((1, "GK"), (2, "DEF"), (3, "MID"), (4, "FWD"))}
        if (len(state.squad_player_ids) != 15 or len(set(state.squad_player_ids)) != 15
                or owned.element_type.value_counts().to_dict() != expected_counts):
            raise ValueError("Invalid V2 starting squad")
    else:
        opening = legacy._available_horizon(data, weeks[0], config.initial_horizon)
        aggregate = legacy._aggregate_projection_for_gameweek(data, opening, weeks[0])
        search = _Search(data, config)
        initial_candidates = [legacy._optimise_team(aggregate, config.rules.budget_m, config)]
        for gw in opening:
            initial_candidates.append(legacy._optimise_team(data[gw], config.rules.budget_m, config))
        team = max(initial_candidates, key=lambda candidate: sum(
            search.lineup(legacy._records_to_squad_ids(candidate), gw)["total_expected_points_with_captain"]
            for gw in opening))
        ids = legacy._records_to_squad_ids(team)
        prices = legacy._current_price_lookup(data[weeks[0]])
        state = legacy.ManagerState(ids, round(config.rules.budget_m - sum(prices[p] for p in ids), 2),
                                    0, {p: prices[p] for p in ids},
                                    last_processed_gameweek=weeks[0] - 1, season_name=season)
    state.season_name = state.season_name or season
    return data, weeks, state


def _policy(initial_ids, decisions, state):
    return _json_safe({"manager_principle": legacy.MANAGER_PRINCIPLE, "manager_version": "v2",
            "initial_squad": list(initial_ids), "decisions": decisions,
            "used_chips": deepcopy(state.used_chips), "final_state": state.to_dict(),
            "summary": {"gameweeks": len(decisions),
                        "start_gameweek": decisions[0]["gameweek"],
                        "end_gameweek": decisions[-1]["gameweek"],
                        "total_expected_points": sum(d["expected_points"] for d in decisions),
                        "transfers_made": sum(len(d["transfers"]) for d in decisions),
                        "chips_used": [d["chip"] for d in decisions if d["chip"]]}})


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def simulate_v2_season(predictions_by_gw, *, gameweeks=None, config=None, initial_state=None, progress_callback=None):
    config = config or legacy.SeasonManagerConfig(manager_version="v2", max_transfers_per_gw=5)
    data, weeks, state = _prepare(predictions_by_gw, gameweeks, config, initial_state)
    initial_ids, decisions = list(state.squad_player_ids), []
    search = _Search(data, config)
    for index, gw in enumerate(weeks):
        result = plan_gameweek(data, state, gw, config,
                               allow_transfers=initial_state is not None or index > 0, search=search)
        history = deepcopy(state.history)
        state, decision = result["state"], result["decision"]
        decision["decision_comparison"] = [{k: a[k] for k in ("action", "expected_points", "objective", "terminal_transfer_value")}
                                            for a in result["alternatives"]]
        state.history = history + [deepcopy(decision)]
        decision["manager_state_after"] = state.to_dict()
        decisions.append(decision)
        if progress_callback:
            progress_callback({"event": "gameweek_complete", "gameweek": gw,
                               "current_gameweek": gw, "gameweek_index": index + 1,
                               "gameweeks_total": len(weeks), "phase": "optimizing_policy",
                               "progress": (index + 1) / len(weeks)})
    return _policy(initial_ids, decisions, state)


def run_v2_simulations(predictions_by_gw, *, gameweeks=None, simulations=128, config=None,
                       random_seed=None, initial_state=None, progress_callback=None):
    """Shared selection scenarios, fresh validation, then replan next deadline.

    Fixed candidate continuations cannot observe future random outcomes. They
    are plans under today's information, not clairvoyant adaptive policies.
    """
    config = config or legacy.SeasonManagerConfig(manager_version="v2", max_transfers_per_gw=5)
    data, weeks, state = _prepare(predictions_by_gw, gameweeks, config, initial_state)
    if not 1 <= simulations <= 10000:
        raise ValueError("V2 simulations must be between 1 and 10000")
    search = _Search(data, config)
    result = plan_gameweek(data, state, weeks[0], config,
                           allow_transfers=initial_state is not None, search=search)
    # Score only the actual search horizon. Further weeks are freshly replanned.
    alternatives = result["alternatives"]
    alternatives.sort(key=lambda a: (a["action"] != "hold", -a["objective"]))
    horizon = [d["gameweek"] for d in alternatives[0]["decisions"]]
    scenario_data = {gw: data[gw] for gw in horizon}
    seed_sequence = np.random.SeedSequence(random_seed)
    selection_seed, validation_seed = [int(s.generate_state(1)[0]) for s in seed_sequence.spawn(2)]
    scenario_options = dict(noise_scale=config.monte_carlo_noise_scale,
                            injury_probability=config.scenario_injury_probability,
                            team_correlation=config.scenario_team_correlation)
    if progress_callback:
        progress_callback({"phase": "simulating", "progress": .1,
                           "completed_simulations": 0, "total_simulations": simulations})
    def scenario_progress(fraction, phase):
        if progress_callback:
            progress_callback({"phase": phase, "progress": .1 + .4 * fraction + (.4 if phase == "validating_plans" else 0),
                               "completed_simulations": int(simulations * fraction), "total_simulations": simulations})
    selection = compare_shared_futures(alternatives, scenario_data, count=simulations,
                                        seed=selection_seed, **scenario_options,
                                        progress_callback=lambda f: scenario_progress(f, "comparing_plans"))
    finalists = select_finalists(alternatives, selection, config)
    validation = compare_shared_futures([alternatives[i] for i in finalists], scenario_data,
                                        count=simulations, seed=validation_seed, **scenario_options,
                                        progress_callback=lambda f: scenario_progress(f, "validating_plans"))
    assessed = assess_paid_transfers([alternatives[i] for i in finalists], validation, config)
    winner = assessed["winner"]
    selected = alternatives[finalists[winner]]
    # Materialize a normal result, including exact first-decision state, without
    # writing or committing it. Replanning beyond the horizon uses the same engine.
    decisions = [deepcopy(d) for d in selected["decisions"] if d["gameweek"] in weeks]
    committed = legacy.ManagerState.from_dict(state.to_dict())
    for decision in decisions:
        action = decision["planned_action"]
        history = deepcopy(committed.history)
        committed, materialized = search.transition(committed, action, decision["gameweek"])
        decision.update(materialized)
        committed.history = history + [deepcopy(decision)]
        decision["manager_state_after"] = committed.to_dict()
    comparison = [{"action": alternatives[i]["action"],
                   **{k: v for k, v in metrics.items() if k != "scores"},
                   **assessed["diagnostics"][j],
                   "terminal_transfer_value": alternatives[i]["terminal_transfer_value"]}
                  for j, (i, metrics) in enumerate(zip(finalists, validation))]
    decisions[0]["decision_comparison"] = comparison
    decisions[0]["paid_transfer_comparison"] = {
        "free_reference": assessed["diagnostics"][winner]["free_reference_action"],
        "selected": assessed["diagnostics"][winner],
        "gain_buffer_per_paid_transfer": config.paid_transfer_gain_buffer,
        "minimum_win_probability": config.paid_transfer_min_probability,
    }
    decisions[0]["manager_state_after"]["history"][-1]["decision_comparison"] = comparison
    decisions[0]["manager_state_after"]["history"][-1]["paid_transfer_comparison"] = decisions[0]["paid_transfer_comparison"]
    remaining = [gw for gw in weeks if gw > horizon[-1]]
    if remaining:
        tail = simulate_v2_season(data, gameweeks=remaining, config=config, initial_state=committed,
                                  progress_callback=progress_callback)
        decisions.extend(tail["decisions"])
        committed = legacy.ManagerState.from_dict(tail["final_state"])
    policy = _policy(state.squad_player_ids, decisions, committed)
    policy["policy_evaluation"] = {k: v for k, v in validation[winner].items() if k != "scores"}
    # Evaluation totals cover the search horizon, not an extrapolated full season.
    scores = validation[winner]["scores"]
    runs = [{"simulation_id": i + 1, "summary": {"total_expected_points": score,
             "transfers_made": sum(len(d["transfers"]) for d in selected["decisions"])}} for i, score in enumerate(scores)]
    if progress_callback:
        progress_callback({"phase": "complete", "progress": 1.,
                           "completed_simulations": simulations, "total_simulations": simulations})
    return {"manager_principle": legacy.MANAGER_PRINCIPLE, "manager_version": "v2",
            "simulation_mode": "shared_scenarios", "recommended_policy": policy,
            "best_run": policy, "runs": runs, "policy_runs": [policy],
            "planning_continuation": _json_safe(selected["decisions"]),
            "evaluation": {"selection_seed": selection_seed, "validation_seed": validation_seed,
                           "selection_simulations": simulations, "validation_simulations": simulations,
                           "gameweeks": horizon, "comparison": comparison},
            "summary": {"simulations": simulations, "simulation_mode": "shared_scenarios",
                        "evaluation_gameweeks": horizon,
                        "average_total_expected_points": validation[winner]["average_total_points"],
                        "median_total_expected_points": validation[winner]["median_total_points"],
                        "min_total_expected_points": float(min(scores)),
                        "max_total_expected_points": float(max(scores)),
                        **{k: v for k, v in validation[winner].items() if k != "scores"}}}
