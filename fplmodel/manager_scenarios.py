"""Shared future outcomes for fair comparisons of manager policies.

These are explicit scenario assumptions, not calibrated probability guarantees.
Generate outcomes once for the entire player universe, independently of plans.
"""
from __future__ import annotations

import json
from math import sqrt

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .season_manager import _normalise_prediction_map, _score_decision_outcomes


def normalise_forecasts(predictions_by_gw):
    data = _normalise_prediction_map(predictions_by_gw)
    for frame in data.values():
        if frame.player_id.duplicated().any():
            raise ValueError("Duplicate V2 player IDs")
        included = frame.get("probabilities_include_availability", pd.Series(False, index=frame.index)).fillna(False)
        if included.dtype == object:
            included = included.astype(str).str.lower().eq("true")
        gate = pd.to_numeric(frame.get("availability_next_round", pd.Series(np.nan, index=frame.index)), errors="coerce")
        for name in ("availability_this_round", "status_availability"):
            gate = gate.fillna(pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)), errors="coerce"))
        gate = gate.fillna(1.).clip(0, 1)
        readiness = pd.to_numeric(frame.get("readiness_multiplier", pd.Series(1., index=frame.index)), errors="coerce").fillna(1.).clip(0, 1)
        gate = (gate * readiness).where(~included.astype(bool), 1.)
        frame["start_probability"] = (frame.start_probability * gate).clip(0, 1)
        frame["appearance_probability"] = np.maximum(frame.appearance_probability * gate, frame.start_probability).clip(0, 1)
        frame["probabilities_include_availability"] = True
        blank = frame.fixture_multiplier.eq(0)
        frame.loc[blank, ["expected_points", "start_probability", "appearance_probability",
                          "expected_points_lower_80", "expected_points_upper_80"]] = 0.
        if not np.isfinite(frame[["expected_points", "now_cost_millions", "start_probability", "appearance_probability"]].to_numpy()).all():
            raise ValueError("V2 forecasts must contain finite costs, points and probabilities")
    return data


def _fixtures(player):
    value = player.get("fixture_forecasts")
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, list):
        return value
    if float(player.get("fixture_multiplier", 1.0)) == 0:
        return []
    return [player]


def sample_scenarios(predictions_by_gw, *, count, seed, noise_scale=1.,
                     injury_probability=.015, team_correlation=.35):
    """Sample injury paths, correlated rotation, and team/fixture point shocks.

    EP is unconditional on appearance. Conditional means recover that EP before
    additional future injuries; availability is not multiplied into EP again.
    Injury spells last 2–4 GWs. Shared normals provide positive same-team
    dependence while opponent attack shocks can reduce defensive outcomes.
    """
    if not 1 <= count <= 10000:
        raise ValueError("V2 scenarios must be between 1 and 10000")
    if not 0 <= injury_probability <= 1 or not 0 <= team_correlation < 1:
        raise ValueError("Scenario injury probability/correlation is invalid")
    if not np.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("Scenario noise_scale must be nonnegative and finite")
    data = normalise_forecasts(predictions_by_gw)
    rng = np.random.default_rng(seed)
    draws = [{} for _ in range(count)]
    injuries = {}
    for gw, frame in sorted(data.items()):
        teams = sorted(set(frame.team_id.astype(int)))
        attack = {team: rng.normal(size=count) for team in teams}
        defence = {team: rng.normal(size=count) for team in teams}
        rotation = {team: rng.normal(size=count) for team in teams}
        for draw in draws:
            draw[gw] = {}
        for player in frame.sort_values("player_id").to_dict("records"):
            pid, team = int(player["player_id"]), int(player["team_id"])
            recovery = injuries.setdefault(pid, np.zeros(count, dtype=int))
            starts = (rng.random(count) < injury_probability) & (recovery < gw)
            recovery[starts] = gw + rng.integers(1, 4, size=int(starts.sum()))
            injured = recovery >= gw
            total = np.zeros(count)
            appeared = np.zeros(count, dtype=bool)
            for fixture in _fixtures(player):
                mean = float(fixture.get("expected_points", player["expected_points"]))
                p_start = float(fixture.get("start_probability", player["start_probability"]))
                p_app = float(fixture.get("appearance_probability", player["appearance_probability"]))
                p_app = float(np.clip(max(p_start, p_app), 0, 1))
                p_start = float(np.clip(p_start, 0, p_app))
                roll = ndtr(sqrt(.2) * rotation[team] + sqrt(.8) * rng.normal(size=count))
                playing = (roll < p_app) & ~injured
                starting = roll < p_start
                cameo = float(fixture.get("cameo_points", player.get("cameo_points", 1.)))
                cameo_probability = max(0., p_app - p_start)
                if cameo_probability > 0:
                    # Bias-corrected EP can be below the default cameo mean.
                    # Preserve the unconditional mean without inventing a very
                    # negative conditional starter score for a rare starter.
                    cameo = min(cameo, max(0., mean) / cameo_probability)
                if p_start == 0 and p_app > 0:
                    cameo = mean / p_app
                starter_mean = (mean - cameo_probability * cameo) / max(p_start, 1e-9)
                conditional = np.where(starting, starter_mean, cameo)
                lower = float(fixture.get("expected_points_lower_80", mean - 2.))
                upper = float(fixture.get("expected_points_upper_80", mean + 2.))
                sigma = max(.05, (upper - lower) / (2 * 1.2815515655)) * noise_scale
                shared = defence[team] if int(player["element_type"]) <= 2 else attack[team]
                opponent = fixture.get("opponent_id")
                if int(player["element_type"]) <= 2 and opponent in attack:
                    shared = (shared - .35 * attack[opponent]) / sqrt(1 + .35 ** 2)
                shock = sqrt(team_correlation) * shared + sqrt(1 - team_correlation) * rng.normal(size=count)
                # Scale conditional variance once; retain signed football outcomes.
                points = conditional + sigma * shock / sqrt(max(p_app, .05))
                total += np.where(playing, points, 0.)
                appeared |= playing
            for index, draw in enumerate(draws):
                draw[gw][pid] = (bool(appeared[index]), float(total[index]))
    return draws


def compare_plans(plans, scenarios):
    """Paired comparisons; the first plan is the hold/reference continuation."""
    if not plans or not scenarios:
        raise ValueError("Plans and scenarios cannot be empty")
    scores_by_plan = [[sum(_score_decision_outcomes(d, draw[int(d["gameweek"])])
                          for d in plan["decisions"]) for draw in scenarios] for plan in plans]
    return _summarize_scores(scores_by_plan)


def _summarize_scores(scores_by_plan):
    results, reference = [], None
    for scores in scores_by_plan:
        values = np.asarray(scores)
        if reference is None:
            reference = values
        difference = values - reference
        results.append({
            "scores": scores, "average_total_points": float(values.mean()),
            "median_total_points": float(np.median(values)),
            "p10_total_points": float(np.percentile(values, 10)),
            "p90_total_points": float(np.percentile(values, 90)),
            "gain_over_hold": float(difference.mean()),
            "probability_beats_hold": float(np.mean(difference > 0)),
            "paired_standard_error": float(difference.std(ddof=1) / sqrt(len(values))) if len(values) > 1 else 0.,
            "simulations": len(scores),
        })
    return results


def compare_shared_futures(plans, data, *, count, seed, progress_callback=None, **options):
    """Bound memory while keeping every candidate on identical future paths."""
    if not 1 <= count <= 10000:
        raise ValueError("V2 simulations must be between 1 and 10000")
    all_scores = [[] for _ in plans]
    batches = (count + 63) // 64
    seeds = np.random.SeedSequence(seed).spawn(batches)
    completed = 0
    for child in seeds:
        size = min(64, count - completed)
        scenarios = sample_scenarios(data, count=size, seed=int(child.generate_state(1)[0]), **options)
        for scores, result in zip(all_scores, compare_plans(plans, scenarios)):
            scores.extend(result["scores"])
        completed += size
        if progress_callback:
            progress_callback(completed / count)
    return _summarize_scores(all_scores)
