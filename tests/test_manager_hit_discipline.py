from dataclasses import replace

import numpy as np
import pytest

from fplmodel.manager_v2 import assess_paid_transfers, select_finalists, run_v2_simulations, plan_gameweek
from test_manager_v2 import frames, state, config


def candidate(action, hits=0., terminal=1., future_buffer=0.):
    return {"action": action, "decisions": [{"transfer_hit_cost": hits}],
            "terminal_transfer_value": terminal, "risk_adjustment": future_buffer}


def metrics(scores):
    return {"scores": scores, "average_total_points": float(np.mean(scores))}


def test_paid_package_must_beat_best_free_move_not_only_holding():
    plans = [candidate("hold"), candidate("free"), candidate("paid", 16.)]
    scores = [metrics([100.] * 4), metrics([109.] * 4), metrics([110.] * 4)]
    assessed = assess_paid_transfers(plans, scores, config())
    assert assessed["winner"] == 1
    assert assessed["free_reference"] == 1
    assert assessed["diagnostics"][2]["gain_over_free"] == 1.
    assert assessed["diagnostics"][2]["required_paid_gain"] == 9.5
    assert not assessed["diagnostics"][2]["paid_transfer_eligible"]


def test_hit_with_clear_marginal_gain_is_still_allowed_and_hits_arent_charged_twice():
    plans = [candidate("hold"), candidate("paid", 4.)]
    # Scores already include the four-point hit: marginal net gain is four.
    assessed = assess_paid_transfers(plans, [metrics([100.] * 4), metrics([104.] * 4)], config())
    assert assessed["winner"] == 1
    assert assessed["diagnostics"][1]["paid_transfer_eligible"]


def test_uncertain_paid_advantage_is_rejected_despite_a_high_mean():
    plans = [candidate("hold"), candidate("paid", 4.)]
    assessed = assess_paid_transfers(plans,
        [metrics([100.] * 4), metrics([95., 95., 140., 140.])], config())
    assert assessed["winner"] == 0
    assert assessed["diagnostics"][1]["probability_beats_free"] == .5


def test_terminal_saved_transfer_value_cannot_justify_a_paid_move():
    plans = [candidate("hold"), candidate("paid", 4., terminal=100.)]
    assessed = assess_paid_transfers(plans, [metrics([100.] * 4), metrics([100.2] * 4)], config())
    assert assessed["winner"] == 0


def test_free_comparator_is_chosen_by_net_points_without_terminal_value():
    plans = [candidate("hold", terminal=10.), candidate("best_free", terminal=0.),
             candidate("paid", 4., terminal=20.)]
    assessed = assess_paid_transfers(plans,
        [metrics([100.] * 4), metrics([105.] * 4), metrics([107.] * 4)], config())
    assert assessed["free_reference"] == 1
    assert assessed["winner"] != 2
    assert assessed["diagnostics"][2]["gain_over_free"] == 2.


def test_free_packages_are_not_subject_to_paid_transfer_gate():
    plans = [candidate("hold"), candidate("three_free_transfers")]
    assessed = assess_paid_transfers(plans, [metrics([100.] * 4), metrics([106.] * 4)], config())
    assert assessed["winner"] == 1
    assert assessed["diagnostics"][1]["paid_transfers_now"] == 0


def test_free_alternative_cannot_be_dropped_from_fresh_validation():
    plans = [candidate("hold"), candidate("paid1", 4.), candidate("paid2", 8.),
             candidate("paid3", 12.), candidate("best_free")]
    scores = [metrics([100.]), metrics([120.]), metrics([119.]), metrics([118.]), metrics([110.])]
    indices = select_finalists(plans, scores, config())
    assert 0 in indices and 4 in indices


def test_deterministic_and_simulated_planners_prefer_nearly_as_good_free_move():
    data = frames((2,))
    data[2].loc[data[2].player_id == 17, "expected_points"] = 11.
    data[2].loc[data[2].player_id == 19, "expected_points"] = 10.
    cfg = replace(config(), monte_carlo_noise_scale=0., scenario_injury_probability=0.)
    deterministic = plan_gameweek(data, state(free=1), 2, cfg)["decision"]
    simulated = run_v2_simulations(data, initial_state=state(free=1), config=cfg,
                                    simulations=8, random_seed=90)["recommended_policy"]["decisions"][0]
    for decision in (deterministic, simulated):
        assert len(decision["transfers"]) == 1
        assert decision["transfer_hit_cost"] == 0.
    assert simulated["paid_transfer_comparison"]


def test_search_shortlist_keeps_a_free_move_when_paid_packages_rank_higher():
    from fplmodel.manager_v2 import _Search
    from fplmodel.manager_scenarios import normalise_forecasts
    data = frames((2,))
    data[2].loc[data[2].player_id == 17, "expected_points"] = 11.
    data[2].loc[data[2].player_id == 19, "expected_points"] = 11.5
    cfg = replace(config(), planning_candidates=2)
    actions = _Search(normalise_forecasts(data), cfg).candidates(state(free=1), 2, [2])
    assert any(a["action"] != "hold" and len(set(a["target"]) - set(state().squad_player_ids)) == 1 for a in actions)


def test_full_planner_still_takes_a_clearly_worthwhile_hit():
    data = frames((2,))
    data[2].loc[data[2].player_id.isin([17, 19]), "expected_points"] = 20.
    cfg = replace(config(), monte_carlo_noise_scale=0., scenario_injury_probability=0.)
    result = run_v2_simulations(data, initial_state=state(free=1), config=cfg,
                                simulations=8, random_seed=90)
    selected = result["recommended_policy"]["decisions"][0]
    assert len(selected["transfers"]) == 2
    assert selected["transfer_hit_cost"] == 4.
    assert selected["paid_transfer_comparison"]["selected"]["paid_transfer_eligible"]


@pytest.mark.parametrize("field,value", [
    ("paid_transfer_gain_buffer", -1.), ("paid_transfer_gain_buffer", float("nan")),
    ("paid_transfer_min_probability", 1.1), ("paid_transfer_min_probability", -.1),
])
def test_invalid_hit_controls_are_rejected(field, value):
    with pytest.raises(ValueError):
        plan_gameweek(frames((2,)), state(), 2, replace(config(), **{field: value}))
