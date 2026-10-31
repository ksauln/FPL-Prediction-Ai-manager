from dataclasses import replace
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from fplmodel.season_manager import ManagerState, SeasonManagerConfig, simulate_season
from fplmodel.manager_v2 import choose_captains, plan_gameweek, run_v2_simulations
from fplmodel.manager_scenarios import sample_scenarios, compare_plans


def frames(gameweeks=(2, 3, 4)):
    positions = [1, 1] + [2] * 5 + [3] * 5 + [4] * 3 + [2, 3, 3, 4, 1]
    return {gw: pd.DataFrame([
        dict(player_id=i, full_name=f"Player {i}", team_name=f"Team {i % 8}",
             team_id=i % 8, element_type=pos, now_cost_millions=5.0,
             expected_points=4.0 if i <= 15 else 1.0,
             start_probability=1.0, appearance_probability=1.0,
             expected_points_lower_80=1.0, expected_points_upper_80=7.0,
             season_name="2026-27", gameweek=gw)
        for i, pos in enumerate(positions, 1)
    ]) for gw in gameweeks}


def state(free=1):
    return ManagerState(list(range(1, 16)), 0.0, free,
                        {i: 5.0 for i in range(1, 16)},
                        last_processed_gameweek=1, season_name="2026-27")


def config(**kwargs):
    return SeasonManagerConfig(manager_version="v2", enable_chips=False,
                               planning_beam_width=2, planning_candidates=4,
                               max_transfers_per_gw=5, **kwargs)


def test_legacy_is_default_and_v2_does_not_mutate_inputs():
    assert SeasonManagerConfig().manager_version == "legacy"
    data, initial = frames((2,)), state()
    saved = initial.to_dict()
    before = data[2].copy(deep=True)
    result = simulate_season(data, config=config(), initial_state=initial)
    assert result["manager_version"] == "v2"
    assert initial.to_dict() == saved
    pd.testing.assert_frame_equal(data[2], before)
    assert result["decisions"][0]["manager_state_after"]["last_processed_gameweek"] == 2


def test_captain_pair_uses_fallback_without_penalizing_expected_points_twice():
    players = pd.DataFrame([
        dict(player_id=1, full_name="A", expected_points=8., appearance_probability=.5),
        dict(player_id=2, full_name="B", expected_points=7., appearance_probability=1.),
        dict(player_id=3, full_name="C", expected_points=6., appearance_probability=1.),
    ])
    captain, vice, bonus = choose_captains(players)
    assert (captain, vice) == (1, 2)
    assert bonus == pytest.approx(11.5)


def test_weekly_lineups_use_each_weeks_projections():
    data = frames((2, 3))
    data[2].loc[data[2].player_id == 7, "expected_points"] = 0.
    data[3].loc[data[3].player_id == 7, "expected_points"] = 15.
    cfg = replace(config(), max_transfers_per_gw=0)
    result = simulate_season(data, config=cfg, initial_state=state())
    assert 7 not in result["decisions"][0]["starting_player_ids"]
    assert 7 in result["decisions"][1]["starting_player_ids"]
    assert result["decisions"][1]["captain_id"] == 7


def test_roll_is_a_real_action_and_remaining_transfers_have_value():
    data = frames((2,))
    plan = plan_gameweek(data, state(), 2, config())
    assert plan["decision"]["transfers"] == []
    assert plan["decision"]["free_transfers_after"] == 2
    assert any(c["action"] == "hold" for c in plan["alternatives"])
    assert plan["terminal_transfer_value"] > 0


def test_larger_free_transfer_packages_are_considered():
    data = frames((2,))
    for pid in (16, 17, 19):
        data[2].loc[data[2].player_id == pid, "expected_points"] = 20.
    plan = plan_gameweek(data, state(free=3), 2, config())
    assert len(plan["decision"]["transfers"]) >= 3
    assert plan["decision"]["transfer_hit_cost"] == 0
    assert plan["state"].bank_m >= 0


def test_shared_scenarios_are_order_and_candidate_independent():
    data = frames((2, 3))
    a = sample_scenarios(data, count=40, seed=12)
    b = sample_scenarios({gw: df.iloc[::-1] for gw, df in reversed(list(data.items()))},
                         count=40, seed=12)
    assert a == b
    policy = simulate_season(data, config=replace(config(), max_transfers_per_gw=0),
                             initial_state=state())
    results = compare_plans([policy, deepcopy(policy)], a)
    assert results[0]["scores"] == results[1]["scores"]
    assert results[1]["probability_beats_hold"] == 0


def test_scenarios_have_team_correlation_persistent_injuries_and_blanks():
    data = frames((2, 3, 4))
    for df in data.values():
        df["appearance_probability"] = 1.
        df["start_probability"] = 1.
        df.loc[df.player_id == 1, "fixture_multiplier"] = 0.
    draws = sample_scenarios(data, count=1500, seed=42, injury_probability=.3)
    assert all(not d[2][1][0] and d[2][1][1] == 0 for d in draws)
    first_missing = [d for d in draws if not d[2][2][0]]
    assert np.mean([not d[3][2][0] for d in first_missing]) > .7
    # Players 3 and 11 share a team; 3 and 12 do not.
    a = [d[2][3][1] for d in draws]
    same = [d[2][11][1] for d in draws]
    other = [d[2][12][1] for d in draws]
    assert np.corrcoef(a, same)[0, 1] > np.corrcoef(a, other)[0, 1]


def test_repeated_v2_uses_fresh_validation_and_keeps_live_state_unchanged():
    data, initial = frames((2,)), state()
    saved = initial.to_dict()
    result = run_v2_simulations(data, config=config(), initial_state=initial,
                                simulations=20, random_seed=12)
    assert result["evaluation"]["selection_seed"] != result["evaluation"]["validation_seed"]
    assert result["evaluation"]["validation_simulations"] == 20
    assert result["recommended_policy"]["decisions"][0]["decision_comparison"]
    assert initial.to_dict() == saved


def test_invalid_v2_settings_and_gaps_fail_closed():
    with pytest.raises(ValueError, match="beam"):
        plan_gameweek(frames(), state(), 2, replace(config(), planning_beam_width=0))
    with pytest.raises(ValueError, match="consecutive"):
        simulate_season(frames((2, 4)), config=config(), initial_state=state())


def test_roll_now_unlocks_a_funded_two_player_move_next_week_without_a_hit():
    data = frames((2, 3, 4))
    for gw, df in data.items():
        df.loc[df.player_id == 17, "now_cost_millions"] = 7.
        df.loc[df.player_id == 18, "now_cost_millions"] = 3.
        df.loc[df.player_id == 17, "expected_points"] = 1. if gw == 2 else 20.
        df.loc[df.player_id == 18, "expected_points"] = 0. if gw == 2 else 4.
    cfg = replace(config(), max_transfers_per_gw=2, transfer_horizon=3,
                  formations=({"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},))
    result = simulate_season(data, config=cfg, initial_state=state())
    first, second = result["decisions"][:2]
    assert first["transfers"] == []
    assert len(second["transfers"]) == 2
    assert {t["in_player"]["player_id"] for t in second["transfers"]} == {17, 18}
    assert second["transfer_hit_cost"] == 0.


def test_joint_planner_saves_triple_captain_for_the_stronger_week():
    from fplmodel.season_manager import SeasonRules
    data = frames((2, 3))
    data[2].loc[data[2].player_id == 8, "expected_points"] = 8.
    data[3].loc[data[3].player_id == 8, "expected_points"] = 20.
    rules = SeasonRules(chips_by_half={"first": {"triple_captain": 1}, "second": {}})
    cfg = replace(config(), enable_chips=True, rules=rules, max_transfers_per_gw=0)
    result = simulate_season(data, config=cfg, initial_state=state())
    assert [d["chip"] for d in result["decisions"]] == [None, "triple_captain"]


def test_expiring_chip_is_used_for_a_small_positive_gain():
    from fplmodel.season_manager import SeasonRules
    data = frames((19,))
    initial = state()
    initial.last_processed_gameweek = 18
    rules = SeasonRules(chips_by_half={"first": {"triple_captain": 1}, "second": {}})
    cfg = replace(config(), enable_chips=True, rules=rules, max_transfers_per_gw=0)
    result = simulate_season(data, config=cfg, initial_state=initial)
    assert result["decisions"][0]["chip"] == "triple_captain"


def test_availability_is_applied_to_probabilities_once_and_blank_rows_are_zero():
    from fplmodel.manager_scenarios import normalise_forecasts
    data = frames((2,))
    data[2]["availability_next_round"] = .5
    normalized = normalise_forecasts(data)
    assert normalized[2].appearance_probability.iloc[0] == .5
    twice = normalise_forecasts(normalized)
    pd.testing.assert_frame_equal(normalized[2], twice[2])
    data[2]["fixture_multiplier"] = 0.
    blank = normalise_forecasts(data)[2]
    assert blank.expected_points.eq(0).all()
    assert blank.appearance_probability.eq(0).all()


def test_scenario_ranking_is_reproducible_and_compares_identical_futures():
    data = frames((2,))
    cfg = replace(config(), max_transfers_per_gw=0)
    a = run_v2_simulations(data, config=cfg, initial_state=state(), simulations=20, random_seed=12)
    b = run_v2_simulations(data, config=cfg, initial_state=state(), simulations=20, random_seed=12)
    assert a == b


def test_free_hit_returns_to_owned_squad_and_preserves_transfers():
    from fplmodel.season_manager import SeasonRules
    data = frames((2,))
    data[2].loc[data[2].player_id > 15, "expected_points"] = 20.
    rules = SeasonRules(chips_by_half={"first": {"free_hit": 1}, "second": {}})
    cfg = replace(config(), enable_chips=True, rules=rules, max_transfers_per_gw=0)
    initial = state(free=3)
    result = simulate_season(data, config=cfg, initial_state=initial)
    decision = result["decisions"][0]
    assert decision["chip"] == "free_hit"
    assert set(decision["squad_player_ids"]) == set(initial.squad_player_ids)
    assert decision["free_transfers_after"] == 3
    assert set(decision["starting_player_ids"]) - set(initial.squad_player_ids)


def test_free_hit_cannot_repeat_across_chip_half_boundary():
    from fplmodel.season_manager import SeasonRules
    data = frames((19, 20))
    for df in data.values():
        df.loc[df.player_id > 15, "expected_points"] = 20.
    initial = state()
    initial.last_processed_gameweek = 18
    rules = SeasonRules(chips_by_half={"first": {"free_hit": 1}, "second": {"free_hit": 1}})
    cfg = replace(config(), enable_chips=True, rules=rules, max_transfers_per_gw=0)
    plan = plan_gameweek(data, initial, 19, cfg)
    for alternative in plan["alternatives"]:
        assert [d["chip"] for d in alternative["decisions"]] != ["free_hit", "free_hit"]


def test_joint_plan_prepares_bench_boost_while_saving_two_transfers_for_next_week():
    from fplmodel.season_manager import SeasonRules
    data = frames((2, 3))
    for gw, df in data.items():
        df.loc[df.player_id <= 15, "expected_points"] = 10.
        df.loc[df.player_id.isin([2, 6, 7, 12]), "expected_points"] = 1.
        df.loc[df.player_id == 20, "now_cost_millions"] = 6.
        df.loc[df.player_id == 17, "now_cost_millions"] = 7.
        df.loc[df.player_id == 18, "now_cost_millions"] = 3.
        if gw == 3:
            for pid, points in ((17, 30.), (18, 10.), (20, 15.)):
                df.loc[df.player_id == pid, "expected_points"] = points
    initial = state(free=2)
    initial.bank_m = 1.
    rules = SeasonRules(chips_by_half={"first": {"bench_boost": 1}, "second": {}})
    cfg = replace(config(), enable_chips=True, rules=rules, max_transfers_per_gw=2,
                  formations=({"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},))
    result = simulate_season(data, config=cfg, initial_state=initial)
    first, second = result["decisions"]
    assert {t["in_player"]["player_id"] for t in first["transfers"]} == {20}
    assert first["chip"] is None
    assert {t["in_player"]["player_id"] for t in second["transfers"]} == {17, 18}
    assert second["chip"] == "bench_boost"
    assert second["transfer_hit_cost"] == 0.


def test_requested_gameweeks_limit_returned_state_but_keep_planning_lookahead():
    data = frames((2, 3, 4))
    result = run_v2_simulations(data, gameweeks=[2], config=replace(config(), max_transfers_per_gw=0),
                                initial_state=state(), simulations=4, random_seed=1)
    assert [d["gameweek"] for d in result["recommended_policy"]["decisions"]] == [2]
    assert result["recommended_policy"]["final_state"]["last_processed_gameweek"] == 2
    assert result["evaluation"]["gameweeks"] == [2, 3, 4]


def test_owned_squad_can_hold_four_from_a_club_after_a_real_transfer():
    data = frames((2,))
    data[2].loc[data[2].player_id.isin([1, 2, 3, 4]), "team_id"] = 99
    result = simulate_season(data, config=replace(config(), max_transfers_per_gw=0), initial_state=state())
    assert set(result["final_state"]["squad_player_ids"]) == set(state().squad_player_ids)
    assert result["decisions"][0]["transfers"] == []
    data[2].loc[data[2].player_id == 16, "expected_points"] = 25.
    changed = simulate_season(data, config=replace(config(), max_transfers_per_gw=1), initial_state=state())
    held = data[2][data[2].player_id.isin(changed["final_state"]["squad_player_ids"])]
    assert held.groupby("team_id").size().max() <= 3


def test_invalid_owned_position_counts_are_still_rejected():
    data = frames((2,))
    data[2].loc[data[2].player_id == 2, "element_type"] = 3
    with pytest.raises(ValueError, match="Invalid V2 starting squad"):
        simulate_season(data, config=config(), initial_state=state())
