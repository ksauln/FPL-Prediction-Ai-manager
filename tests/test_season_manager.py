from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

try:
    import pulp  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment guard
    raise unittest.SkipTest("pulp is required for season manager optimizer tests") from exc

from fplmodel.season_manager import (
    ManagerState,
    SeasonManagerConfig,
    SeasonRules,
    _score_decision_outcomes,
    _sample_player_outcome,
    _rebase_policy_run,
    _apply_transfers,
    _order_transfers_for_execution,
    load_prediction_files,
    load_manager_state,
    run_repeated_season_simulations,
    save_manager_state,
    simulate_season,
)


def _player(
    player_id: int,
    name: str,
    team_id: int,
    element_type: int,
    cost: float,
    gw_points: dict[int, float],
    lower: float = 1.0,
    upper: float = 9.0,
) -> dict[int, dict[str, object]]:
    return {
        gw: {
            "player_id": player_id,
            "full_name": name,
            "team_name": f"Team {team_id}",
            "team_id": team_id,
            "element_type": element_type,
            "now_cost_millions": cost,
            "expected_points": points,
            "start_probability": 0.9,
            "confidence_score": 80.0,
            "confidence_level": "High",
            "expected_points_lower_80": max(0.0, points - lower),
            "expected_points_upper_80": points + upper,
        }
        for gw, points in gw_points.items()
    }


def _prediction_frames() -> dict[int, pd.DataFrame]:
    players: list[dict[int, dict[str, object]]] = [
        _player(101, "GK One", 1, 1, 4.5, {1: 4.0, 2: 4.0, 3: 4.0}),
        _player(102, "GK Two", 2, 1, 4.0, {1: 3.0, 2: 3.0, 3: 3.0}),
        _player(103, "GK Wild", 3, 1, 4.5, {1: 2.0, 2: 2.0, 3: 8.0}),
        _player(201, "Def One", 1, 2, 5.0, {1: 5.0, 2: 5.0, 3: 5.0}),
        _player(202, "Def Two", 2, 2, 5.0, {1: 4.8, 2: 4.8, 3: 4.8}),
        _player(203, "Def Three", 3, 2, 4.5, {1: 4.4, 2: 4.4, 3: 4.4}),
        _player(204, "Def Four", 4, 2, 4.5, {1: 4.0, 2: 4.0, 3: 4.0}),
        _player(205, "Def Five", 5, 2, 4.0, {1: 3.6, 2: 3.6, 3: 3.6}),
        _player(206, "Def Six", 6, 2, 4.0, {1: 3.0, 2: 3.0, 3: 3.0}),
        _player(207, "Def Free Hit", 7, 2, 4.5, {1: 2.0, 2: 2.0, 3: 8.0}),
        _player(301, "Mid One", 1, 3, 9.0, {1: 8.0, 2: 8.0, 3: 8.0}),
        _player(302, "Mid Two", 2, 3, 8.0, {1: 7.0, 2: 7.0, 3: 7.0}),
        _player(303, "Mid Three", 3, 3, 7.0, {1: 6.0, 2: 6.0, 3: 6.0}),
        _player(304, "Mid Four", 4, 3, 6.0, {1: 5.0, 2: 5.0, 3: 5.0}),
        _player(305, "Mid Five", 5, 3, 5.5, {1: 4.5, 2: 4.5, 3: 4.5}),
        _player(306, "Mid Six", 6, 3, 5.0, {1: 4.0, 2: 4.0, 3: 4.0}),
        _player(307, "Mid Breakout", 7, 3, 5.0, {1: 3.0, 2: 13.0, 3: 13.0}),
        _player(308, "Mid Free Hit", 8, 3, 5.5, {1: 2.0, 2: 2.0, 3: 12.0}),
        _player(401, "Fwd One", 1, 4, 8.0, {1: 7.0, 2: 7.0, 3: 7.0}),
        _player(402, "Fwd Two", 2, 4, 7.0, {1: 6.0, 2: 6.0, 3: 6.0}),
        _player(403, "Fwd Three", 3, 4, 6.0, {1: 5.0, 2: 5.0, 3: 5.0}),
        _player(404, "Fwd Four", 4, 4, 5.5, {1: 4.0, 2: 4.0, 3: 4.0}),
        _player(405, "Fwd Free Hit", 5, 4, 5.5, {1: 2.0, 2: 2.0, 3: 12.0}),
    ]
    frames: dict[int, pd.DataFrame] = {}
    for gw in (1, 2, 3):
        frames[gw] = pd.DataFrame([player[gw] for player in players])
    return frames


class SeasonManagerTests(unittest.TestCase):
    def test_simulate_season_tracks_stateful_decisions(self) -> None:
        config = SeasonManagerConfig(
            initial_horizon=1,
            transfer_horizon=1,
            transfer_gain_threshold=2.0,
            max_transfers_per_gw=1,
            enable_chips=False,
        )

        result = simulate_season(_prediction_frames(), gameweeks=[1, 2], config=config)

        self.assertEqual(result["summary"]["gameweeks"], 2)
        self.assertEqual(len(result["initial_squad"]), 15)
        self.assertIn("stateful manager", result["manager_principle"])

        gw1 = result["decisions"][0]
        gw2 = result["decisions"][1]
        self.assertEqual(gw1["gameweek"], 1)
        self.assertEqual(gw1["transfers"], [])
        self.assertEqual(gw2["gameweek"], 2)
        self.assertEqual(len(gw2["transfers"]), 1)
        self.assertEqual(gw2["transfers"][0]["in_player"]["full_name"], "Mid Breakout")
        self.assertGreaterEqual(gw2["free_transfers_after"], 1)
        self.assertEqual(gw2["captain"], "Mid Breakout")
        self.assertNotEqual(gw2["captain"], gw2["vice_captain"])

    def test_chip_strategy_can_use_free_hit_without_changing_squad(self) -> None:
        rules = SeasonRules(
            chips_by_half={
                "first": {"free_hit": 1, "wildcard": 0, "bench_boost": 0, "triple_captain": 0},
                "second": {"free_hit": 0, "wildcard": 0, "bench_boost": 0, "triple_captain": 0},
            }
        )
        config = SeasonManagerConfig(
            initial_horizon=2,
            transfer_horizon=1,
            chip_lookahead=1,
            max_transfers_per_gw=0,
            free_hit_gain_threshold=6.0,
            wildcard_gain_threshold=999.0,
            enable_chips=True,
            rules=rules,
        )

        result = simulate_season(_prediction_frames(), gameweeks=[1, 2, 3], config=config)

        gw3 = result["decisions"][2]
        pre_free_hit_ids = set(result["decisions"][1]["squad_player_ids"])
        post_free_hit_ids = set(gw3["squad_player_ids"])

        self.assertEqual(gw3["chip"], "free_hit")
        self.assertEqual(gw3["team_context"], "free_hit")
        self.assertEqual(gw3["financial_context"], "owned_squad")
        self.assertEqual(pre_free_hit_ids, post_free_hit_ids)
        self.assertGreater(gw3["chip_gain"], 6.0)

    def test_free_hit_cannot_be_used_in_gameweek_one(self) -> None:
        rules = SeasonRules(
            chips_by_half={
                "first": {"free_hit": 1, "wildcard": 0, "bench_boost": 0, "triple_captain": 0},
                "second": {"free_hit": 0, "wildcard": 0, "bench_boost": 0, "triple_captain": 0},
            }
        )
        config = SeasonManagerConfig(
            initial_horizon=2,
            transfer_horizon=1,
            max_transfers_per_gw=0,
            free_hit_gain_threshold=0.0,
            enable_chips=True,
            rules=rules,
        )

        result = simulate_season(_prediction_frames(), gameweeks=[1], config=config)

        self.assertIsNone(result["decisions"][0]["chip"])

    def test_wildcard_cannot_be_used_in_gameweek_one(self) -> None:
        rules = SeasonRules(
            chips_by_half={
                "first": {"free_hit": 0, "wildcard": 1, "bench_boost": 0, "triple_captain": 0},
                "second": {"free_hit": 0, "wildcard": 0, "bench_boost": 0, "triple_captain": 0},
            }
        )
        config = SeasonManagerConfig(
            initial_horizon=1,
            wildcard_gain_threshold=-999.0,
            free_hit_gain_threshold=999.0,
            enable_chips=True,
            rules=rules,
        )

        result = simulate_season(_prediction_frames(), gameweeks=[1], config=config)

        self.assertIsNone(result["decisions"][0]["chip"])

    def test_free_hit_cannot_be_used_in_consecutive_gameweeks(self) -> None:
        predictions = _prediction_frames()
        for gw in (2, 3):
            predictions[gw].loc[predictions[gw]["player_id"] == 207, "expected_points"] = 12.0
            predictions[gw].loc[predictions[gw]["player_id"] == 308, "expected_points"] = 12.0
            predictions[gw].loc[predictions[gw]["player_id"] == 405, "expected_points"] = 12.0
        rules = SeasonRules(
            chips_by_half={
                "first": {"free_hit": 2, "wildcard": 0, "bench_boost": 0, "triple_captain": 0},
                "second": {"free_hit": 0, "wildcard": 0, "bench_boost": 0, "triple_captain": 0},
            }
        )
        config = SeasonManagerConfig(
            initial_horizon=1,
            transfer_horizon=1,
            max_transfers_per_gw=0,
            free_hit_gain_threshold=0.0,
            enable_chips=True,
            rules=rules,
        )

        result = simulate_season(predictions, gameweeks=[1, 2, 3], config=config)

        self.assertEqual(result["decisions"][1]["chip"], "free_hit")
        self.assertNotEqual(result["decisions"][2]["chip"], "free_hit")

    def test_free_hit_and_wildcard_preserve_free_transfer_count(self) -> None:
        predictions = _prediction_frames()
        rules = SeasonRules(
            chips_by_half={
                "first": {"free_hit": 1, "wildcard": 1, "bench_boost": 0, "triple_captain": 0},
                "second": {"free_hit": 0, "wildcard": 0, "bench_boost": 0, "triple_captain": 0},
            }
        )
        free_hit_config = SeasonManagerConfig(
            initial_horizon=2,
            transfer_horizon=1,
            chip_lookahead=1,
            max_transfers_per_gw=0,
            free_hit_gain_threshold=6.0,
            wildcard_gain_threshold=999.0,
            enable_chips=True,
            rules=rules,
        )
        wildcard_config = SeasonManagerConfig(
            initial_horizon=1,
            transfer_horizon=1,
            chip_lookahead=2,
            max_transfers_per_gw=0,
            wildcard_gain_threshold=-999.0,
            free_hit_gain_threshold=999.0,
            enable_chips=True,
            rules=rules,
            strategic_chip_gameweeks=[2],
        )

        free_hit_result = simulate_season(predictions, gameweeks=[1, 2, 3], config=free_hit_config)
        wildcard_result = simulate_season(predictions, gameweeks=[1, 2], config=wildcard_config)

        self.assertEqual(free_hit_result["decisions"][2]["chip"], "free_hit")
        self.assertEqual(
            free_hit_result["decisions"][2]["free_transfers_after"],
            free_hit_result["decisions"][2]["free_transfers_before"],
        )
        self.assertEqual(wildcard_result["decisions"][1]["chip"], "wildcard")
        self.assertEqual(
            wildcard_result["decisions"][1]["free_transfers_after"],
            wildcard_result["decisions"][1]["free_transfers_before"],
        )

    def test_multi_gameweek_initial_pick_uses_current_gameweek_prices(self) -> None:
        predictions = _prediction_frames()
        predictions[1].loc[predictions[1]["player_id"] == 307, "now_cost_millions"] = 50.0
        predictions[2].loc[predictions[2]["player_id"] == 307, "now_cost_millions"] = 5.0
        config = SeasonManagerConfig(
            initial_horizon=2,
            transfer_horizon=1,
            enable_chips=False,
        )

        result = simulate_season(predictions, gameweeks=[1], config=config)

        self.assertNotIn(307, result["initial_squad"])
        self.assertLessEqual(result["decisions"][0]["team"]["total_cost"], 100.0)

    def test_paid_transfer_hits_are_deducted_from_gameweek_points(self) -> None:
        rules = SeasonRules(free_transfer_per_gameweek=0)
        config = SeasonManagerConfig(
            initial_horizon=1,
            transfer_horizon=1,
            transfer_gain_threshold=0.0,
            max_transfers_per_gw=1,
            enable_chips=False,
            rules=rules,
        )

        result = simulate_season(_prediction_frames(), gameweeks=[1, 2], config=config)

        gw2 = result["decisions"][1]
        self.assertEqual(len(gw2["transfers"]), 1)
        self.assertEqual(gw2["transfer_hit_cost"], 4.0)
        self.assertAlmostEqual(
            gw2["expected_points"],
            gw2["expected_points_before_transfer_hits"] - 4.0,
        )

    def test_transfer_sale_value_uses_half_profit_rule(self) -> None:
        predictions = _prediction_frames()
        predictions[2].loc[predictions[2]["player_id"] == 305, "now_cost_millions"] = 5.9
        config = SeasonManagerConfig(
            initial_horizon=1,
            transfer_horizon=1,
            transfer_gain_threshold=2.0,
            max_transfers_per_gw=1,
            enable_chips=False,
        )

        result = simulate_season(predictions, gameweeks=[1, 2], config=config)

        transfer = result["decisions"][1]["transfers"][0]
        self.assertEqual(transfer["out_player"]["full_name"], "Mid Five")
        self.assertEqual(transfer["out_purchase_price"], 5.5)
        self.assertEqual(transfer["out_sale_value"], 5.7)
        self.assertEqual(transfer["in_purchase_price"], 5.0)
        self.assertGreater(result["decisions"][1]["team_value_m"], 100.0)

    def test_multi_transfer_recommendations_are_ordered_to_keep_bank_non_negative(self) -> None:
        expensive_first = {
            "out_player": {"player_id": 1},
            "in_player": {"player_id": 11},
            "out_sale_value": 4.0,
            "in_purchase_price": 6.0,
        }
        funding_transfer = {
            "out_player": {"player_id": 2},
            "in_player": {"player_id": 12},
            "out_sale_value": 8.0,
            "in_purchase_price": 6.0,
        }
        ordered = _order_transfers_for_execution(
            [expensive_first, funding_transfer],
            starting_bank_m=0.0,
        )
        state = ManagerState(
            squad_player_ids=[1, 2],
            bank_m=0.0,
            free_transfers=2,
            purchase_price_by_player_id={1: 4.0, 2: 8.0},
        )

        _apply_transfers(state, ordered)

        self.assertEqual(ordered[0]["out_player"]["player_id"], 2)
        self.assertEqual(state.squad_player_ids, [11, 12])
        self.assertEqual(state.bank_m, 0.0)

    def test_owned_squad_lineup_does_not_fail_when_team_counts_drift(self) -> None:
        predictions = _prediction_frames()
        drifted_ids = [202, 203, 204, 205]
        predictions[2].loc[predictions[2]["player_id"].isin(drifted_ids), "team_id"] = 1
        predictions[2].loc[predictions[2]["player_id"].isin(drifted_ids), "team_name"] = "Team 1"
        config = SeasonManagerConfig(
            initial_horizon=1,
            transfer_horizon=1,
            enable_chips=False,
        )

        result = simulate_season(predictions, gameweeks=[1, 2], config=config)

        self.assertEqual(result["summary"]["gameweeks"], 2)
        self.assertEqual(len(result["decisions"][1]["starting_player_ids"]), 11)
        self.assertEqual(len(result["decisions"][1]["bench_player_ids"]), 4)

    def test_repeated_simulations_aggregate_runs_and_best_run(self) -> None:
        config = SeasonManagerConfig(
            initial_horizon=1,
            transfer_horizon=1,
            transfer_gain_threshold=2.0,
            max_transfers_per_gw=1,
            enable_chips=False,
        )

        result = run_repeated_season_simulations(
            _prediction_frames(),
            gameweeks=[1, 2],
            simulations=5,
            config=config,
            random_seed=7,
        )

        self.assertEqual(result["summary"]["simulations"], 5)
        self.assertEqual(len(result["runs"]), 5)
        self.assertIn("average_total_expected_points", result["summary"])
        self.assertIn("best_run", result)
        self.assertEqual(result["best_run"]["summary"]["gameweeks"], 2)

    def test_fixed_policy_mode_runs_many_point_simulations_from_ai_policy(self) -> None:
        config = SeasonManagerConfig(
            initial_horizon=1,
            transfer_horizon=1,
            transfer_gain_threshold=2.0,
            max_transfers_per_gw=1,
            enable_chips=False,
        )

        result = run_repeated_season_simulations(
            _prediction_frames(),
            gameweeks=[1, 2],
            simulations=20,
            config=config,
            random_seed=7,
            simulation_mode="fixed_policy",
        )

        self.assertEqual(result["summary"]["simulation_mode"], "fixed_policy")
        self.assertEqual(result["summary"]["simulations"], 20)
        self.assertEqual(result["policy_run"]["summary"]["gameweeks"], 2)
        self.assertEqual(result["policy_run"]["summary"]["transfers_made"], 1)
        self.assertEqual(len(result["runs"]), 20)

    def test_periodic_reoptimization_refreshes_policy_by_interval(self) -> None:
        config = SeasonManagerConfig(
            initial_horizon=1,
            transfer_horizon=1,
            transfer_gain_threshold=2.0,
            max_transfers_per_gw=1,
            enable_chips=False,
        )

        result = run_repeated_season_simulations(
            _prediction_frames(),
            gameweeks=[1, 2],
            simulations=7,
            config=config,
            random_seed=7,
            simulation_mode="periodic_reoptimization",
            policy_refresh_interval=3,
        )

        self.assertEqual(result["summary"]["simulation_mode"], "periodic_reoptimization")
        self.assertEqual(result["summary"]["policy_reoptimizations"], 3)
        self.assertEqual(len(result["policy_runs"]), 3)
        self.assertEqual(len(result["runs"]), 7)
        self.assertEqual(result["runs"][0]["policy_block"], 1)
        self.assertEqual(result["runs"][-1]["policy_block"], 3)

        policy_means = {
            policy["policy_block"]: policy["policy_evaluation"]["average_total_points"]
            for policy in result["policy_runs"]
        }
        recommended_block = result["recommended_policy"]["policy_block"]
        self.assertEqual(recommended_block, max(policy_means, key=policy_means.get))
        self.assertEqual(result["best_run"]["policy_block"], recommended_block)
        self.assertIn("best_outcome_run", result)

    def test_periodic_policy_is_evaluated_on_common_base_predictions(self) -> None:
        base = _prediction_frames()
        sampled = {gw: frame.copy() for gw, frame in base.items()}
        sampled[1]["expected_points"] = sampled[1]["expected_points"] * 2.0
        config = SeasonManagerConfig(enable_chips=False, initial_horizon=1)
        sampled_policy = simulate_season(sampled, gameweeks=[1], config=config)

        rebased = _rebase_policy_run(sampled_policy, base)

        base_lookup = base[1].set_index("player_id")["expected_points"]
        for player in rebased["decisions"][0]["team"]["squad"]:
            self.assertEqual(
                float(player["expected_points"]),
                float(base_lookup.loc[int(player["player_id"])]),
            )
        self.assertLess(
            rebased["summary"]["total_expected_points"],
            sampled_policy["summary"]["total_expected_points"],
        )

    def test_rebased_policy_refreshes_transfer_and_chip_gain_history(self) -> None:
        base = _prediction_frames()
        sampled = {gw: frame.copy() for gw, frame in base.items()}
        sampled[2].loc[
            sampled[2]["player_id"].eq(307), "expected_points"
        ] = 20.0
        transfer_config = SeasonManagerConfig(
            enable_chips=False,
            initial_horizon=1,
            transfer_horizon=1,
            transfer_gain_threshold=0.0,
            max_transfers_per_gw=1,
        )
        transfer_policy = simulate_season(
            sampled,
            gameweeks=[1, 2],
            config=transfer_config,
        )
        rebased_transfer = _rebase_policy_run(
            transfer_policy,
            base,
            transfer_config,
        )
        transfer_decision = rebased_transfer["decisions"][1]
        self.assertNotEqual(
            transfer_decision["transfer_gain"],
            transfer_policy["decisions"][1]["transfer_gain"],
        )
        self.assertEqual(
            transfer_decision["manager_state_after"]["history"][1]["transfer_gain"],
            transfer_decision["transfer_gain"],
        )

        rules = SeasonRules(
            chips_by_half={
                "first": {"free_hit": 0, "wildcard": 0, "bench_boost": 0, "triple_captain": 1},
                "second": {"free_hit": 0, "wildcard": 0, "bench_boost": 0, "triple_captain": 0},
            }
        )
        chip_config = SeasonManagerConfig(
            rules=rules,
            enable_chips=True,
            initial_horizon=1,
            max_transfers_per_gw=0,
            triple_captain_gain_threshold=0.0,
        )
        chip_policy = simulate_season(
            sampled,
            gameweeks=[1],
            config=chip_config,
        )
        rebased_chip = _rebase_policy_run(chip_policy, base, chip_config)
        chip_decision = rebased_chip["decisions"][0]
        captain_points = float(
            base[1].set_index("player_id").loc[chip_decision["captain_id"], "expected_points"]
        )
        self.assertEqual(chip_decision["chip"], "triple_captain")
        self.assertEqual(chip_decision["chip_gain"], captain_points)
        self.assertEqual(
            chip_decision["manager_state_after"]["history"][0]["chip_gain"],
            chip_decision["chip_gain"],
        )

    def test_manager_state_round_trip_and_partial_season_continuation(self) -> None:
        config = SeasonManagerConfig(
            initial_horizon=1,
            transfer_horizon=1,
            transfer_gain_threshold=2.0,
            max_transfers_per_gw=1,
            enable_chips=False,
        )
        opening = simulate_season(_prediction_frames(), gameweeks=[1], config=config)
        state = ManagerState.from_dict(opening["final_state"])

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manager-state.json"
            save_manager_state(state, path)
            loaded = load_manager_state(path)

        continued = simulate_season(
            _prediction_frames(),
            gameweeks=[2],
            config=config,
            initial_state=loaded,
        )

        self.assertEqual(loaded.squad_player_ids, state.squad_player_ids)
        self.assertEqual(loaded.bank_m, state.bank_m)
        self.assertEqual(loaded.free_transfers, state.free_transfers)
        self.assertEqual(
            loaded.purchase_price_by_player_id,
            state.purchase_price_by_player_id,
        )
        self.assertEqual(continued["initial_squad"], state.squad_player_ids)
        self.assertEqual(len(continued["decisions"][0]["transfers"]), 1)
        self.assertEqual(continued["decisions"][0]["transfers"][0]["in_player"]["player_id"], 307)

    def test_live_state_rejects_skipped_or_replayed_gameweeks(self) -> None:
        opening = simulate_season(
            _prediction_frames(),
            gameweeks=[1],
            config=SeasonManagerConfig(enable_chips=False, initial_horizon=1),
        )
        state = ManagerState.from_dict(opening["final_state"])

        with self.assertRaisesRegex(ValueError, "GW2"):
            simulate_season(
                _prediction_frames(),
                gameweeks=[3],
                config=SeasonManagerConfig(enable_chips=False),
                initial_state=state,
            )

    def test_free_hit_gain_is_compared_with_best_normal_transfer(self) -> None:
        rules = SeasonRules(
            chips_by_half={
                "first": {"free_hit": 1, "wildcard": 0, "bench_boost": 0, "triple_captain": 0},
                "second": {"free_hit": 0, "wildcard": 0, "bench_boost": 0, "triple_captain": 0},
            }
        )
        config = SeasonManagerConfig(
            rules=rules,
            initial_horizon=1,
            transfer_horizon=1,
            transfer_gain_threshold=0.0,
            max_transfers_per_gw=1,
            free_hit_gain_threshold=0.1,
            enable_chips=True,
        )

        result = simulate_season(_prediction_frames(), gameweeks=[1, 2], config=config)

        self.assertEqual(len(result["decisions"][1]["transfers"]), 1)
        self.assertIsNone(result["decisions"][1]["chip"])

    def test_decision_exposes_compact_state_for_the_next_live_run(self) -> None:
        config = SeasonManagerConfig(enable_chips=False, initial_horizon=1)

        result = simulate_season(_prediction_frames(), gameweeks=[1], config=config)

        state_after = result["decisions"][0]["manager_state_after"]
        self.assertEqual(state_after["last_processed_gameweek"], 1)
        self.assertEqual(len(state_after["history"]), 1)
        self.assertNotIn("manager_state_after", state_after["history"][0])
        self.assertEqual(
            state_after["squad_player_ids"],
            result["decisions"][0]["squad_player_ids"],
        )

    def test_configured_free_transfer_top_up_is_available_for_target_gameweek(self) -> None:
        rules = SeasonRules(free_transfer_topups={2: 5})
        config = SeasonManagerConfig(
            rules=rules,
            enable_chips=False,
            initial_horizon=1,
            max_transfers_per_gw=0,
        )

        result = simulate_season(_prediction_frames(), gameweeks=[1, 2], config=config)

        self.assertEqual(result["decisions"][0]["free_transfers_after"], 5)
        self.assertEqual(result["decisions"][1]["free_transfers_before"], 5)

    def test_triple_captain_is_held_for_a_better_forecast_gameweek(self) -> None:
        predictions = _prediction_frames()
        predictions[1].loc[predictions[1]["player_id"] == 301, "expected_points"] = 8.0
        predictions[2].loc[predictions[2]["player_id"] == 301, "expected_points"] = 14.0
        rules = SeasonRules(
            chips_by_half={
                "first": {"free_hit": 0, "wildcard": 0, "bench_boost": 0, "triple_captain": 1},
                "second": {"free_hit": 0, "wildcard": 0, "bench_boost": 0, "triple_captain": 0},
            }
        )
        config = SeasonManagerConfig(
            rules=rules,
            enable_chips=True,
            initial_horizon=1,
            max_transfers_per_gw=0,
            triple_captain_gain_threshold=5.0,
        )

        result = simulate_season(predictions, gameweeks=[1, 2], config=config)

        self.assertIsNone(result["decisions"][0]["chip"])
        self.assertEqual(result["decisions"][1]["chip"], "triple_captain")

    def test_prediction_loader_rejects_wrong_or_untagged_season(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            frame = _prediction_frames()[1].copy()
            frame["season_name"] = "2025-26"
            frame["gameweek"] = 1
            frame.to_csv(output_dir / "predictions_gw1.csv", index=False)

            with self.assertRaisesRegex(ValueError, "2026-27"):
                load_prediction_files(output_dir, expected_season_name="2026-27")

            frame.drop(columns=["season_name"]).to_csv(
                output_dir / "predictions_gw1.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "season_name"):
                load_prediction_files(output_dir, expected_season_name="2026-27")

    def test_sampled_scoring_applies_autosub_and_vice_captain(self) -> None:
        starters = [
            {"player_id": 1, "element_type": 1, "bench_order": 0},
            {"player_id": 2, "element_type": 2, "bench_order": 0},
            {"player_id": 3, "element_type": 2, "bench_order": 0},
            {"player_id": 4, "element_type": 2, "bench_order": 0},
            {"player_id": 5, "element_type": 3, "bench_order": 0},
            {"player_id": 6, "element_type": 3, "bench_order": 0},
            {"player_id": 7, "element_type": 3, "bench_order": 0},
            {"player_id": 8, "element_type": 3, "bench_order": 0},
            {"player_id": 9, "element_type": 4, "bench_order": 0},
            {"player_id": 10, "element_type": 4, "bench_order": 0},
            {"player_id": 11, "element_type": 4, "bench_order": 0},
        ]
        bench = [
            {"player_id": 12, "element_type": 3, "bench_order": 1},
            {"player_id": 13, "element_type": 2, "bench_order": 2},
            {"player_id": 14, "element_type": 4, "bench_order": 3},
            {"player_id": 15, "element_type": 1, "bench_order": 4},
        ]
        decision = {
            "team": {"squad": starters, "bench": bench},
            "captain_id": 5,
            "vice_captain_id": 6,
            "chip": None,
            "transfer_hit_cost": 0.0,
        }
        outcomes = {player_id: (True, 2.0) for player_id in range(1, 16)}
        outcomes[5] = (False, 0.0)
        outcomes[8] = (False, 0.0)

        total = _score_decision_outcomes(decision, outcomes)

        # Ten playing starters, first bench replacement, plus vice-captain bonus.
        self.assertEqual(total, 24.0)

    def test_cameo_probability_is_not_taken_from_start_probability(self) -> None:
        class StubRng:
            def random(self) -> float:
                return 0.5

            def normal(self, mean: float, sigma: float) -> float:
                return mean

        played, points = _sample_player_outcome(
            {
                "expected_points": 1.4,
                "expected_points_lower_80": 0.5,
                "expected_points_upper_80": 2.3,
                "start_probability": 0.1,
                "appearance_probability": 0.9,
                "cameo_points": 1.0,
            },
            StubRng(),
            1.0,
        )

        self.assertTrue(played)
        self.assertEqual(points, 1.0)

    def test_sampled_outcomes_match_start_cameo_and_availability_expectation(self) -> None:
        player = {
            "expected_points": 1.65,
            "expected_points_lower_80": 0.5,
            "expected_points_upper_80": 2.8,
            "start_probability": 0.5,
            "appearance_probability": 0.8,
            "availability_next_round": 0.5,
            "cameo_points": 1.0,
        }
        rng = np.random.default_rng(123)

        outcomes = [
            _sample_player_outcome(player, rng, noise_scale=0.0)
            for _ in range(80_000)
        ]
        appearance_rate = np.mean([played for played, _points in outcomes])
        average_points = np.mean([points for _played, points in outcomes])

        self.assertAlmostEqual(appearance_rate, 0.4, delta=0.01)
        self.assertAlmostEqual(average_points, 1.65, delta=0.03)

    def test_repeated_simulations_emit_structured_progress(self) -> None:
        events: list[dict[str, object]] = []

        result = run_repeated_season_simulations(
            _prediction_frames(),
            gameweeks=[1],
            simulations=3,
            config=SeasonManagerConfig(enable_chips=False, initial_horizon=1),
            random_seed=7,
            simulation_mode="fixed_policy",
            progress_callback=events.append,
        )

        self.assertEqual(result["summary"]["simulations"], 3)
        self.assertEqual(events[-1]["event"], "simulation_progress")
        self.assertEqual(events[-1]["completed_simulations"], 3)
        self.assertEqual(events[-1]["total_simulations"], 3)
        self.assertEqual(events[-1]["phase"], "complete")


if __name__ == "__main__":
    unittest.main()
