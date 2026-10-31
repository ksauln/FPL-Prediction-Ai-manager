from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

try:
    import sklearn
except ImportError as exc:  # pragma: no cover - environment guard
    raise unittest.SkipTest("scikit-learn is required for feature workflow tests") from exc
sklearn_version = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
if sklearn_version < (1, 3):  # pragma: no cover - environment guard
    raise unittest.SkipTest("scikit-learn >= 1.3 is required for feature workflow tests")
from fplmodel.features import (
    _add_team_context_features,
    _prepare_player_static_features,
    build_training_and_pred_frames,
    expand_for_double_gw,
)
import fplmodel.model as model_module
from fplmodel.model import ModelCandidate, _evaluate_model_candidates, predict_expected_points
from fplmodel.state import ModelState
from main import (
    _combine_ensemble_expected_points,
    add_prediction_confidence,
    blend_official_expected_points,
    blend_provider_expected_points,
    infer_season_name,
    list_current_player_history_files,
    remap_external_histories_to_current_players,
    scale_prediction_totals_for_fixtures,
    target_gameweek_deadline,
    validate_external_history_coverage,
    validate_season_name,
)


def _elements_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "code": 1001,
                "full_name": "Player One",
                "team_id": 1,
                "element_type": 3,
                "now_cost_millions": 7.5,
                "minutes": 0,
                "status": "a",
                "ep_next": "6.4",
                "price_change_percent": "87",
            },
            {
                "player_id": 2,
                "code": 1002,
                "full_name": "Player Two",
                "team_id": 2,
                "element_type": 4,
                "now_cost_millions": 8.0,
                "minutes": 0,
                "status": "a",
                "ep_next": "2.1",
                "price_change_percent": "-42",
            },
        ]
    )


def _teams_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team_id": 1, "name": "Alpha", "short_name": "ALP"},
            {"team_id": 2, "name": "Beta", "short_name": "BET"},
        ]
    )


def _fixtures_df(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _history_row(
    player_id: int,
    season_name: str,
    round_number: int,
    total_points: float,
    fixture: int,
    team: int,
    opponent_team: int,
) -> dict[str, object]:
    return {
        "player_id": player_id,
        "season_name": season_name,
        "round": round_number,
        "fixture": fixture,
        "kickoff_time": f"{season_name[:4]}-08-{round_number:02d}T12:00:00Z",
        "total_points": total_points,
        "minutes": 90,
        "was_home": team == 1,
        "team": team,
        "opponent_team": opponent_team,
        "team_h_score": 2 if team == 1 else 0,
        "team_a_score": 0 if team == 1 else 2,
        "value": 75,
    }


class FeatureWorkflowTests(unittest.TestCase):
    def test_static_features_parse_real_fpl_status_news_shapes(self) -> None:
        parsed = _prepare_player_static_features(
            pd.DataFrame(
                [
                    {
                        "player_id": 1,
                        "status": "i",
                        "news": "Leg injury - Expected back 28 Nov",
                        "news_added": "2026-08-20T12:00:00Z",
                    },
                    {
                        "player_id": 2,
                        "status": "s",
                        "news": "Suspended until 29 Aug",
                        "news_added": "2026-08-20T12:00:00Z",
                    },
                    {
                        "player_id": 3,
                        "status": "i",
                        "news": "Ankle injury - Expected back unknown",
                        "news_added": "2026-08-20T12:00:00Z",
                    },
                ]
            ),
            current_season_name="2026-27",
        ).set_index("player_id")

        self.assertEqual(str(parsed.loc[1, "status_return_date"].date()), "2026-11-28")
        self.assertEqual(str(parsed.loc[2, "status_return_date"].date()), "2026-08-29")
        self.assertTrue(pd.isna(parsed.loc[3, "status_return_date"]))

    def test_fixture_scaling_keeps_raw_and_corrected_dgw_outputs_comparable(self) -> None:
        predictions = pd.DataFrame(
            {
                "fixture_multiplier": [2, 1, 0],
                "expected_points_raw": [4.0, 4.0, 4.0],
                "expected_points": [4.5, 4.5, 4.5],
                "expected_points_raw__model": [3.0, 3.0, 3.0],
                "expected_points__model": [3.5, 3.5, 3.5],
                "points_hat__model": [5.0, 5.0, 5.0],
                "cameo_points__model": [1.0, 1.0, 1.0],
                "cameo_points": [1.0, 1.0, 1.0],
                "start_probability__model": [0.7, 0.7, 0.7],
            }
        )

        scaled = scale_prediction_totals_for_fixtures(predictions)

        self.assertEqual(scaled["expected_points_raw"].tolist(), [8.0, 4.0, 0.0])
        self.assertEqual(scaled["expected_points"].tolist(), [9.0, 4.5, 0.0])
        self.assertEqual(
            scaled["expected_points_raw__model"].tolist(), [6.0, 3.0, 0.0]
        )
        self.assertEqual(scaled["expected_points__model"].tolist(), [7.0, 3.5, 0.0])
        self.assertEqual(scaled["start_probability__model"].tolist(), [0.7, 0.7, 0.7])

    def test_prediction_features_align_with_sparse_latest_row_indexes(self) -> None:
        rows: list[dict[str, object]] = []
        for player_id, points_offset, team, opponent in [(1, 0, 1, 2), (2, 4, 2, 1)]:
            for round_number in range(1, 5):
                rows.append(
                    _history_row(
                        player_id=player_id,
                        season_name="2025-26",
                        round_number=round_number,
                        total_points=points_offset + round_number,
                        fixture=(player_id * 100) + round_number,
                        team=team,
                        opponent_team=opponent,
                    )
                )
        histories = pd.DataFrame(rows)
        histories.index = range(10, 10 + len(histories))

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                _elements_df(),
                _teams_df(),
                histories,
                next_gw=5,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
            )

        pred_by_player = X_pred.set_index("player_id")
        self.assertEqual(pred_by_player.loc[1, "total_points_lag1"], 4)
        self.assertEqual(pred_by_player.loc[2, "total_points_lag1"], 8)
        self.assertAlmostEqual(pred_by_player.loc[1, "total_points_ma5"], 2.5)
        self.assertAlmostEqual(pred_by_player.loc[2, "total_points_ma5"], 6.5)

    def test_training_rows_keep_one_match_behind_features(self) -> None:
        histories = pd.DataFrame(
            [
                _history_row(1, "2025-26", round_number, round_number, round_number, 1, 2)
                for round_number in range(1, 6)
            ]
        )

        with TemporaryDirectory() as tmpdir:
            X_train, y_train, _, _ = build_training_and_pred_frames(
                _elements_df().iloc[[0]].copy(),
                _teams_df(),
                histories,
                next_gw=6,
                last_finished_gw=5,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                current_season_name="2025-26",
            )

        target_row = X_train.loc[y_train.eq(5.0)].iloc[-1]
        self.assertEqual(float(target_row["total_points_lag1"]), 4.0)
        self.assertAlmostEqual(float(target_row["total_points_ma5"]), 2.5)

    def test_training_uses_completed_prior_seasons_beyond_current_gw(self) -> None:
        rows = [
            _history_row(1, "2024-25", round_number, round_number, round_number, 1, 2)
            for round_number in range(1, 6)
        ]
        rows.append(_history_row(1, "2025-26", 1, 10, 101, 1, 2))

        with TemporaryDirectory() as tmpdir:
            X_train, y_train, _, _ = build_training_and_pred_frames(
                _elements_df().iloc[[0]].copy(),
                _teams_df(),
                pd.DataFrame(rows),
                next_gw=2,
                last_finished_gw=1,
                state=ModelState(path=Path(tmpdir) / "state.json"),
            )

        self.assertGreaterEqual(len(X_train), 2)
        self.assertIn(4.0, set(y_train.tolist()))
        self.assertIn(5.0, set(y_train.tolist()))

    def test_preseason_cold_start_trains_on_prior_year_and_includes_new_players(self) -> None:
        histories = pd.DataFrame(
            [
                _history_row(1, "2025-26", round_number, round_number, round_number, 1, 2)
                for round_number in range(1, 6)
            ]
        )

        with TemporaryDirectory() as tmpdir:
            X_train, _, X_pred, _ = build_training_and_pred_frames(
                _elements_df(),
                _teams_df(),
                histories,
                next_gw=1,
                last_finished_gw=0,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                current_season_name="2026-27",
            )

        self.assertGreater(len(X_train), 0)
        self.assertEqual(set(X_pred["player_id"]), {1, 2})
        new_player = X_pred.set_index("player_id").loc[2]
        self.assertEqual(float(new_player["history_match_count"]), 0.0)
        self.assertEqual(float(new_player["current_season_match_count"]), 0.0)

    def test_prediction_counts_only_completed_current_season_matches(self) -> None:
        rows = [
            _history_row(1, "2025-26", round_number, 4, round_number, 1, 2)
            for round_number in range(1, 6)
        ]
        rows.extend(
            _history_row(1, "2026-27", round_number, 4, 100 + round_number, 1, 2)
            for round_number in range(1, 4)
        )

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                _elements_df().iloc[[0]].copy(),
                _teams_df(),
                pd.DataFrame(rows),
                next_gw=3,
                last_finished_gw=2,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                current_season_name="2026-27",
            )

        prediction = X_pred.iloc[0]
        self.assertEqual(float(prediction["history_match_count"]), 7.0)
        self.assertEqual(float(prediction["current_season_match_count"]), 2.0)

    def test_role_evidence_is_prediction_metadata_not_a_training_feature(self) -> None:
        histories = pd.DataFrame(
            [
                _history_row(1, "2025-26", round_number, 4, round_number, 1, 2)
                for round_number in range(1, 6)
            ]
        )
        preseason = pd.DataFrame(
            {
                "player_code": [1001, 1002],
                # Published enrichment snapshots retain identity labels for
                # operator auditing.  The live team mapping must remain the
                # canonical prediction label after this frame is merged.
                "team_name": ["Archived Alpha", "Archived Beta"],
                "team_preseason_matches": [4, 4],
                "preseason_appearances": [4, 4],
                "preseason_starts": [4, 3],
                "preseason_minutes": [320, 260],
                "preseason_last_two_minutes": [170, 140],
                "source_confidence": [0.8, 0.8],
                "role_target_gameweek": [1, 1],
                "role_change_score": [0.45, -0.35],
                "role_evidence_quality": [0.8, 0.6],
                "role_data_source": ["manual:test", "manual:test"],
            }
        )

        with TemporaryDirectory() as tmpdir:
            X_train, _, X_pred, _ = build_training_and_pred_frames(
                _elements_df(),
                _teams_df(),
                histories,
                next_gw=1,
                last_finished_gw=0,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                current_season_name="2026-27",
                preseason_player_df=preseason,
            )

        self.assertNotIn("role_change_score", X_train.columns)
        self.assertEqual(X_pred.set_index("player_id")["team_name"].to_dict(), {1: "Alpha", 2: "Beta"})
        role_by_player = X_pred.set_index("player_id")["role_change_score"]
        self.assertAlmostEqual(float(role_by_player.loc[1]), 0.45)
        self.assertAlmostEqual(float(role_by_player.loc[2]), -0.35)

    def test_historical_replay_uses_the_price_from_the_target_round(self) -> None:
        rows = [
            {
                **_history_row(1, "2025-26", round_number, round_number, round_number, 1, 2),
                "value": 70 + round_number,
            }
            for round_number in range(1, 5)
        ]

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                _elements_df().iloc[[0]].copy(),
                _teams_df(),
                pd.DataFrame(rows),
                next_gw=4,
                last_finished_gw=3,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                current_season_name="2025-26",
            )

        self.assertAlmostEqual(float(X_pred.iloc[0]["now_cost_millions"]), 7.4)

    def test_prediction_row_uses_target_fixture_home_and_away_context(self) -> None:
        rows: list[dict[str, object]] = []
        for player_id, team, opponent in [(1, 1, 2), (2, 2, 1)]:
            for round_number in range(1, 5):
                rows.append(
                    _history_row(
                        player_id,
                        "2025-26",
                        round_number,
                        4,
                        round_number,
                        team,
                        opponent,
                    )
                )
        fixtures = pd.DataFrame([{"event": 5, "team_h": 1, "team_a": 2}])

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                _elements_df(),
                _teams_df(),
                pd.DataFrame(rows),
                next_gw=5,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                fixtures_df=fixtures,
                current_season_name="2025-26",
            )

        pred = X_pred.set_index("player_id")
        self.assertEqual(float(pred.loc[1, "was_home"]), 1.0)
        self.assertEqual(float(pred.loc[2, "was_home"]), 0.0)

    def test_target_team_context_excludes_rows_after_last_finished_gameweek(self) -> None:
        rows = []
        for player_id, team, opponent in [(1, 1, 2), (2, 2, 1)]:
            first = _history_row(player_id, "2025-26", 1, 4, 1, team, opponent)
            future = _history_row(player_id, "2025-26", 3, 4, 3, team, opponent)
            if team == 1:
                first.update({"team_h_score": 2, "team_a_score": 0, "was_home": True})
                future.update({"team_h_score": 9, "team_a_score": 0, "was_home": True})
            else:
                first.update({"team_h_score": 2, "team_a_score": 0, "was_home": False})
                future.update({"team_h_score": 9, "team_a_score": 0, "was_home": False})
            rows.extend([first, future])
        fixtures = pd.DataFrame([{"event": 2, "team_h": 1, "team_a": 2}])

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                _elements_df(),
                _teams_df(),
                pd.DataFrame(rows),
                next_gw=2,
                last_finished_gw=1,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                fixtures_df=fixtures,
                current_season_name="2025-26",
            )

        pred = X_pred.set_index("player_id")
        self.assertEqual(float(pred.loc[1, "team_goals_for_ma5"]), 2.0)
        self.assertEqual(float(pred.loc[2, "team_goals_for_ma5"]), 0.0)

    def test_gw1_team_context_uses_prior_season_stable_team_codes(self) -> None:
        teams = _teams_df().assign(code=[101, 202])
        rows: list[dict[str, object]] = []
        for round_number in range(1, 6):
            rows.extend(
                [
                    {
                        **_history_row(1, "2025-26", round_number, 4, round_number, 91, 92),
                        "team_code": 101,
                        "opponent_team_code": 202,
                        "was_home": True,
                        "team_h_score": 2,
                        "team_a_score": 0,
                    },
                    {
                        **_history_row(2, "2025-26", round_number, 2, round_number, 92, 91),
                        "team_code": 202,
                        "opponent_team_code": 101,
                        "was_home": False,
                        "team_h_score": 2,
                        "team_a_score": 0,
                    },
                ]
            )
        fixtures = pd.DataFrame([{"event": 1, "team_h": 1, "team_a": 2}])

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                _elements_df(),
                teams,
                pd.DataFrame(rows),
                next_gw=1,
                last_finished_gw=0,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                fixtures_df=fixtures,
                current_season_name="2026-27",
            )

        pred = X_pred.set_index("player_id")
        self.assertEqual(float(pred.loc[1, "team_goals_for_ma5"]), 2.0)
        self.assertEqual(float(pred.loc[2, "team_goals_against_ma5"]), 2.0)
        self.assertEqual(pred.loc[1, "team_prior_source"], "history:2025-26")

    def test_promoted_team_context_uses_optional_team_prior_snapshot(self) -> None:
        elements = _elements_df().iloc[[0]].copy()
        elements["team_id"] = 7
        teams = pd.DataFrame(
            [
                {"team_id": 7, "name": "Coventry City", "code": 9},
                {"team_id": 1, "name": "Alpha", "code": 3},
            ]
        )
        histories = pd.DataFrame(
            [
                {
                    **_history_row(1, "2025-26", round_number, 4, round_number, 1, 2),
                    "team_code": 3,
                }
                for round_number in range(1, 6)
            ]
        )
        fixtures = pd.DataFrame([{"event": 1, "team_h": 7, "team_a": 1}])
        team_priors = pd.DataFrame(
            {
                "team_code": [9],
                "team_goals_for_ma5": [1.8],
                "team_goals_against_ma5": [0.9],
                "team_goal_diff_ma5": [0.9],
                "team_clean_sheet_ma5": [0.4],
                "team_conceded_two_plus_ma5": [0.2],
                "team_match_points_ma5": [2.2],
                "team_prior_source": ["football_data_co_uk:2025-26:E1"],
            }
        )

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                elements,
                teams,
                histories,
                next_gw=1,
                last_finished_gw=0,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                fixtures_df=fixtures,
                current_season_name="2026-27",
                team_priors_df=team_priors,
            )

        prediction = X_pred.iloc[0]
        self.assertAlmostEqual(float(prediction["team_goals_for_ma5"]), 1.8)
        self.assertEqual(prediction["team_prior_source"], "football_data_co_uk:2025-26:E1")

    def test_current_position_fdr_and_live_signals_reach_prediction(self) -> None:
        rows = [
            {
                **_history_row(1, "2025-26", round_number, round_number, round_number, 1, 2),
                "element_type": 4,
                "official_expected_points": float(round_number) + 0.5,
                "fixture_difficulty": 4,
            }
            for round_number in range(1, 5)
        ]
        fixtures = pd.DataFrame(
            [
                {
                    "event": 5,
                    "team_h": 1,
                    "team_a": 2,
                    "team_h_difficulty": 2,
                    "team_a_difficulty": 4,
                }
            ]
        )

        with TemporaryDirectory() as tmpdir:
            X_train, _, X_pred, train_metadata = build_training_and_pred_frames(
                _elements_df().iloc[[0]].copy(),
                _teams_df(),
                pd.DataFrame(rows),
                next_gw=5,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                fixtures_df=fixtures,
                current_season_name="2026-27",
            )

        self.assertNotIn("official_expected_points", X_train.columns)
        self.assertIn("fixture_difficulty", X_train.columns)
        self.assertEqual(float(X_train.iloc[-1]["position_fwd"]), 1.0)
        self.assertEqual(int(train_metadata.iloc[-1]["element_type"]), 4)

        prediction = X_pred.iloc[0]
        self.assertAlmostEqual(float(prediction["official_ep_next"]), 6.4)
        self.assertEqual(float(prediction["fixture_difficulty"]), 2.0)
        self.assertEqual(float(prediction["position_mid"]), 1.0)
        self.assertEqual(float(prediction["position_fwd"]), 0.0)
        self.assertEqual(float(prediction["price_change_percent"]), 87.0)

    def test_future_replay_does_not_reuse_ep_next_for_the_wrong_gameweek(self) -> None:
        rows = [
            {
                **_history_row(1, "2025-26", round_number, round_number, round_number, 1, 2),
                "official_expected_points": float(round_number) + 0.5,
            }
            for round_number in range(1, 5)
        ]

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                _elements_df().iloc[[0]].copy(),
                _teams_df(),
                pd.DataFrame(rows),
                next_gw=6,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                current_season_name="2026-27",
                official_next_gw=5,
            )

        self.assertTrue(pd.isna(X_pred.iloc[0]["official_ep_next"]))
        self.assertNotIn("official_expected_points", X_pred.columns)

    def test_future_replay_neutralizes_transient_availability_flags(self) -> None:
        elements = _elements_df().iloc[[0]].copy()
        elements.loc[:, "status"] = "d"
        elements.loc[:, "chance_of_playing_this_round"] = 25
        elements.loc[:, "chance_of_playing_next_round"] = 50
        histories = pd.DataFrame(
            [_history_row(1, "2025-26", round_number, round_number, round_number, 1, 2) for round_number in range(1, 5)]
        )

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                elements,
                _teams_df(),
                histories,
                next_gw=8,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                current_season_name="2026-27",
                official_next_gw=5,
            )

        prediction = X_pred.iloc[0]
        self.assertAlmostEqual(float(prediction["availability_this_round"]), 1.0)
        self.assertAlmostEqual(float(prediction["availability_next_round"]), 1.0)
        self.assertAlmostEqual(float(prediction["status_availability"]), 1.0)
        self.assertEqual(int(prediction["status_injury_flag"]), 0)
        self.assertEqual(int(prediction["injury_risk_flag"]), 0)

    def test_future_replay_keeps_dated_injury_before_explicit_return_date(self) -> None:
        elements = _elements_df().iloc[[0]].copy()
        elements.loc[:, "status"] = "i"
        elements.loc[:, "chance_of_playing_this_round"] = 0
        elements.loc[:, "chance_of_playing_next_round"] = 0
        elements.loc[:, "news"] = "Expected back 14 Sep"
        elements.loc[:, "news_added"] = "2026-08-20T12:00:00Z"
        histories = pd.DataFrame(
            [_history_row(1, "2025-26", round_number, round_number, round_number, 1, 2) for round_number in range(1, 5)]
        )
        fixtures = _fixtures_df(
            {
                "event": 8,
                "team_h": 1,
                "team_a": 2,
                "kickoff_time": "2026-09-13T12:00:00Z",
            }
        )

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                elements,
                _teams_df(),
                histories,
                next_gw=8,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                fixtures_df=fixtures,
                current_season_name="2026-27",
                official_next_gw=5,
            )

        prediction = X_pred.iloc[0]
        self.assertEqual(str(prediction["status_return_date"].date()), "2026-09-14")
        self.assertEqual(int(prediction["future_status_applied_flag"]), 1)
        self.assertAlmostEqual(float(prediction["availability_this_round"]), 0.0)
        self.assertAlmostEqual(float(prediction["availability_next_round"]), 0.0)
        self.assertAlmostEqual(float(prediction["status_availability"]), 0.0)
        self.assertEqual(int(prediction["status_injury_flag"]), 1)
        self.assertEqual(int(prediction["injury_risk_flag"]), 1)

    def test_future_replay_clears_dated_injury_on_explicit_return_date_boundary(self) -> None:
        elements = _elements_df().iloc[[0]].copy()
        elements.loc[:, "status"] = "i"
        elements.loc[:, "chance_of_playing_this_round"] = 0
        elements.loc[:, "chance_of_playing_next_round"] = 0
        elements.loc[:, "news"] = "Expected back 14 Sep"
        elements.loc[:, "news_added"] = "2026-08-20T12:00:00Z"
        histories = pd.DataFrame(
            [_history_row(1, "2025-26", round_number, round_number, round_number, 1, 2) for round_number in range(1, 5)]
        )
        fixtures = _fixtures_df(
            {
                "event": 8,
                "team_h": 1,
                "team_a": 2,
                "kickoff_time": "2026-09-14T12:00:00Z",
            }
        )

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                elements,
                _teams_df(),
                histories,
                next_gw=8,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                fixtures_df=fixtures,
                current_season_name="2026-27",
                official_next_gw=5,
            )

        prediction = X_pred.iloc[0]
        self.assertEqual(str(prediction["status_return_date"].date()), "2026-09-14")
        self.assertEqual(int(prediction["future_status_applied_flag"]), 0)
        self.assertAlmostEqual(float(prediction["availability_this_round"]), 1.0)
        self.assertAlmostEqual(float(prediction["availability_next_round"]), 1.0)
        self.assertAlmostEqual(float(prediction["status_availability"]), 1.0)
        self.assertEqual(int(prediction["status_injury_flag"]), 0)
        self.assertEqual(int(prediction["injury_risk_flag"]), 0)

    def test_future_replay_keeps_dated_suspension_before_explicit_return_date(self) -> None:
        elements = _elements_df().iloc[[0]].copy()
        elements.loc[:, "status"] = "s"
        elements.loc[:, "chance_of_playing_this_round"] = 0
        elements.loc[:, "chance_of_playing_next_round"] = 0
        elements.loc[:, "news"] = "Suspended until 17 Sep"
        elements.loc[:, "news_added"] = "2026-08-20T12:00:00Z"
        histories = pd.DataFrame(
            [_history_row(1, "2025-26", round_number, round_number, round_number, 1, 2) for round_number in range(1, 5)]
        )
        fixtures = _fixtures_df(
            {
                "event": 8,
                "team_h": 1,
                "team_a": 2,
                "kickoff_time": "2026-09-16T12:00:00Z",
            }
        )

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                elements,
                _teams_df(),
                histories,
                next_gw=8,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                fixtures_df=fixtures,
                current_season_name="2026-27",
                official_next_gw=5,
            )

        prediction = X_pred.iloc[0]
        self.assertEqual(str(prediction["status_return_date"].date()), "2026-09-17")
        self.assertEqual(int(prediction["future_status_applied_flag"]), 1)
        self.assertAlmostEqual(float(prediction["availability_this_round"]), 0.0)
        self.assertAlmostEqual(float(prediction["availability_next_round"]), 0.0)
        self.assertAlmostEqual(float(prediction["status_availability"]), 0.0)
        self.assertEqual(int(prediction["status_injury_flag"]), 0)
        self.assertEqual(int(prediction["injury_risk_flag"]), 1)

    def test_future_replay_keeps_persistent_unavailability_flags(self) -> None:
        elements = _elements_df().iloc[[0]].copy()
        elements.loc[:, "status"] = "u"
        elements.loc[:, "chance_of_playing_this_round"] = 0
        elements.loc[:, "chance_of_playing_next_round"] = 0
        histories = pd.DataFrame(
            [_history_row(1, "2025-26", round_number, round_number, round_number, 1, 2) for round_number in range(1, 5)]
        )

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                elements,
                _teams_df(),
                histories,
                next_gw=8,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                current_season_name="2026-27",
                official_next_gw=5,
            )

        prediction = X_pred.iloc[0]
        self.assertAlmostEqual(float(prediction["availability_this_round"]), 0.0)
        self.assertAlmostEqual(float(prediction["availability_next_round"]), 0.0)
        self.assertAlmostEqual(float(prediction["status_availability"]), 0.0)
        self.assertEqual(int(prediction["status_injury_flag"]), 0)
        self.assertEqual(int(prediction["injury_risk_flag"]), 1)

    def test_future_replay_neutralizes_undated_suspension_flags(self) -> None:
        elements = _elements_df().iloc[[0]].copy()
        elements.loc[:, "status"] = "s"
        elements.loc[:, "chance_of_playing_this_round"] = 0
        elements.loc[:, "chance_of_playing_next_round"] = 0
        histories = pd.DataFrame(
            [_history_row(1, "2025-26", round_number, round_number, round_number, 1, 2) for round_number in range(1, 5)]
        )

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                elements,
                _teams_df(),
                histories,
                next_gw=8,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                current_season_name="2026-27",
                official_next_gw=5,
            )

        prediction = X_pred.iloc[0]
        self.assertTrue(pd.isna(prediction["status_return_date"]))
        self.assertEqual(int(prediction["future_status_applied_flag"]), 0)
        self.assertAlmostEqual(float(prediction["availability_this_round"]), 1.0)
        self.assertAlmostEqual(float(prediction["availability_next_round"]), 1.0)
        self.assertAlmostEqual(float(prediction["status_availability"]), 1.0)
        self.assertEqual(int(prediction["status_injury_flag"]), 0)
        self.assertEqual(int(prediction["injury_risk_flag"]), 0)

    def test_future_replay_neutralizes_unknown_return_injury(self) -> None:
        elements = _elements_df().iloc[[0]].copy()
        elements.loc[:, "status"] = "i"
        elements.loc[:, "chance_of_playing_this_round"] = 0
        elements.loc[:, "chance_of_playing_next_round"] = 0
        elements.loc[:, "news"] = "Expected back unknown"
        elements.loc[:, "news_added"] = "2026-08-20T12:00:00Z"
        histories = pd.DataFrame(
            [_history_row(1, "2025-26", round_number, round_number, round_number, 1, 2) for round_number in range(1, 5)]
        )
        fixtures = _fixtures_df(
            {
                "event": 8,
                "team_h": 1,
                "team_a": 2,
                "kickoff_time": "2026-09-10T12:00:00Z",
            }
        )

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                elements,
                _teams_df(),
                histories,
                next_gw=8,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                fixtures_df=fixtures,
                current_season_name="2026-27",
                official_next_gw=5,
            )

        prediction = X_pred.iloc[0]
        self.assertTrue(pd.isna(prediction["status_return_date"]))
        self.assertEqual(int(prediction["future_status_applied_flag"]), 0)
        self.assertAlmostEqual(float(prediction["availability_this_round"]), 1.0)
        self.assertAlmostEqual(float(prediction["availability_next_round"]), 1.0)
        self.assertAlmostEqual(float(prediction["status_availability"]), 1.0)
        self.assertEqual(int(prediction["status_injury_flag"]), 0)
        self.assertEqual(int(prediction["injury_risk_flag"]), 0)

    def test_official_next_gameweek_preserves_current_availability_flags(self) -> None:
        elements = _elements_df().iloc[[0]].copy()
        elements.loc[:, "status"] = "s"
        elements.loc[:, "chance_of_playing_this_round"] = 100
        elements.loc[:, "chance_of_playing_next_round"] = 0
        elements.loc[:, "news"] = "Suspended until 17 Sep"
        elements.loc[:, "news_added"] = "2026-08-20T12:00:00Z"
        histories = pd.DataFrame(
            [_history_row(1, "2025-26", round_number, round_number, round_number, 1, 2) for round_number in range(1, 5)]
        )
        fixtures = _fixtures_df(
            {
                "event": 5,
                "team_h": 1,
                "team_a": 2,
                "kickoff_time": "2026-09-05T12:00:00Z",
            }
        )

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                elements,
                _teams_df(),
                histories,
                next_gw=5,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                fixtures_df=fixtures,
                current_season_name="2026-27",
                official_next_gw=5,
            )

        prediction = X_pred.iloc[0]
        self.assertEqual(str(prediction["status_return_date"].date()), "2026-09-17")
        self.assertEqual(int(prediction["future_status_applied_flag"]), 0)
        self.assertAlmostEqual(float(prediction["availability_this_round"]), 1.0)
        self.assertAlmostEqual(float(prediction["availability_next_round"]), 0.0)
        self.assertAlmostEqual(float(prediction["status_availability"]), 0.0)
        self.assertEqual(int(prediction["status_injury_flag"]), 0)
        self.assertEqual(int(prediction["injury_risk_flag"]), 1)

    def test_future_replay_neutralizes_dated_status_without_target_fixture(self) -> None:
        elements = _elements_df().iloc[[0]].copy()
        elements.loc[:, "status"] = "s"
        elements.loc[:, "chance_of_playing_this_round"] = 0
        elements.loc[:, "chance_of_playing_next_round"] = 0
        elements.loc[:, "news"] = "Suspended until 17 Sep"
        elements.loc[:, "news_added"] = "2026-08-20T12:00:00Z"
        histories = pd.DataFrame(
            [_history_row(1, "2025-26", round_number, round_number, round_number, 1, 2) for round_number in range(1, 5)]
        )
        fixtures = _fixtures_df(
            {
                "event": 8,
                "team_h": 2,
                "team_a": 3,
                "kickoff_time": "2026-09-16T12:00:00Z",
            }
        )

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                elements,
                _teams_df(),
                histories,
                next_gw=8,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                fixtures_df=fixtures,
                current_season_name="2026-27",
                official_next_gw=5,
            )

        prediction = X_pred.iloc[0]
        self.assertEqual(str(prediction["status_return_date"].date()), "2026-09-17")
        self.assertEqual(int(prediction["future_status_applied_flag"]), 0)
        self.assertTrue(pd.isna(prediction["target_fixture_kickoff"]))
        self.assertAlmostEqual(float(prediction["availability_this_round"]), 1.0)
        self.assertAlmostEqual(float(prediction["availability_next_round"]), 1.0)
        self.assertAlmostEqual(float(prediction["status_availability"]), 1.0)
        self.assertEqual(int(prediction["status_injury_flag"]), 0)
        self.assertEqual(int(prediction["injury_risk_flag"]), 0)

    def test_future_replay_counterfactual_current_bootstrap_shapes(self) -> None:
        elements = pd.DataFrame(
            [
                {
                    **_elements_df().iloc[0].to_dict(),
                    "player_id": 1,
                    "code": 1001,
                    "full_name": "Yankuba Minteh",
                    "team_id": 1,
                    "status": "i",
                    "chance_of_playing_this_round": 0,
                    "chance_of_playing_next_round": 0,
                    "news": "Leg injury - Expected back 28 Nov",
                    "news_added": "2026-08-20T12:00:00Z",
                },
                {
                    **_elements_df().iloc[1].to_dict(),
                    "player_id": 2,
                    "code": 1002,
                    "full_name": "Carlos Baleba",
                    "team_id": 2,
                    "status": "i",
                    "chance_of_playing_this_round": 0,
                    "chance_of_playing_next_round": 0,
                    "news": "Ankle injury - Expected back 23 Aug",
                    "news_added": "2026-08-20T12:00:00Z",
                },
            ]
        )
        histories = pd.DataFrame(
            [
                _history_row(1, "2025-26", round_number, round_number, round_number, 1, 2)
                for round_number in range(1, 5)
            ]
            + [
                _history_row(2, "2025-26", round_number, round_number, 100 + round_number, 2, 1)
                for round_number in range(1, 5)
            ]
        )
        fixtures = _fixtures_df(
            {
                "event": 8,
                "team_h": 1,
                "team_a": 3,
                "kickoff_time": "2026-11-27T12:00:00Z",
            },
            {
                "event": 8,
                "team_h": 2,
                "team_a": 4,
                "kickoff_time": "2026-08-24T12:00:00Z",
            },
        )

        with TemporaryDirectory() as tmpdir:
            _, _, X_pred, _ = build_training_and_pred_frames(
                elements,
                _teams_df(),
                histories,
                next_gw=8,
                last_finished_gw=4,
                state=ModelState(path=Path(tmpdir) / "state.json"),
                fixtures_df=fixtures,
                current_season_name="2026-27",
                official_next_gw=5,
            )

        pred = X_pred.set_index("full_name")
        self.assertEqual(int(pred.loc["Yankuba Minteh", "future_status_applied_flag"]), 1)
        self.assertAlmostEqual(float(pred.loc["Yankuba Minteh", "status_availability"]), 0.0)
        self.assertEqual(int(pred.loc["Carlos Baleba", "future_status_applied_flag"]), 0)
        self.assertAlmostEqual(float(pred.loc["Carlos Baleba", "status_availability"]), 1.0)

    def test_blank_gameweek_players_receive_zero_fixture_multiplier(self) -> None:
        predictions = pd.DataFrame({"player_id": [1, 2], "team_id": [1, 2]})
        fixtures = pd.DataFrame([{"event": 7, "team_h": 1, "team_a": 3}])

        result = expand_for_double_gw(predictions, fixtures, next_gw=7)

        multipliers = result.set_index("player_id")["fixture_multiplier"]
        self.assertEqual(float(multipliers.loc[1]), 1.0)
        self.assertEqual(float(multipliers.loc[2]), 0.0)

    def test_team_context_rolls_over_unique_team_fixtures(self) -> None:
        hist = pd.DataFrame(
            [
                _history_row(1, "2025-26", 1, 0, 1, 1, 2),
                _history_row(3, "2025-26", 1, 0, 1, 1, 2),
                {
                    **_history_row(1, "2025-26", 2, 0, 2, 1, 2),
                    "team_h_score": 1,
                    "team_a_score": 1,
                },
                {
                    **_history_row(3, "2025-26", 2, 0, 2, 1, 2),
                    "team_h_score": 1,
                    "team_a_score": 1,
                },
            ]
        )

        out = _add_team_context_features(hist, windows=(5,))
        fixture_two = out[out["fixture"] == 2].sort_values("player_id")

        self.assertEqual(fixture_two["team_goals_for_ma5"].tolist(), [2.0, 2.0])

    def test_prediction_confidence_adds_score_level_and_interval(self) -> None:
        predictions = pd.DataFrame(
            {
                "player_id": [1, 2],
                "expected_points": [6.0, 3.0],
                "expected_points__a": [5.8, 1.5],
                "expected_points__b": [6.2, 4.5],
                "start_probability__a": [0.9, 0.55],
                "start_probability__b": [0.92, 0.45],
                "reliability_weight": [1.0, 0.3],
                "availability_next_round": [1.0, 0.5],
            }
        )

        out = add_prediction_confidence(
            predictions,
            per_model_corrected_cols=["expected_points__a", "expected_points__b"],
            per_model_start_cols=["start_probability__a", "start_probability__b"],
        )

        self.assertIn("confidence_score", out.columns)
        self.assertIn("confidence_level", out.columns)
        self.assertIn("expected_points_lower_80", out.columns)
        self.assertIn("expected_points_upper_80", out.columns)
        self.assertGreater(out.loc[0, "confidence_score"], out.loc[1, "confidence_score"])
        self.assertLessEqual(out.loc[0, "expected_points_lower_80"], out.loc[0, "expected_points"])
        self.assertGreaterEqual(out.loc[0, "expected_points_upper_80"], out.loc[0, "expected_points"])

    def test_sparse_player_without_preseason_evidence_has_low_confidence(self) -> None:
        predictions = pd.DataFrame(
            {
                "expected_points": [4.0],
                "expected_points__a": [4.0],
                "expected_points__b": [4.0],
                "start_probability__a": [0.01],
                "start_probability__b": [0.01],
                "reliability_weight": [0.0],
                "preseason_prior_quality": [0.0],
                "availability_next_round": [1.0],
            }
        )

        out = add_prediction_confidence(
            predictions,
            per_model_corrected_cols=["expected_points__a", "expected_points__b"],
            per_model_start_cols=["start_probability__a", "start_probability__b"],
        )

        self.assertEqual(out.loc[0, "confidence_level"], "Low")
        self.assertLess(float(out.loc[0, "confidence_score"]), 50.0)

    def test_predict_expected_points_uses_separate_start_and_appearance_probabilities(self) -> None:
        class StubClassifier:
            def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
                return np.array([[0.99, 0.01]])

        class StubAppearanceClassifier:
            def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
                return np.array([[0.5, 0.5]])

        class StubRegressor:
            def predict(self, feats: pd.DataFrame) -> np.ndarray:
                return np.array([6.0])

        with TemporaryDirectory() as tmpdir:
            state = ModelState(path=Path(tmpdir) / "state.json")
            state.player_bias["1"] = 4.0
            state.position_bias["1"] = 2.0

            X_pred = pd.DataFrame(
                {
                    "player_id": [1],
                    "full_name": ["Reserve Keeper"],
                    "team_name": ["Alpha"],
                    "now_cost_millions": [4.0],
                    "team_id": [1],
                    "element_type": [1],
                    "season_minutes": [270],
                    "feature_a": [1.0],
                }
            )

            out = predict_expected_points(
                X_pred,
                StubClassifier(),
                StubRegressor(),
                state,
                appearance_clf=StubAppearanceClassifier(),
            )

        self.assertAlmostEqual(out.loc[0, "p_start"], 0.01, places=6)
        self.assertAlmostEqual(out.loc[0, "p_appearance"], 0.5, places=6)
        self.assertAlmostEqual(out.loc[0, "expected_points_raw"], 0.55, places=6)
        self.assertAlmostEqual(out.loc[0, "expected_points"], 0.61, places=6)

    def test_zero_current_season_minutes_do_not_zero_prediction(self) -> None:
        class StubClassifier:
            def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
                return np.array([[0.1, 0.9]])

        class StubRegressor:
            def predict(self, feats: pd.DataFrame) -> np.ndarray:
                return np.array([5.0])

        class StubAppearanceClassifier:
            def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
                return np.array([[0.0, 1.0]])

        with TemporaryDirectory() as tmpdir:
            state = ModelState(path=Path(tmpdir) / "state.json")
            X_pred = pd.DataFrame(
                {
                    "player_id": [1],
                    "full_name": ["New Signing"],
                    "team_name": ["Alpha"],
                    "now_cost_millions": [6.0],
                    "team_id": [1],
                    "element_type": [3],
                    "season_minutes": [0],
                    "history_match_count": [0],
                    "feature_a": [0.0],
                }
            )

            out = predict_expected_points(
                X_pred,
                StubClassifier(),
                StubRegressor(),
                state,
                appearance_clf=StubAppearanceClassifier(),
            )

        self.assertEqual(float(out.loc[0, "reliability_weight"]), 0.0)
        self.assertAlmostEqual(float(out.loc[0, "expected_points"]), 4.6)

    def test_preseason_usage_prior_replaces_unreliable_start_probabilities(self) -> None:
        class StubClassifier:
            def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
                return np.array([[0.9, 0.1], [0.9, 0.1]])

        class StubAppearanceClassifier:
            def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
                return np.array([[0.8, 0.2], [0.8, 0.2]])

        class StubRegressor:
            def predict(self, feats: pd.DataFrame) -> np.ndarray:
                return np.array([6.0, 6.0])

        X_pred = pd.DataFrame(
            {
                "player_id": [1, 2],
                "full_name": ["Cold Starter", "Established Player"],
                "team_name": ["Alpha", "Beta"],
                "now_cost_millions": [6.0, 8.0],
                "team_id": [1, 2],
                "element_type": [3, 4],
                "history_match_count": [0, 20],
                "prior_p_start": [0.9, 0.9],
                "prior_p_appearance": [0.98, 0.98],
                "preseason_prior_quality": [1.0, 1.0],
                "readiness_multiplier": [1.0, 1.0],
                "feature_a": [0.0, 0.0],
            }
        )

        with TemporaryDirectory() as tmpdir:
            out = predict_expected_points(
                X_pred,
                StubClassifier(),
                StubRegressor(),
                ModelState(path=Path(tmpdir) / "state.json"),
                appearance_clf=StubAppearanceClassifier(),
            )

        self.assertGreater(float(out.loc[0, "p_start"]), 0.6)
        self.assertAlmostEqual(float(out.loc[1, "p_start"]), 0.1)
        self.assertGreater(float(out.loc[0, "appearance_prior_weight"]), 0.0)
        self.assertEqual(float(out.loc[1, "appearance_prior_weight"]), 0.0)
        self.assertGreater(float(out.loc[0, "start_prior_weight"]), 0.0)
        self.assertEqual(float(out.loc[1, "start_prior_weight"]), 0.0)
        self.assertAlmostEqual(float(out.loc[0, "model_p_start"]), 0.1)

    def test_preseason_role_signal_fades_with_current_season_matches(self) -> None:
        class StubClassifier:
            def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
                assert "current_season_match_count" not in feats.columns
                return np.array([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]])

        class StubAppearanceClassifier:
            def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
                return np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

        class StubRegressor:
            def predict(self, feats: pd.DataFrame) -> np.ndarray:
                return np.array([5.0, 5.0, 5.0])

        X_pred = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "full_name": ["GW1", "Five Matches", "Ten Matches"],
                "team_name": ["Alpha", "Beta", "Gamma"],
                "now_cost_millions": [6.0, 8.0, 7.0],
                "team_id": [1, 2, 3],
                "element_type": [3, 3, 3],
                "history_match_count": [20, 20, 20],
                "current_season_match_count": [0, 5, 10],
                "role_change_score": [1.0, 1.0, 1.0],
                "role_evidence_quality": [1.0, 1.0, 1.0],
                "role_target_gameweek": [1, 1, 1],
                "feature_a": [0.0, 0.0, 0.0],
            }
        )

        with TemporaryDirectory() as tmpdir:
            out = predict_expected_points(
                X_pred,
                StubClassifier(),
                StubRegressor(),
                ModelState(path=Path(tmpdir) / "state.json"),
                appearance_clf=StubAppearanceClassifier(),
            )

        self.assertAlmostEqual(float(out.loc[0, "pre_role_points_hat"]), 5.0)
        self.assertAlmostEqual(float(out.loc[0, "points_hat"]), 5.5)
        self.assertAlmostEqual(float(out.loc[0, "role_ep_multiplier"]), 1.1)
        self.assertAlmostEqual(float(out.loc[0, "role_adjustment_weight"]), 1.0)
        self.assertAlmostEqual(float(out.loc[1, "points_hat"]), 5.25)
        self.assertAlmostEqual(float(out.loc[1, "role_ep_multiplier"]), 1.05)
        self.assertAlmostEqual(float(out.loc[1, "role_adjustment_weight"]), 0.5)
        self.assertAlmostEqual(float(out.loc[2, "points_hat"]), 5.0)
        self.assertAlmostEqual(float(out.loc[2, "role_ep_multiplier"]), 1.0)
        self.assertEqual(out["reliability_weight"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(out["p_start"].tolist(), [0.9, 0.9, 0.9])
        self.assertEqual(out["p_appearance"].tolist(), [1.0, 1.0, 1.0])

    def test_model_selection_uses_distinct_appearance_and_conditional_points_targets(self) -> None:
        candidate = ModelCandidate(
            name="stub",
            display_name="Stub",
            build_classifier=lambda: object(),
            build_regressor=lambda: object(),
        )
        calls: list[tuple[str, int, list[float]]] = []

        def fake_tune(build, params, X, y, **kwargs):
            calls.append((kwargs["label"], len(X), list(np.asarray(y, dtype=float))))
            return 0.5, 0.0, {}

        with TemporaryDirectory() as tmpdir:
            with (
                patch("fplmodel.model._tune_and_score", side_effect=fake_tune),
                patch("fplmodel.model.MODELS_DIR", Path(tmpdir)),
            ):
                result = _evaluate_model_candidates(
                    [candidate],
                    pd.DataFrame({"feature": [1, 2, 3, 4]}),
                    np.array([0, 1, 0, 1]),
                    np.array([1, 1, 0, 1]),
                    None,
                    None,
                    pd.DataFrame({"feature": [2, 4]}),
                    pd.Series([5.0, 7.0]),
                    None,
                    None,
                )

        by_label = {label: (size, target) for label, size, target in calls}
        self.assertEqual(by_label["classifier[stub]"][1], [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(
            by_label["appearance_classifier[stub]"][1],
            [1.0, 1.0, 0.0, 1.0],
        )
        self.assertEqual(by_label["regressor[stub]"], (2, [5.0, 7.0]))
        self.assertEqual(result[0]["appearance_balanced_accuracy"], 0.5)

    def test_train_models_returns_and_persists_three_selected_models(self) -> None:
        rows = 80
        minutes = np.tile([0, 30, 90, 90], rows // 4)
        X = pd.DataFrame(
            {
                "feature_a": np.linspace(0.0, 1.0, rows),
                "minutes_lag1": np.roll(minutes, 1),
            }
        )
        y = pd.Series(np.where(minutes >= 60, 5.0, np.where(minutes > 0, 1.0, 0.0)))
        metadata = pd.DataFrame(
            {
                "season_name": ["2025-26"] * rows,
                "round": np.arange(1, rows + 1),
                "minutes": minutes,
                "element_type": np.tile([1, 2, 3, 4], rows // 4),
            }
        )
        candidate = model_module._build_model_candidates()[0]

        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            with (
                patch("fplmodel.model._build_model_candidates", return_value=[candidate]),
                patch("fplmodel.model.MODELS_DIR", model_dir),
                patch("fplmodel.model.CLF_PATH", model_dir / "classifier.joblib"),
                patch(
                    "fplmodel.model.APPEARANCE_CLF_PATH",
                    model_dir / "appearance_classifier.joblib",
                ),
                patch("fplmodel.model.REG_PATH", model_dir / "regressor.joblib"),
                patch(
                    "fplmodel.model.CAMEO_POINTS_PATH",
                    model_dir / "cameo_points.joblib",
                ),
                patch(
                    "fplmodel.model.ENSEMBLE_PATH",
                    model_dir / "ensemble_models.joblib",
                ),
            ):
                start_clf, appearance_clf, reg, cameo_points, bundles = (
                    model_module.train_models(X, y, metadata)
                )

            self.assertTrue((model_dir / "classifier.joblib").exists())
            self.assertTrue((model_dir / "appearance_classifier.joblib").exists())
            self.assertTrue((model_dir / "regressor.joblib").exists())
            self.assertTrue((model_dir / "cameo_points.joblib").exists())
            self.assertTrue((model_dir / "ensemble_models.joblib").exists())

        self.assertEqual(len(bundles), 1)
        self.assertEqual(set(cameo_points), {1, 2, 3, 4})
        self.assertEqual(start_clf.predict_proba(X.iloc[:2]).shape, (2, 2))
        self.assertEqual(appearance_clf.predict_proba(X.iloc[:2]).shape, (2, 2))
        self.assertEqual(reg.predict(X.iloc[:2]).shape, (2,))

    def test_combine_ensemble_expected_points_uses_corrected_model_mean(self) -> None:
        predictions = pd.DataFrame(
            {
                "player_id": [1],
                "expected_points_raw__a": [0.0],
                "expected_points_raw__b": [0.2],
                "expected_points__a": [0.1],
                "expected_points__b": [0.3],
            }
        )

        out = _combine_ensemble_expected_points(
            predictions,
            per_model_raw_cols=["expected_points_raw__a", "expected_points_raw__b"],
            per_model_corrected_cols=["expected_points__a", "expected_points__b"],
        )

        self.assertAlmostEqual(out.loc[0, "expected_points_raw"], 0.1, places=6)
        self.assertAlmostEqual(out.loc[0, "expected_points"], 0.2, places=6)

    def test_live_official_ep_is_a_bounded_next_gameweek_ensemble_input(self) -> None:
        predictions = pd.DataFrame(
            {
                "player_id": [1, 2],
                "expected_points": [8.0, 4.0],
                "official_ep_next": [4.0, np.nan],
                "reliability_weight": [1.0, 0.0],
            }
        )

        out = blend_official_expected_points(predictions, weight=0.35)

        self.assertAlmostEqual(out.loc[0, "expected_points"], 6.6)
        self.assertAlmostEqual(out.loc[0, "model_expected_points"], 8.0)
        self.assertAlmostEqual(out.loc[0, "official_ep_blend_weight"], 0.35)
        self.assertAlmostEqual(out.loc[1, "expected_points"], 4.0)
        self.assertAlmostEqual(out.loc[1, "official_ep_blend_weight"], 0.0)

    def test_official_ep_blend_increases_for_cold_starts(self) -> None:
        predictions = pd.DataFrame(
            {
                "expected_points": [1.0, 1.0],
                "official_ep_next": [5.0, 5.0],
                "reliability_weight": [0.0, 1.0],
            }
        )

        out = blend_official_expected_points(
            predictions,
            weight=0.35,
            cold_start_weight=0.85,
        )

        self.assertAlmostEqual(float(out.loc[0, "official_ep_blend_weight"]), 0.85)
        self.assertAlmostEqual(float(out.loc[1, "official_ep_blend_weight"]), 0.35)
        self.assertGreater(float(out.loc[0, "expected_points"]), float(out.loc[1, "expected_points"]))

    def test_optional_provider_ep_snapshot_is_a_bounded_ensemble_input(self) -> None:
        predictions = pd.DataFrame(
            {
                "expected_points": [4.0, 4.0],
                "provider_expected_points": [6.0, np.nan],
            }
        )

        out = blend_provider_expected_points(predictions, weight=0.2)

        self.assertAlmostEqual(float(out.loc[0, "expected_points"]), 4.4)
        self.assertAlmostEqual(float(out.loc[0, "provider_ep_blend_weight"]), 0.2)
        self.assertAlmostEqual(float(out.loc[1, "expected_points"]), 4.0)
        self.assertEqual(float(out.loc[1, "provider_ep_blend_weight"]), 0.0)

    def test_player_history_file_listing_ignores_stale_player_caches(self) -> None:
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            for filename in [
                "player_1.json",
                "player_2.json",
                "player_999.json",
                "player_invalid.json",
                "bootstrap-static.json",
            ]:
                (raw_dir / filename).write_text("{}", encoding="utf-8")

            files = list_current_player_history_files(raw_dir, [2, 1])

        self.assertEqual(files, ["player_1.json", "player_2.json"])

    def test_external_history_is_mapped_by_stable_player_code(self) -> None:
        external = pd.DataFrame(
            {
                "player_id": [777, 888],
                "player_code": [1001, 9999],
                "season_name": ["2025-26", "2025-26"],
                "element_type": [4, 2],
            }
        )

        remapped = remap_external_histories_to_current_players(external, _elements_df())

        self.assertEqual(len(remapped), 2)
        mapped = remapped[remapped["player_code"].eq(1001)].iloc[0]
        historical_only = remapped[remapped["player_code"].eq(9999)].iloc[0]
        self.assertEqual(int(mapped["player_id"]), 1)
        self.assertEqual(int(mapped["element_type"]), 4)
        self.assertEqual(int(mapped["current_element_type"]), 3)
        self.assertTrue(bool(mapped["position_changed"]))
        self.assertTrue(bool(mapped["is_current_player_mapped"]))
        self.assertLess(int(historical_only["player_id"]), 0)
        self.assertFalse(bool(historical_only["is_current_player_mapped"]))
        self.assertEqual(int(historical_only["historical_player_id"]), 888)

    def test_season_validation_rejects_stale_api_data(self) -> None:
        events = pd.DataFrame({"deadline_time": ["2025-08-15T17:30:00Z"]})
        actual = infer_season_name(events)

        with self.assertRaisesRegex(RuntimeError, "2026-27"):
            validate_season_name(actual, expected_season_name="2026-27")

    def test_snapshot_cutoff_fails_closed_when_target_deadline_is_missing(self) -> None:
        events = pd.DataFrame(
            {
                "event_id": [1],
                "deadline_time": ["2026-08-21T17:30:00Z"],
            }
        )

        with self.assertRaisesRegex(ValueError, "GW2 deadline"):
            target_gameweek_deadline(events, 2)

    def test_preseason_history_validation_requires_all_gameweeks(self) -> None:
        external = pd.DataFrame(
            {
                "season_name": ["2025-26"] * 37,
                "round": list(range(1, 38)),
            }
        )

        with self.assertRaisesRegex(RuntimeError, "GW38"):
            validate_external_history_coverage(external, "2025-26")


if __name__ == "__main__":
    unittest.main()
