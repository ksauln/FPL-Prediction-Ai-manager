from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fplmodel.features import _add_team_context_features, build_training_and_pred_frames
from fplmodel.state import ModelState
from main import add_prediction_confidence


def _elements_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "full_name": "Player One",
                "team_id": 1,
                "element_type": 3,
                "now_cost_millions": 7.5,
            },
            {
                "player_id": 2,
                "full_name": "Player Two",
                "team_id": 2,
                "element_type": 4,
                "now_cost_millions": 8.0,
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
        self.assertEqual(pred_by_player.loc[1, "total_points_lag1"], 3)
        self.assertEqual(pred_by_player.loc[2, "total_points_lag1"], 7)

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


if __name__ == "__main__":
    unittest.main()
