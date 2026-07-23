from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from fplmodel.evaluation import evaluate_last_finished_gw_and_update_state
from fplmodel.state import ModelState


class _Classifier:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([[0.0, 1.0]]), (len(features), 1))


class _Regressor:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), 4.0)


class EvaluationTests(unittest.TestCase):
    def test_residual_evaluation_uses_only_the_current_season(self) -> None:
        histories = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "element_type": 3,
                    "season_name": "2025-26",
                    "round": 1,
                    "total_points": 1,
                },
                {
                    "player_id": 1,
                    "element_type": 3,
                    "season_name": "2026-27",
                    "round": 1,
                    "total_points": 7,
                },
            ]
        )
        features = pd.DataFrame(
            [{"player_id": 1, "element_type": 3, "feature": 1.0}]
        )

        with TemporaryDirectory() as tmpdir:
            state = ModelState(
                path=Path(tmpdir) / "state.json",
                season_name="2026-27",
            )
            residuals = evaluate_last_finished_gw_and_update_state(
                _Classifier(),
                _Classifier(),
                _Regressor(),
                {3: 1.0},
                features,
                histories,
                1,
                state,
                current_season_name="2026-27",
            )

        self.assertEqual(len(residuals), 1)
        self.assertAlmostEqual(float(residuals.iloc[0]["residual"]), 3.0)
        self.assertEqual(state.last_evaluated_gw, 1)

    def test_future_replay_cannot_update_state_from_prior_season_rows(self) -> None:
        histories = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "element_type": 3,
                    "season_name": "2025-26",
                    "round": 2,
                    "total_points": 9,
                }
            ]
        )
        features = pd.DataFrame(
            [{"player_id": 1, "element_type": 3, "feature": 1.0}]
        )

        with TemporaryDirectory() as tmpdir:
            state = ModelState(
                path=Path(tmpdir) / "state.json",
                season_name="2026-27",
            )
            residuals = evaluate_last_finished_gw_and_update_state(
                _Classifier(),
                _Classifier(),
                _Regressor(),
                {3: 1.0},
                features,
                histories,
                2,
                state,
                current_season_name="2026-27",
            )

        self.assertTrue(residuals.empty)
        self.assertEqual(state.last_evaluated_gw, 0)
        self.assertEqual(state.player_bias, {})


if __name__ == "__main__":
    unittest.main()
