from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from fplmodel.evaluation import (
    evaluate_last_finished_gw_and_update_state,
    prequential_training_mask,
)
from fplmodel.state import ModelState


class _Classifier:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([[0.0, 1.0]]), (len(features), 1))


class _Regressor:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), 4.0)


class EvaluationTests(unittest.TestCase):
    def test_prequential_training_mask_excludes_the_evaluated_gameweek(self) -> None:
        metadata = pd.DataFrame(
            {
                "season_name": ["2025-26", "2026-27", "2026-27", "2026-27"],
                "round": [38, 3, 4, 5],
            }
        )

        mask = prequential_training_mask(
            metadata,
            evaluation_gw=4,
            current_season_name="2026-27",
        )

        self.assertEqual(mask.tolist(), [True, True, False, False])

    def test_rich_training_frame_evaluates_the_shifted_target_row(self) -> None:
        class FeatureRegressor:
            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return features["feature"].to_numpy(dtype=float)

        training_rows = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "element_type": 3,
                    "season_name": "2026-27",
                    "round": 4,
                    "fixture": 40,
                    "total_points": 7.0,
                    "feature": 2.0,
                },
                {
                    "player_id": 1,
                    "element_type": 3,
                    "season_name": "2026-27",
                    "round": 5,
                    "fixture": 50,
                    "total_points": 9.0,
                    "feature": 99.0,
                },
            ]
        )

        with TemporaryDirectory() as tmpdir:
            state = ModelState(path=Path(tmpdir) / "state.json", season_name="2026-27")
            residuals = evaluate_last_finished_gw_and_update_state(
                _Classifier(),
                _Classifier(),
                FeatureRegressor(),
                {3: 1.0},
                training_rows,
                pd.DataFrame(),
                4,
                state,
                current_season_name="2026-27",
            )

        self.assertEqual(len(residuals), 1)
        self.assertAlmostEqual(float(residuals.iloc[0]["residual"]), 5.0)
        self.assertEqual(int(residuals.iloc[0]["fixture"]), 40)

    def test_rich_training_frame_keeps_double_gameweek_fixtures_distinct(self) -> None:
        training_rows = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "element_type": 3,
                    "season_name": "2026-27",
                    "round": 6,
                    "fixture": fixture,
                    "total_points": points,
                    "feature": 1.0,
                }
                for fixture, points in ((60, 3.0), (61, 5.0))
            ]
        )

        with TemporaryDirectory() as tmpdir:
            state = ModelState(path=Path(tmpdir) / "state.json", season_name="2026-27")
            residuals = evaluate_last_finished_gw_and_update_state(
                _Classifier(),
                _Classifier(),
                _Regressor(),
                {3: 1.0},
                training_rows,
                pd.DataFrame(),
                6,
                state,
                current_season_name="2026-27",
            )

        self.assertEqual(set(residuals["fixture"]), {60, 61})
        self.assertEqual(len(residuals), 2)

    def test_already_evaluated_gameweek_is_not_applied_twice(self) -> None:
        histories = pd.DataFrame(
            [{"player_id": 1, "element_type": 3, "season_name": "2026-27", "round": 1, "total_points": 7}]
        )
        features = pd.DataFrame([{"player_id": 1, "element_type": 3, "feature": 1.0}])

        with TemporaryDirectory() as tmpdir:
            state = ModelState(path=Path(tmpdir) / "state.json", season_name="2026-27")
            state.last_evaluated_gw = 1
            residuals = evaluate_last_finished_gw_and_update_state(
                _Classifier(), _Classifier(), _Regressor(), {3: 1.0}, features,
                histories, 1, state, current_season_name="2026-27",
            )

        self.assertTrue(residuals.empty)
        self.assertEqual(state.player_bias, {})

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
