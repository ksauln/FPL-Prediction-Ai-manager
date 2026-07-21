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
    build_training_and_pred_frames,
    expand_for_double_gw,
)
import fplmodel.model as model_module
from fplmodel.model import ModelCandidate, _evaluate_model_candidates, predict_expected_points
from fplmodel.state import ModelState
from main import (
    _combine_ensemble_expected_points,
    add_prediction_confidence,
    infer_season_name,
    list_current_player_history_files,
    remap_external_histories_to_current_players,
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
            ):
                start_clf, appearance_clf, reg, cameo_points, bundles = (
                    model_module.train_models(X, y, metadata)
                )

            self.assertTrue((model_dir / "classifier.joblib").exists())
            self.assertTrue((model_dir / "appearance_classifier.joblib").exists())
            self.assertTrue((model_dir / "regressor.joblib").exists())
            self.assertTrue((model_dir / "cameo_points.joblib").exists())

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
            }
        )

        remapped = remap_external_histories_to_current_players(external, _elements_df())

        self.assertEqual(remapped["player_id"].tolist(), [1])
        self.assertEqual(remapped["player_code"].tolist(), [1001])

    def test_season_validation_rejects_stale_api_data(self) -> None:
        events = pd.DataFrame({"deadline_time": ["2025-08-15T17:30:00Z"]})
        actual = infer_season_name(events)

        with self.assertRaisesRegex(RuntimeError, "2026-27"):
            validate_season_name(actual, expected_season_name="2026-27")

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
