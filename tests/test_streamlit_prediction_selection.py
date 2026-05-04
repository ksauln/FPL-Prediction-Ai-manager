from __future__ import annotations

import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from streamlit_app import _load_predictions_for_horizon, _select_prediction_gameweek


class StreamlitPredictionSelectionTests(unittest.TestCase):
    def test_prefers_official_next_gameweek_over_first_unfinished_output(self) -> None:
        events = pd.DataFrame(
            [
                {"id": 34, "finished": True, "is_current": False, "is_next": False},
                {"id": 35, "finished": False, "is_current": True, "is_next": False},
                {"id": 36, "finished": False, "is_current": False, "is_next": True},
            ]
        )

        selected = _select_prediction_gameweek([33, 34, 35, 36, 37], events)

        self.assertEqual(selected, 36)

    def test_uses_first_future_output_when_official_next_is_missing(self) -> None:
        events = pd.DataFrame(
            [
                {"id": 34, "finished": True, "is_current": False, "is_next": False},
                {"id": 35, "finished": False, "is_current": True, "is_next": False},
                {"id": 36, "finished": False, "is_current": False, "is_next": True},
            ]
        )

        selected = _select_prediction_gameweek([33, 34, 35], events)

        self.assertEqual(selected, 35)

    def test_falls_back_to_latest_output_without_event_state(self) -> None:
        selected = _select_prediction_gameweek([33, 34, 35, 36], None)

        self.assertEqual(selected, 36)

    def test_horizon_loader_starts_from_official_next_gameweek(self) -> None:
        events = pd.DataFrame(
            [
                {"id": 34, "finished": True, "is_current": False, "is_next": False},
                {"id": 35, "finished": False, "is_current": True, "is_next": False},
                {"id": 36, "finished": False, "is_current": False, "is_next": True},
            ]
        )

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            for gw in (35, 36, 37):
                pd.DataFrame(
                    {
                        "player_id": [1],
                        "full_name": ["Player One"],
                        "team_name": ["Alpha"],
                        "team_id": [1],
                        "element_type": [3],
                        "now_cost_millions": [7.5],
                        "expected_points": [float(gw)],
                    }
                ).to_csv(output_dir / f"predictions_gw{gw}.csv", index=False)

            with (
                patch("streamlit_app.OUTPUTS_DIR", output_dir),
                patch("streamlit_app._load_bootstrap_events", return_value=events),
            ):
                loaded_gws, predictions_by_gw, missing = _load_predictions_for_horizon(2)

        self.assertEqual(loaded_gws, [36, 37])
        self.assertEqual(missing, [])
        self.assertEqual(predictions_by_gw[36].loc[0, "expected_points"], 36.0)


if __name__ == "__main__":
    unittest.main()
