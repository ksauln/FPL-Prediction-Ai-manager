from __future__ import annotations

import unittest
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pytest

pytest.importorskip("jinja2", minversion="3.0.0")
pytest.importorskip("streamlit")

from fplmodel.manager_dashboard import (
    _decision_frame,
    _default_season_name,
    _format_duration,
    _parse_topups,
    _select_season_source,
)
from fplmodel.prediction_artifacts import PredictionSeasonSource


class ManagerDashboardTests(unittest.TestCase):
    def test_default_season_changes_in_july(self) -> None:
        missing = Path("/path/that/does/not/exist/bootstrap.json")
        self.assertEqual(
            _default_season_name(datetime(2026, 6, 30), bootstrap_path=missing),
            "2025-26",
        )
        self.assertEqual(
            _default_season_name(datetime(2026, 7, 1), bootstrap_path=missing),
            "2026-27",
        )

    def test_default_season_prefers_the_bootstrap_fixture_season(self) -> None:
        with TemporaryDirectory() as tmpdir:
            bootstrap_path = Path(tmpdir) / "bootstrap.json"
            bootstrap_path.write_text(
                json.dumps(
                    {
                        "events": [
                            {"deadline_time": "2025-08-15T17:30:00Z"},
                            {"deadline_time": "2026-05-24T13:30:00Z"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                _default_season_name(
                    datetime(2026, 7, 20), bootstrap_path=bootstrap_path
                ),
                "2025-26",
            )

    def test_duration_and_transfer_topup_formatting(self) -> None:
        self.assertEqual(_format_duration(3661), "1h 1m 1s")
        self.assertEqual(_parse_topups("16:5, 30:3"), {16: 5, 30: 3})
        with self.assertRaisesRegex(ValueError, "GW:count"):
            _parse_topups("GW16 gets five")

    def test_decision_frame_contains_weekly_manager_actions(self) -> None:
        frame = _decision_frame(
            [
                {
                    "gameweek": 4,
                    "expected_points": 61.25,
                    "captain": "Forward One",
                    "vice_captain": "Midfielder One",
                    "chip": "triple_captain",
                    "transfers": [{"out_player": {}, "in_player": {}}],
                    "transfer_hit_cost": 4.0,
                    "free_transfers_after": 1,
                    "bank_m": 0.7,
                    "team_value_m": 101.4,
                }
            ]
        )

        self.assertEqual(frame.loc[0, "GW"], 4)
        self.assertEqual(frame.loc[0, "Chip"], "Triple Captain")
        self.assertEqual(frame.loc[0, "Transfers"], 1)
        self.assertEqual(frame.loc[0, "Captain"], "Forward One")

    def test_season_selector_returns_the_selected_prediction_source(self) -> None:
        current = PredictionSeasonSource(
            "2025-26", Path("current"), Path("current-bootstrap.json"), True
        )
        previous = PredictionSeasonSource(
            "2024-25", Path("previous"), Path("previous-bootstrap.json"), True
        )
        container = Mock()
        container.selectbox.return_value = "2024-25"

        season_name, source = _select_season_source(
            {"2025-26": current, "2024-25": previous},
            default_season="2025-26",
            container=container,
        )

        self.assertEqual(season_name, "2024-25")
        self.assertIs(source, previous)
        self.assertEqual(container.selectbox.call_args.kwargs["index"], 0)


if __name__ == "__main__":
    unittest.main()
