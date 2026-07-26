from __future__ import annotations

import unittest
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pandas as pd
import pytest

pytest.importorskip("jinja2", minversion="3.0.0")
pytest.importorskip("streamlit")

from fplmodel.manager_dashboard import (
    _build_fixture_card_context,
    _decision_frame,
    _default_season_name,
    _format_duration,
    _parse_topups,
    _select_season_source,
    _team_pitch_html,
)
from fplmodel.prediction_artifacts import PredictionSeasonSource


class ManagerDashboardTests(unittest.TestCase):
    def test_fixture_card_context_includes_opponent_venue_and_difficulty(self) -> None:
        fixtures = pd.DataFrame(
            [
                {
                    "event": 4,
                    "team_h": 1,
                    "team_a": 2,
                    "team_h_difficulty": 2,
                    "team_a_difficulty": 4,
                },
                {
                    "event": 4,
                    "team_h": 3,
                    "team_a": 1,
                    "team_h_difficulty": 3,
                    "team_a_difficulty": 5,
                },
            ]
        )
        teams = pd.DataFrame(
            [
                {"id": 1, "short_name": "AAA"},
                {"id": 2, "short_name": "BBB"},
                {"id": 3, "short_name": "CCC"},
                {"id": 4, "short_name": "DDD"},
            ]
        )

        context = _build_fixture_card_context(fixtures, teams, gameweek=4)

        self.assertEqual(
            context[1],
            {
                "opponent": "BBB / CCC",
                "venue": "H / A",
                "difficulty": "2 / 5",
            },
        )
        self.assertEqual(
            context[2],
            {"opponent": "AAA", "venue": "A", "difficulty": "4"},
        )
        self.assertEqual(
            context[4],
            {"opponent": "No fixture", "venue": "—", "difficulty": "—"},
        )

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

    def test_team_pitch_html_renders_formation_roles_and_escaped_names(self) -> None:
        starters = []
        player_id = 1
        for element_type, count in ((1, 1), (2, 4), (3, 4), (4, 2)):
            for _ in range(count):
                starters.append(
                    {
                        "player_id": player_id,
                        "element_type": element_type,
                        "full_name": (
                            "<script>Captain</script>"
                            if player_id == 1
                            else f"Player {player_id}"
                        ),
                        "team_name": f"Club {player_id}",
                        "team_id": player_id,
                        "expected_points": 5.0 + player_id / 10,
                        "now_cost_millions": 5.0 + player_id / 10,
                    }
                )
                player_id += 1
        bench = [
            {
                "player_id": player_id + index,
                "element_type": element_type,
                "full_name": f"Bench {index + 1}",
                "team_name": f"Club {player_id + index}",
                "team_id": player_id + index,
                "expected_points": 2.0,
                "now_cost_millions": 4.5,
                "bench_order": index + 1,
            }
            for index, element_type in enumerate((1, 2, 3, 4))
        ]
        decision = {
            "gameweek": 1,
            "expected_points": 71.4,
            "captain_id": 1,
            "vice_captain_id": 2,
            "chip": "bench_boost",
            "free_transfers_after": 3,
            "team": {"squad": starters, "bench": bench},
        }

        rendered = _team_pitch_html(
            decision,
            fixture_context_by_team={
                1: {
                    "opponent": "Rivals <FC>",
                    "venue": "H",
                    "difficulty": "2",
                },
                2: {
                    "opponent": "United",
                    "venue": "A",
                    "difficulty": "4",
                },
            },
        )

        self.assertIn("GW1", rendered)
        self.assertIn("4-4-2", rendered)
        self.assertIn("Bench Boost", rendered)
        self.assertIn("71.4", rendered)
        self.assertIn("Free transfers", rendered)
        self.assertIn("<strong>3</strong>", rendered)
        self.assertIn("£5.1m", rendered)
        self.assertIn("£4.5m", rendered)
        self.assertIn("vs Rivals &lt;FC&gt;", rendered)
        self.assertIn('aria-label="Venue Home">H</span>', rendered)
        self.assertIn("FDR 2", rendered)
        self.assertIn('aria-label="Venue Away">A</span>', rendered)
        self.assertIn("FDR 4", rendered)
        self.assertEqual(rendered.count('class="ai-player-card ai-player-card--starter"'), 11)
        self.assertEqual(rendered.count('class="ai-player-card ai-player-card--bench"'), 4)
        self.assertIn('aria-label="Captain"', rendered)
        self.assertIn('aria-label="Vice-captain"', rendered)
        self.assertIn("&lt;script&gt;Captain&lt;/script&gt;", rendered)
        self.assertNotIn("<script>Captain</script>", rendered)


if __name__ == "__main__":
    unittest.main()
