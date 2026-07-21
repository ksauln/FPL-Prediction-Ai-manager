from __future__ import annotations

import unittest
from unittest.mock import Mock

from fplmodel.dashboard_updates import refresh_dashboard_data


class DashboardUpdateTests(unittest.TestCase):
    def test_refreshes_live_data_then_builds_remaining_four_gameweek_horizon(self) -> None:
        run_pipeline = Mock(
            return_value={"season_name": "2026-27", "next_gw": 8}
        )
        replay_gameweeks = Mock(
            return_value=[
                {"season_name": "2026-27", "next_gw": gameweek}
                for gameweek in (9, 10, 11)
            ]
        )

        result = refresh_dashboard_data(
            horizon=4,
            run_pipeline_fn=run_pipeline,
            replay_gameweeks_fn=replay_gameweeks,
        )

        run_pipeline.assert_called_once_with(force_refetch=True)
        replay_gameweeks.assert_called_once_with(
            start_gw=9,
            end_gw=11,
            force_refetch=False,
            expected_season_name="2026-27",
        )
        self.assertEqual(result["season_name"], "2026-27")
        self.assertEqual(result["gameweeks"], [8, 9, 10, 11])
        self.assertEqual(len(result["runs"]), 4)

    def test_caps_the_prediction_horizon_at_gameweek_38(self) -> None:
        run_pipeline = Mock(
            return_value={"season_name": "2026-27", "next_gw": 37}
        )
        replay_gameweeks = Mock(
            return_value=[{"season_name": "2026-27", "next_gw": 38}]
        )

        result = refresh_dashboard_data(
            horizon=4,
            run_pipeline_fn=run_pipeline,
            replay_gameweeks_fn=replay_gameweeks,
        )

        replay_gameweeks.assert_called_once_with(
            start_gw=38,
            end_gw=38,
            force_refetch=False,
            expected_season_name="2026-27",
        )
        self.assertEqual(result["gameweeks"], [37, 38])

    def test_skips_replay_when_gameweek_38_is_next(self) -> None:
        run_pipeline = Mock(
            return_value={"season_name": "2026-27", "next_gw": 38}
        )
        replay_gameweeks = Mock()

        result = refresh_dashboard_data(
            horizon=4,
            run_pipeline_fn=run_pipeline,
            replay_gameweeks_fn=replay_gameweeks,
        )

        replay_gameweeks.assert_not_called()
        self.assertEqual(result["gameweeks"], [38])

    def test_rejects_non_positive_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "horizon"):
            refresh_dashboard_data(horizon=0)


if __name__ == "__main__":
    unittest.main()
