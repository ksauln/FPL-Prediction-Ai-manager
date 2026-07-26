from __future__ import annotations

import unittest
from unittest.mock import Mock

from fplmodel.dashboard_updates import (
    build_update_control_state,
    refresh_dashboard_data,
)


class DashboardUpdateTests(unittest.TestCase):
    def test_update_control_uses_absolute_target_when_next_gameweek_is_known(self) -> None:
        state = build_update_control_state(8)

        self.assertEqual(state["mode"], "through_gameweek")
        self.assertEqual(state["options"], list(range(8, 39)))
        self.assertEqual(state["default"], 11)

    def test_update_control_uses_relative_horizon_when_metadata_is_unavailable(self) -> None:
        state = build_update_control_state(None)

        self.assertEqual(state["mode"], "horizon")
        self.assertEqual(state["options"], list(range(1, 7)))
        self.assertEqual(state["default"], 4)

    def test_update_control_disables_refresh_after_gameweek_38(self) -> None:
        state = build_update_control_state(39)

        self.assertEqual(state["mode"], "season_complete")
        self.assertEqual(state["options"], [])
        self.assertIsNone(state["default"])

    def test_builds_predictions_through_selected_gameweek(self) -> None:
        run_pipeline = Mock(
            return_value={"season_name": "2026-27", "next_gw": 8}
        )
        replay_gameweeks = Mock(
            return_value=[
                {"season_name": "2026-27", "next_gw": gameweek}
                for gameweek in (9, 10, 11, 12)
            ]
        )

        result = refresh_dashboard_data(
            through_gameweek=12,
            run_pipeline_fn=run_pipeline,
            replay_gameweeks_fn=replay_gameweeks,
        )

        run_pipeline.assert_called_once_with(force_refetch=True)
        replay_gameweeks.assert_called_once_with(
            start_gw=9,
            end_gw=12,
            force_refetch=False,
            expected_season_name="2026-27",
        )
        self.assertEqual(result["start_gameweek"], 8)
        self.assertEqual(result["end_gameweek"], 12)
        self.assertEqual(result["gameweeks"], [8, 9, 10, 11, 12])

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

    def test_rejects_selected_gameweek_outside_the_season(self) -> None:
        run_pipeline = Mock()

        with self.assertRaisesRegex(ValueError, "through_gameweek"):
            refresh_dashboard_data(
                through_gameweek=39,
                run_pipeline_fn=run_pipeline,
                replay_gameweeks_fn=Mock(),
            )

        run_pipeline.assert_not_called()

    def test_rejects_selected_gameweek_before_live_next_gameweek(self) -> None:
        run_pipeline = Mock(
            return_value={"season_name": "2026-27", "next_gw": 8}
        )
        replay_gameweeks = Mock()

        with self.assertRaisesRegex(
            ValueError,
            "through_gameweek must be GW8 or later",
        ):
            refresh_dashboard_data(
                through_gameweek=7,
                run_pipeline_fn=run_pipeline,
                replay_gameweeks_fn=replay_gameweeks,
            )

        replay_gameweeks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
