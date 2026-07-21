from __future__ import annotations

import unittest

import pandas as pd

try:
    import pulp  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment guard
    raise unittest.SkipTest("pulp is required for transfer optimizer tests") from exc

from fplmodel.team_picker import pick_best_xi


class TeamPickerTransferTests(unittest.TestCase):
    def test_transfer_optimizer_can_fund_two_player_package(self) -> None:
        rows = [
            (101, 1, 1, 4.5, 4.0),
            (102, 2, 1, 4.0, 3.0),
            (201, 1, 2, 5.0, 5.0),
            (202, 2, 2, 5.0, 4.8),
            (203, 3, 2, 4.5, 4.4),
            (204, 4, 2, 4.5, 4.0),
            (205, 5, 2, 4.0, 3.6),
            (301, 1, 3, 10.0, 1.0),
            (302, 2, 3, 8.0, 7.0),
            (303, 3, 3, 7.0, 6.0),
            (304, 4, 3, 6.0, 5.0),
            (305, 5, 3, 4.5, 1.0),
            (306, 6, 3, 5.5, 9.0),
            (307, 7, 3, 9.0, 10.0),
            (401, 1, 4, 8.0, 7.0),
            (402, 2, 4, 7.0, 6.0),
            (403, 3, 4, 6.0, 5.0),
        ]
        predictions = pd.DataFrame(
            [
                {
                    "player_id": player_id,
                    "full_name": f"Player {player_id}",
                    "team_id": team_id,
                    "team_name": f"Team {team_id}",
                    "element_type": element_type,
                    "now_cost_millions": cost,
                    "expected_points": points,
                }
                for player_id, team_id, element_type, cost, points in rows
            ]
        )
        current_ids = {
            101,
            102,
            201,
            202,
            203,
            204,
            205,
            301,
            302,
            303,
            304,
            305,
            401,
            402,
            403,
        }
        sale_values = {
            player_id: float(
                predictions.loc[predictions["player_id"] == player_id, "now_cost_millions"].iloc[0]
            )
            for player_id in current_ids
        }

        result = pick_best_xi(
            predictions,
            formations=[{"GK": 1, "DEF": 3, "MID": 5, "FWD": 2}],
            current_player_ids=current_ids,
            bank_m=0.0,
            sale_value_by_player_id=sale_values,
            max_transfers=2,
            free_transfers=2,
        )

        selected_ids = {
            int(player["player_id"])
            for player in result["squad"] + result["bench"]
        }
        self.assertIn(306, selected_ids)
        self.assertIn(307, selected_ids)
        self.assertEqual(result["transfers_made"], 2)


if __name__ == "__main__":
    unittest.main()
