from __future__ import annotations

from copy import deepcopy
import unittest

from fplmodel.live_rules import (
    season_rules_from_bootstrap,
    validate_live_fpl_configuration,
)


def _live_bootstrap() -> dict[str, object]:
    return {
        "game_config": {
            "rules": {
                "squad_squadplay": 11,
                "squad_squadsize": 15,
                "squad_team_limit": 3,
                "squad_total_spend": 1000,
                "ui_currency_multiplier": 10,
                "transfers_cap": 20,
                "max_extra_free_transfers": 4,
            },
            "scoring": {
                "long_play": 2,
                "short_play": 1,
                "goals_conceded": {"DEF": -1, "FWD": 0, "GKP": -1, "MID": 0},
                "saves": 1,
                "goals_scored": {"DEF": 6, "FWD": 4, "GKP": 10, "MID": 5},
                "assists": 3,
                "clean_sheets": {"DEF": 4, "FWD": 0, "GKP": 4, "MID": 1},
                "penalties_saved": 5,
                "penalties_missed": -2,
                "yellow_cards": -1,
                "red_cards": -3,
                "own_goals": -2,
                "bonus": 1,
                "defensive_contribution": {
                    "DEF": 2,
                    "FWD": 2,
                    "GKP": 0,
                    "MID": 2,
                },
            },
        },
        "element_types": [
            {
                "id": 1,
                "singular_name_short": "GKP",
                "squad_select": 2,
                "squad_min_play": 1,
                "squad_max_play": 1,
            },
            {
                "id": 2,
                "singular_name_short": "DEF",
                "squad_select": 5,
                "squad_min_play": 3,
                "squad_max_play": 5,
            },
            {
                "id": 3,
                "singular_name_short": "MID",
                "squad_select": 5,
                "squad_min_play": 2,
                "squad_max_play": 5,
            },
            {
                "id": 4,
                "singular_name_short": "FWD",
                "squad_select": 3,
                "squad_min_play": 1,
                "squad_max_play": 3,
            },
        ],
        "chips": [
            {"name": "wildcard", "number": 1, "start_event": 2, "stop_event": 19},
            {"name": "wildcard", "number": 1, "start_event": 20, "stop_event": 38},
            {"name": "freehit", "number": 1, "start_event": 2, "stop_event": 19},
            {"name": "freehit", "number": 1, "start_event": 20, "stop_event": 38},
            {"name": "bboost", "number": 1, "start_event": 1, "stop_event": 19},
            {"name": "bboost", "number": 1, "start_event": 20, "stop_event": 38},
            {"name": "3xc", "number": 1, "start_event": 1, "stop_event": 19},
            {"name": "3xc", "number": 1, "start_event": 20, "stop_event": 38},
        ],
    }


class LiveRulesTests(unittest.TestCase):
    def test_live_2026_configuration_matches_manager_constraints(self) -> None:
        live = validate_live_fpl_configuration(_live_bootstrap())
        rules = season_rules_from_bootstrap(_live_bootstrap())

        self.assertEqual(live.squad_size, 15)
        self.assertEqual(live.max_per_team, 3)
        self.assertEqual(live.position_limits, {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3})
        self.assertEqual(rules.budget_m, 100.0)
        self.assertEqual(rules.max_free_transfers, 5)
        self.assertEqual(rules.first_half_end_gw, 19)
        self.assertEqual(rules.free_transfer_topups, {})
        self.assertEqual(rules.chips_by_half["first"]["free_hit"], 1)
        self.assertEqual(rules.chips_by_half["second"]["triple_captain"], 1)

    def test_live_rule_validation_stops_on_transfer_rule_drift(self) -> None:
        bootstrap = deepcopy(_live_bootstrap())
        bootstrap["game_config"]["rules"]["max_extra_free_transfers"] = 5

        with self.assertRaisesRegex(RuntimeError, "max free transfers"):
            validate_live_fpl_configuration(bootstrap)

    def test_live_rule_validation_stops_on_scoring_rule_drift(self) -> None:
        bootstrap = deepcopy(_live_bootstrap())
        bootstrap["game_config"]["scoring"]["goals_scored"]["MID"] = 4

        with self.assertRaisesRegex(RuntimeError, "goals_scored"):
            validate_live_fpl_configuration(bootstrap)

    def test_live_rule_validation_stops_on_position_limit_drift(self) -> None:
        bootstrap = deepcopy(_live_bootstrap())
        bootstrap["element_types"][2]["squad_select"] = 6

        with self.assertRaisesRegex(RuntimeError, "position limits"):
            validate_live_fpl_configuration(bootstrap)


if __name__ == "__main__":
    unittest.main()
