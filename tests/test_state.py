from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fplmodel.state import ModelState


class ModelStateTests(unittest.TestCase):
    def test_state_resets_player_ids_when_season_changes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            previous = ModelState(path=path, season_name="2025-26")
            previous.player_bias["1"] = 4.5
            previous.position_bias["3"] = 1.2
            previous.last_evaluated_gw = 38
            previous.save()

            current = ModelState(path=path, season_name="2026-27")

        self.assertEqual(current.season_name, "2026-27")
        self.assertEqual(current.player_bias, {})
        self.assertEqual(current.position_bias, {})
        self.assertEqual(current.last_evaluated_gw, 0)


if __name__ == "__main__":
    unittest.main()
