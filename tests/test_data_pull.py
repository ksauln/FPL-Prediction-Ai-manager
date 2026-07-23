from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from fplmodel import data_pull, external_history
from fplmodel.utils import get_current_and_last_finished_gw


class DataPullTests(unittest.TestCase):
    def test_provisional_gameweek_is_not_treated_as_final(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "id": 1,
                    "finished": True,
                    "data_checked": True,
                    "is_next": False,
                },
                {
                    "id": 2,
                    "finished": True,
                    "data_checked": False,
                    "is_next": False,
                },
                {
                    "id": 3,
                    "finished": False,
                    "data_checked": False,
                    "is_next": True,
                },
            ]
        )

        next_gameweek, last_finished = get_current_and_last_finished_gw(events)

        self.assertEqual(next_gameweek, 3)
        self.assertEqual(last_finished, 1)

    def test_player_history_fetches_only_current_season_and_versions_cache(self) -> None:
        payload = {
            "history": [
                {
                    "fixture": 1,
                    "round": 1,
                    "kickoff_time": "2026-08-21T19:00:00Z",
                }
            ]
        }
        with TemporaryDirectory() as tmpdir:
            with (
                patch.object(data_pull, "RAW_DIR", Path(tmpdir)),
                patch.object(data_pull, "_safe_get_json", return_value=payload) as get_json,
            ):
                result = data_pull.fetch_player_history(123, force=True, seasons_back=5)

            cached = json.loads((Path(tmpdir) / "player_123.json").read_text(encoding="utf-8"))

        self.assertEqual(get_json.call_count, 1)
        self.assertEqual(result["_history_seasons"], 1)
        self.assertEqual(cached["_history_cache_schema_version"], data_pull.PLAYER_HISTORY_CACHE_SCHEMA_VERSION)
        self.assertEqual(len({row["season_name"] for row in cached["history"]}), 1)

    def test_old_player_history_cache_schema_is_not_fresh(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "player_1.json"
            path.write_text(
                json.dumps(
                    {
                        "_fetched_ts": data_pull.unix_now(),
                        "_history_seasons": 6,
                        "_history_cache_schema_version": 1,
                    }
                ),
                encoding="utf-8",
            )

            self.assertFalse(data_pull._player_cache_fresh(path, min_seasons=1))

    def test_external_history_exposes_stable_player_code(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            season_dir = root / "2024-25"
            gw_dir = season_dir / "gws"
            gw_dir.mkdir(parents=True)
            pd.DataFrame([{"id": 1, "name": "Alpha"}, {"id": 2, "name": "Beta"}]).to_csv(
                season_dir / "teams.csv", index=False
            )
            pd.DataFrame([{"id": 77, "code": 999001}]).to_csv(
                season_dir / "players_raw.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "id": 1,
                        "event": 1,
                        "team_h": 1,
                        "team_a": 2,
                        "team_h_difficulty": 2,
                        "team_a_difficulty": 4,
                    }
                ]
            ).to_csv(season_dir / "fixtures.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "element": 77,
                        "name": "Historical Player",
                        "position": "DEF",
                        "team": "Alpha",
                        "round": 1,
                        "fixture": 1,
                        "total_points": 6,
                        "minutes": 90,
                        "was_home": True,
                        "xP": 3.7,
                    }
                ]
            ).to_csv(gw_dir / "gw1.csv", index=False)

            with patch.object(external_history, "EXTERNAL_HISTORY_DIR", root):
                result = external_history.load_external_histories(["2024-25"])

        self.assertEqual(result.loc[0, "player_code"], 999001)
        self.assertEqual(result.loc[0, "element_type"], 2)
        self.assertAlmostEqual(float(result.loc[0, "official_expected_points"]), 3.7)
        self.assertEqual(float(result.loc[0, "fixture_difficulty"]), 2.0)

    def test_bootstrap_metadata_refetches_when_cache_is_stale(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bootstrap-static.json"
            path.write_text(json.dumps({"events": [{"id": 1}]}), encoding="utf-8")
            stale_time = time.time() - 3600
            os.utime(path, (stale_time, stale_time))
            fresh_payload = {"events": [{"id": 2}]}
            with (
                patch.object(data_pull, "RAW_DIR", Path(tmpdir)),
                patch.object(data_pull, "LIVE_METADATA_CACHE_TTL_SECONDS", 60),
                patch.object(data_pull, "_safe_get_json", return_value=fresh_payload) as get_json,
            ):
                result = data_pull.fetch_bootstrap_static()

        self.assertEqual(result, fresh_payload)
        get_json.assert_called_once()


if __name__ == "__main__":
    unittest.main()
