from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.import_preseason_usage import build_preseason_usage


class ImportPreseasonUsageTests(unittest.TestCase):
    def _write_inputs(self, root: Path, *, unique_players: int = 11) -> tuple[Path, Path, Path, Path]:
        bootstrap = root / "bootstrap-static.json"
        bootstrap.write_text(
            json.dumps(
                {
                    "teams": [
                        {"id": 1, "code": 101, "name": "Example FC", "short_name": "EXA"}
                    ],
                    "elements": [
                        {
                            "id": player_id,
                            "code": 1000 + player_id,
                            "first_name": "Player",
                            "second_name": str(player_id),
                            "team": 1,
                            "can_select": True,
                            "removed": False,
                        }
                        for player_id in range(1, 13)
                    ],
                }
            ),
            encoding="utf-8",
        )

        core = root / "core"
        core.mkdir()
        pd.DataFrame(
            [
                {
                    "player_id": player_id,
                    "player_code": 1000 + player_id,
                    "team_code": 101,
                }
                for player_id in range(1, 13)
            ]
        ).to_csv(core / "players.csv", index=False)
        pd.DataFrame(
            [
                {
                    "match_id": match_id,
                    "kickoff_time": played_at,
                    "home_team": 101,
                    "away_team": 999,
                    "finished": True,
                }
                for match_id, played_at in (
                    ("core-one", "2026-08-08T14:00:00Z"),
                    ("core-two", "2026-08-15T14:00:00Z"),
                )
            ]
        ).to_csv(core / "matches.csv", index=False)

        observations: list[dict[str, object]] = []
        # Ten exact positive rows per side and 900 observed minutes make each
        # selected Core side independently credible.  Their union covers eleven
        # players, while player 12 remains genuinely unobserved.
        for player_id in range(1, 11):
            observations.append(
                {"player_id": player_id, "match_id": "core-one", "minutes_played": 90}
            )
        second_ids = list(range(2, 12)) if unique_players == 11 else list(range(1, 11))
        for player_id in second_ids:
            observations.append(
                {"player_id": player_id, "match_id": "core-two", "minutes_played": 90}
            )
        pd.DataFrame(observations).to_csv(core / "playermatchstats.csv", index=False)

        selection = root / "selection.csv"
        pd.DataFrame(
            [
                {
                    "team_code": 101,
                    "evidence_match_id": "example_one",
                    "core_match_id": "core-one",
                    "played_at": "2026-08-08T14:00:00Z",
                    "finished": True,
                    "official_complete": False,
                    "absence_safe": True,
                },
                {
                    "team_code": 101,
                    "evidence_match_id": "example_two",
                    "core_match_id": "core-two",
                    "played_at": "2026-08-15T14:00:00Z",
                    "finished": True,
                    "official_complete": False,
                    "absence_safe": True,
                },
            ]
        ).to_csv(selection, index=False)

        supplements = root / "supplements.csv"
        pd.DataFrame(
            [
                {
                    "player_code": 1002,
                    "team_name": "Example FC",
                    "evidence_match_id": "example_two",
                    "played_at": "2026-08-15T14:00:00Z",
                    "minutes": 80,
                    "source_url": "https://example.com/official-report",
                }
            ]
        ).to_csv(supplements, index=False)
        return bootstrap, core, selection, supplements

    def test_builds_positive_only_full_roster_usage_and_official_rows_override_core(self) -> None:
        with TemporaryDirectory() as directory:
            bootstrap, core, selection, supplements = self._write_inputs(Path(directory))

            usage = build_preseason_usage(
                bootstrap_path=bootstrap,
                core_dir=core,
                selection_path=selection,
                supplement_path=supplements,
                cutoff="2026-08-21T17:30:00Z",
                core_source_url="https://github.com/example/core/tree/pinned",
            ).set_index("player_code")

        self.assertEqual(len(usage), 12)
        self.assertEqual(int(usage.loc[1002, "team_preseason_matches"]), 2)
        self.assertEqual(int(usage.loc[1002, "preseason_appearances"]), 2)
        self.assertEqual(float(usage.loc[1002, "preseason_minutes"]), 170.0)
        self.assertEqual(float(usage.loc[1002, "preseason_last_two_minutes"]), 170.0)
        self.assertAlmostEqual(float(usage.loc[1002, "source_confidence"]), 0.75)
        self.assertTrue(pd.isna(usage.loc[1002, "preseason_starts"]))
        self.assertIn("https://example.com/official-report", str(usage.loc[1002, "notes"]))

        # An unobserved player is retained for strict live-roster identity, but no
        # zero usage is manufactured.
        for column in (
            "team_preseason_matches",
            "preseason_appearances",
            "preseason_starts",
            "preseason_minutes",
            "preseason_last_two_minutes",
            "source_confidence",
        ):
            self.assertTrue(pd.isna(usage.loc[1012, column]))

    def test_fails_when_selected_positive_evidence_covers_only_ten_players(self) -> None:
        with TemporaryDirectory() as directory:
            bootstrap, core, selection, supplements = self._write_inputs(
                Path(directory), unique_players=10
            )

            with self.assertRaisesRegex(ValueError, "meaningful club coverage.*10/11"):
                build_preseason_usage(
                    bootstrap_path=bootstrap,
                    core_dir=core,
                    selection_path=selection,
                    supplement_path=supplements,
                    cutoff="2026-08-21T17:30:00Z",
                    core_source_url="https://github.com/example/core/tree/pinned",
                )

    def test_fails_when_a_selected_core_side_is_not_credible(self) -> None:
        with TemporaryDirectory() as directory:
            bootstrap, core, selection, supplements = self._write_inputs(Path(directory))
            stats = pd.read_csv(core / "playermatchstats.csv")
            stats.loc[stats["match_id"].eq("core-one"), "minutes_played"] = 45
            stats.to_csv(core / "playermatchstats.csv", index=False)

            with self.assertRaisesRegex(ValueError, "selected evidence side is not credible"):
                build_preseason_usage(
                    bootstrap_path=bootstrap,
                    core_dir=core,
                    selection_path=selection,
                    supplement_path=supplements,
                    cutoff="2026-08-21T17:30:00Z",
                    core_source_url="https://github.com/example/core/tree/pinned",
                )

    def test_partial_official_match_counts_only_for_players_observed_in_it(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bootstrap, core, selection, supplements = self._write_inputs(root)

            stats = pd.read_csv(core / "playermatchstats.csv")
            stats = pd.concat(
                [
                    stats,
                    pd.DataFrame(
                        [{"player_id": 12, "match_id": "core-one", "minutes_played": 90}]
                    ),
                ],
                ignore_index=True,
            )
            stats.to_csv(core / "playermatchstats.csv", index=False)

            selected = pd.read_csv(selection)
            selected.loc[1, "core_match_id"] = ""
            selected.loc[1, "official_complete"] = False
            selected.loc[1, "absence_safe"] = False
            selected.to_csv(selection, index=False)
            pd.DataFrame(
                [
                    {
                        "player_code": 1000 + player_id,
                        "team_name": "Example FC",
                        "evidence_match_id": "example_two",
                        "played_at": "2026-08-15T14:00:00Z",
                        "minutes": 45,
                        "source_url": "https://example.com/partial-official-report",
                    }
                    for player_id in range(1, 12)
                ]
            ).to_csv(supplements, index=False)

            usage = build_preseason_usage(
                bootstrap_path=bootstrap,
                core_dir=core,
                selection_path=selection,
                supplement_path=supplements,
                cutoff="2026-08-21T17:30:00Z",
                core_source_url="https://github.com/example/core/tree/pinned",
            ).set_index("player_code")

        self.assertEqual(int(usage.loc[1001, "team_preseason_matches"]), 2)
        self.assertEqual(int(usage.loc[1012, "team_preseason_matches"]), 1)

    def test_complete_official_selection_requires_the_stronger_minute_floor(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bootstrap, core, selection, supplements = self._write_inputs(root)
            selected = pd.read_csv(selection)
            selected.loc[1, "core_match_id"] = ""
            selected.loc[1, "official_complete"] = True
            selected.loc[1, "absence_safe"] = True
            selected.to_csv(selection, index=False)
            pd.DataFrame(
                [
                    {
                        "player_code": 1000 + player_id,
                        "team_name": "Example FC",
                        "evidence_match_id": "example_two",
                        "played_at": "2026-08-15T14:00:00Z",
                        "minutes": 45,
                        "source_url": "https://example.com/complete-report",
                    }
                    for player_id in range(1, 12)
                ]
            ).to_csv(supplements, index=False)

            with self.assertRaisesRegex(ValueError, "selected evidence side is not credible"):
                build_preseason_usage(
                    bootstrap_path=bootstrap,
                    core_dir=core,
                    selection_path=selection,
                    supplement_path=supplements,
                    cutoff="2026-08-21T17:30:00Z",
                    core_source_url="https://github.com/example/core/tree/pinned",
                )

    def test_complete_official_rows_can_replace_a_discarded_core_side(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bootstrap, core, selection, supplements = self._write_inputs(root)
            stats = pd.read_csv(core / "playermatchstats.csv")
            stats.loc[stats["match_id"].eq("core-one"), "minutes_played"] = 45
            stats.to_csv(core / "playermatchstats.csv", index=False)
            selected = pd.read_csv(selection)
            selected.loc[0, "official_complete"] = True
            selected.to_csv(selection, index=False)
            official = pd.read_csv(supplements)
            official = pd.concat(
                [
                    official,
                    pd.DataFrame(
                        [
                            {
                                "player_code": 1000 + player_id,
                                "team_name": "Example FC",
                                "evidence_match_id": "example_one",
                                "played_at": "2026-08-08T14:00:00Z",
                                "minutes": 90,
                                "source_url": "https://example.com/complete-report",
                            }
                            for player_id in range(1, 12)
                        ]
                    ),
                ],
                ignore_index=True,
            )
            official.to_csv(supplements, index=False)

            usage = build_preseason_usage(
                bootstrap_path=bootstrap,
                core_dir=core,
                selection_path=selection,
                supplement_path=supplements,
                cutoff="2026-08-21T17:30:00Z",
                core_source_url="https://github.com/example/core/tree/pinned",
            )

        self.assertEqual(len(usage), 12)

    def test_rejects_selected_evidence_after_the_cutoff(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bootstrap, core, selection, supplements = self._write_inputs(root)
            selected = pd.read_csv(selection)
            selected.loc[1, "played_at"] = "2026-08-22T14:00:00Z"
            selected.to_csv(selection, index=False)

            with self.assertRaisesRegex(ValueError, "on or before the cutoff"):
                build_preseason_usage(
                    bootstrap_path=bootstrap,
                    core_dir=core,
                    selection_path=selection,
                    supplement_path=supplements,
                    cutoff="2026-08-21T17:30:00Z",
                    core_source_url="https://github.com/example/core/tree/pinned",
                )

    def test_rejects_same_date_supplement_timestamp_after_the_cutoff(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bootstrap, core, selection, supplements = self._write_inputs(root)
            selected = pd.read_csv(selection)
            selected.loc[1, "core_match_id"] = ""
            selected.loc[1, "played_at"] = "2026-08-21T12:00:00Z"
            selected.loc[1, "official_complete"] = True
            selected.loc[1, "absence_safe"] = True
            selected.to_csv(selection, index=False)
            official = pd.read_csv(supplements)
            official.loc[0, "played_at"] = "2026-08-21T23:00:00Z"
            official.to_csv(supplements, index=False)

            with self.assertRaisesRegex(ValueError, "Official supplement evidence.*cutoff"):
                build_preseason_usage(
                    bootstrap_path=bootstrap,
                    core_dir=core,
                    selection_path=selection,
                    supplement_path=supplements,
                    cutoff="2026-08-21T17:30:00Z",
                    core_source_url="https://github.com/example/core/tree/pinned",
                )


if __name__ == "__main__":
    unittest.main()
