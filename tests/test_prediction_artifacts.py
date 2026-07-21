from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from fplmodel.prediction_artifacts import (
    archive_prediction_file,
    discover_prediction_season_sources,
    infer_bootstrap_season_name,
    inspect_prediction_artifacts,
    migrate_legacy_prediction_files,
)


def _write_bootstrap(
    path: Path,
    start_year: int = 2025,
    team_name: str = "Arsenal",
) -> None:
    payload = {
        "events": [
            {"id": 1, "deadline_time": f"{start_year}-08-15T17:30:00Z"},
            {"id": 38, "deadline_time": f"{start_year + 1}-05-24T13:30:00Z"},
        ],
        "teams": [{"id": 1, "name": team_name}],
        "elements": [
            {
                "id": 1,
                "first_name": "David",
                "second_name": "Raya Martin",
                "team": 1,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _legacy_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1],
            "full_name": ["David Raya Martin"],
            "team_id": [1],
            "team_name": ["Arsenal"],
            "expected_points": [5.25],
        }
    )


class PredictionArtifactTests(unittest.TestCase):
    def test_infers_season_from_bootstrap_deadlines(self) -> None:
        with TemporaryDirectory() as tmpdir:
            bootstrap_path = Path(tmpdir) / "bootstrap.json"
            _write_bootstrap(bootstrap_path)

            self.assertEqual(infer_bootstrap_season_name(bootstrap_path), "2025-26")

    def test_migration_validates_and_backs_up_before_tagging(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bootstrap_path = root / "bootstrap.json"
            output_dir = root / "outputs"
            output_dir.mkdir()
            _write_bootstrap(bootstrap_path)
            legacy_path = output_dir / "predictions_gw1.csv"
            original_text = (
                "player_id,full_name,team_id,team_name,expected_points\n"
                "1,David Raya Martin,1,Arsenal,0.123456789012345678901234\n"
            )
            legacy_path.write_text(original_text, encoding="utf-8")

            result = migrate_legacy_prediction_files(
                output_dir=output_dir,
                bootstrap_path=bootstrap_path,
                expected_season_name="2025-26",
            )

            migrated = pd.read_csv(legacy_path)
            backup = pd.read_csv(result.backup_dir / legacy_path.name)
            self.assertEqual(result.migrated_gameweeks, (1,))
            self.assertEqual(set(migrated["season_name"]), {"2025-26"})
            self.assertEqual(set(migrated["gameweek"]), {1})
            self.assertNotIn("season_name", backup.columns)
            self.assertEqual(
                legacy_path.read_text(encoding="utf-8"),
                original_text.replace(
                    "expected_points\n",
                    "expected_points,season_name,gameweek\n",
                ).replace(
                    "0.123456789012345678901234\n",
                    "0.123456789012345678901234,2025-26,1\n",
                ),
            )

    def test_migration_rejects_a_different_expected_season(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bootstrap_path = root / "bootstrap.json"
            output_dir = root / "outputs"
            output_dir.mkdir()
            _write_bootstrap(bootstrap_path)
            _legacy_predictions().to_csv(output_dir / "predictions_gw1.csv", index=False)

            with self.assertRaisesRegex(ValueError, "2025-26"):
                migrate_legacy_prediction_files(
                    output_dir=output_dir,
                    bootstrap_path=bootstrap_path,
                    expected_season_name="2026-27",
                )

    def test_migration_rejects_players_not_in_the_bootstrap(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bootstrap_path = root / "bootstrap.json"
            output_dir = root / "outputs"
            output_dir.mkdir()
            _write_bootstrap(bootstrap_path)
            frame = _legacy_predictions()
            frame.loc[0, "player_id"] = 999
            frame.to_csv(output_dir / "predictions_gw1.csv", index=False)

            with self.assertRaisesRegex(ValueError, "player IDs"):
                migrate_legacy_prediction_files(
                    output_dir=output_dir,
                    bootstrap_path=bootstrap_path,
                    expected_season_name="2025-26",
                )

    def test_migration_revalidates_already_tagged_player_identity(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bootstrap_path = root / "bootstrap.json"
            output_dir = root / "outputs"
            output_dir.mkdir()
            _write_bootstrap(bootstrap_path)
            frame = _legacy_predictions()
            frame["full_name"] = "Wrong Player"
            frame["season_name"] = "2025-26"
            frame["gameweek"] = 1
            frame.to_csv(output_dir / "predictions_gw1.csv", index=False)

            with self.assertRaisesRegex(ValueError, "full_name"):
                migrate_legacy_prediction_files(
                    output_dir=output_dir,
                    bootstrap_path=bootstrap_path,
                    expected_season_name="2025-26",
                )

    def test_inspection_reports_incomplete_or_conflicting_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bootstrap_path = root / "bootstrap.json"
            output_dir = root / "outputs"
            output_dir.mkdir()
            _write_bootstrap(bootstrap_path)
            missing_gameweek = _legacy_predictions()
            missing_gameweek["season_name"] = "2025-26"
            missing_gameweek.to_csv(output_dir / "predictions_gw1.csv", index=False)
            wrong_season = _legacy_predictions()
            wrong_season["season_name"] = "2024-25"
            wrong_season["gameweek"] = 2
            wrong_season.to_csv(output_dir / "predictions_gw2.csv", index=False)

            inspection = inspect_prediction_artifacts(
                output_dir=output_dir,
                bootstrap_path=bootstrap_path,
                expected_season_name="2025-26",
            )

            self.assertEqual(inspection.ready_gameweeks, ())
            self.assertEqual(inspection.migratable_gameweeks, ())
            issues = dict(inspection.issues)
            self.assertIn("missing or invalid gameweek", issues[1])
            self.assertIn("2024-25", issues[2])

    def test_archives_predictions_with_their_bootstrap_snapshot(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "outputs"
            output_dir.mkdir()
            bootstrap_path = root / "bootstrap.json"
            _write_bootstrap(bootstrap_path)
            predictions_path = output_dir / "predictions_gw1.csv"
            frame = _legacy_predictions()
            frame["season_name"] = "2025-26"
            frame["gameweek"] = 1
            frame.to_csv(predictions_path, index=False)
            original_bytes = predictions_path.read_bytes()

            archived_path = archive_prediction_file(
                predictions_path=predictions_path,
                output_root=output_dir,
                bootstrap_path=bootstrap_path,
                season_name="2025-26",
                gameweek=1,
            )
            sources = discover_prediction_season_sources(
                output_root=output_dir,
                current_bootstrap_path=bootstrap_path,
            )

            self.assertEqual(
                archived_path,
                output_dir / "seasons" / "2025-26" / "predictions_gw1.csv",
            )
            self.assertEqual(archived_path.read_bytes(), original_bytes)
            self.assertEqual(
                (archived_path.parent / "bootstrap-static.json").read_bytes(),
                bootstrap_path.read_bytes(),
            )
            self.assertEqual(sources["2025-26"].predictions_dir, archived_path.parent)
            self.assertTrue(sources["2025-26"].archived)

    def test_discovers_root_and_archived_prediction_seasons(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "outputs"
            output_dir.mkdir()
            current_bootstrap = root / "bootstrap-current.json"
            previous_bootstrap = root / "bootstrap-previous.json"
            _write_bootstrap(current_bootstrap, start_year=2025)
            _write_bootstrap(previous_bootstrap, start_year=2024)

            current = _legacy_predictions()
            current["season_name"] = "2025-26"
            current["gameweek"] = 1
            current.to_csv(output_dir / "predictions_gw1.csv", index=False)
            previous_path = root / "previous_gw1.csv"
            previous = _legacy_predictions()
            previous["season_name"] = "2024-25"
            previous["gameweek"] = 1
            previous.to_csv(previous_path, index=False)
            archive_prediction_file(
                predictions_path=previous_path,
                output_root=output_dir,
                bootstrap_path=previous_bootstrap,
                season_name="2024-25",
                gameweek=1,
            )

            sources = discover_prediction_season_sources(
                output_root=output_dir,
                current_bootstrap_path=current_bootstrap,
            )

            self.assertEqual(set(sources), {"2025-26", "2024-25"})
            self.assertFalse(sources["2025-26"].archived)
            self.assertTrue(sources["2024-25"].archived)

    def test_each_archived_gameweek_keeps_its_matching_bootstrap(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "outputs"
            output_dir.mkdir()
            first_bootstrap = root / "bootstrap-gw1.json"
            second_bootstrap = root / "bootstrap-gw2.json"
            _write_bootstrap(first_bootstrap, team_name="Arsenal")
            _write_bootstrap(second_bootstrap, team_name="Renamed Arsenal")

            first = _legacy_predictions()
            first["season_name"] = "2025-26"
            first["gameweek"] = 1
            first_path = root / "source-gw1.csv"
            first.to_csv(first_path, index=False)
            archive_prediction_file(
                predictions_path=first_path,
                output_root=output_dir,
                bootstrap_path=first_bootstrap,
                season_name="2025-26",
                gameweek=1,
            )

            second = _legacy_predictions()
            second["team_name"] = "Renamed Arsenal"
            second["season_name"] = "2025-26"
            second["gameweek"] = 2
            second_path = root / "source-gw2.csv"
            second.to_csv(second_path, index=False)
            archive_prediction_file(
                predictions_path=second_path,
                output_root=output_dir,
                bootstrap_path=second_bootstrap,
                season_name="2025-26",
                gameweek=2,
            )

            archive_dir = output_dir / "seasons" / "2025-26"
            inspection = inspect_prediction_artifacts(
                output_dir=archive_dir,
                bootstrap_path=archive_dir / "bootstrap-static.json",
                expected_season_name="2025-26",
            )

            self.assertTrue((archive_dir / "bootstrap_gw1.json").is_file())
            self.assertTrue((archive_dir / "bootstrap_gw2.json").is_file())
            self.assertEqual(inspection.ready_gameweeks, (1, 2))
            self.assertEqual(inspection.issues, ())

    def test_discovers_legacy_root_files_alongside_an_older_archive(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "outputs"
            output_dir.mkdir()
            current_bootstrap = root / "bootstrap-current.json"
            previous_bootstrap = root / "bootstrap-previous.json"
            _write_bootstrap(current_bootstrap, start_year=2025)
            _write_bootstrap(previous_bootstrap, start_year=2024)
            _legacy_predictions().to_csv(
                output_dir / "predictions_gw1.csv", index=False
            )

            previous = _legacy_predictions()
            previous["season_name"] = "2024-25"
            previous["gameweek"] = 1
            previous_path = root / "previous-gw1.csv"
            previous.to_csv(previous_path, index=False)
            archive_prediction_file(
                predictions_path=previous_path,
                output_root=output_dir,
                bootstrap_path=previous_bootstrap,
                season_name="2024-25",
                gameweek=1,
            )

            sources = discover_prediction_season_sources(
                output_root=output_dir,
                current_bootstrap_path=current_bootstrap,
            )

            self.assertEqual(set(sources), {"2025-26", "2024-25"})
            self.assertEqual(sources["2025-26"].predictions_dir, output_dir)


if __name__ == "__main__":
    unittest.main()
