from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from fplmodel.preseason import (
    derive_preseason_role_evidence,
    derive_preseason_player_priors,
    load_preseason_snapshot_context,
    normalize_tournament_workload,
)
from fplmodel.snapshot_catalog import SnapshotCatalog
from fplmodel.team_priors import build_team_priors_from_results
from scripts.update_preseason_snapshots import (
    _default_football_data_url,
    _validated_football_data_url,
    build_team_prior_snapshot,
    create_player_usage_template,
    create_role_evidence_template,
    create_workload_template,
)


def _strict_player_universe(players_per_team: int = 11) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for team_id in range(1, 21):
        for squad_number in range(players_per_team):
            player_code = team_id * 1000 + squad_number
            rows.append(
                {
                    "player_code": player_code,
                    "player_id": player_code + 5000,
                    "full_name": f"Player {player_code}",
                    "team_id": team_id,
                    "team_code": team_id * 10,
                    "team_name": f"Team {team_id}",
                    "can_select": True,
                    "removed": False,
                }
            )
    return pd.DataFrame(rows)


def _complete_usage(players: pd.DataFrame) -> pd.DataFrame:
    return players[
        ["player_code", "player_id", "full_name", "team_code", "team_name"]
    ].assign(
        team_preseason_matches=4,
        preseason_appearances=4,
        preseason_starts=3,
        preseason_minutes=280,
        preseason_last_two_minutes=150,
        source_confidence=0.8,
    )


def _complete_team_priors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_code": [team_id * 10 for team_id in range(1, 21)],
            "team_id": list(range(1, 21)),
            "team_goals_for_ma5": [1.0] * 20,
            "team_goals_against_ma5": [1.0] * 20,
            "team_goal_diff_ma5": [0.0] * 20,
            "team_clean_sheet_ma5": [0.2] * 20,
            "team_conceded_two_plus_ma5": [0.3] * 20,
            "team_match_points_ma5": [1.5] * 20,
            "team_prior_matches": [5] * 20,
            "team_prior_as_of": ["2026-05-24"] * 20,
            "team_prior_source": ["manual"] * 20,
        }
    )


class SnapshotCatalogTests(unittest.TestCase):
    def test_refuses_to_persist_a_credential_in_source_url(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "credentials"):
                catalog.publish_frame(
                    pd.DataFrame({"player_code": [1]}),
                    season_name="2026-27",
                    topic="fixture_odds",
                    provider="odds",
                    captured_at="2026-08-01T12:00:00Z",
                    source_url="https://example.test/odds?apiKey=secret",
                    rights_class="provider_licensed",
                )

    def test_refuses_userinfo_and_nested_manifest_urls_with_credentials(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "credentials"):
                catalog.publish_frame(
                    pd.DataFrame({"player_code": [1]}),
                    season_name="2026-27",
                    topic="fixture_odds",
                    provider="odds",
                    captured_at="2026-08-01T12:00:00Z",
                    source_url="https://user:secret@example.test/odds",
                    rights_class="provider_licensed",
                )
            with self.assertRaisesRegex(ValueError, "credentials"):
                catalog.publish_frame(
                    pd.DataFrame({"team_code": [9]}),
                    season_name="2026-27",
                    topic="team_priors",
                    provider="manual",
                    captured_at="2026-08-01T12:00:00Z",
                    source_url="https://example.test/ok",
                    rights_class="local_use_only",
                    extra_manifest={"nested": {"source_urls": ["https://example.test/file?token=secret"]}},
                )
            with self.assertRaisesRegex(ValueError, "credentials"):
                catalog.publish_frame(
                    pd.DataFrame({"team_code": [9]}),
                    season_name="2026-27",
                    topic="team_priors",
                    provider="manual",
                    captured_at="2026-08-01T12:00:00Z",
                    source_url="https://example.test/ok",
                    rights_class="local_use_only",
                    extra_manifest={
                        "links": [
                            "https://example.test/file?subscription-key=secret"
                        ]
                    },
                )

    def test_resolves_latest_snapshot_at_or_before_cutoff(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            older = catalog.publish_frame(
                pd.DataFrame({"player_code": [1], "preseason_minutes": [90]}),
                season_name="2026-27",
                topic="preseason_player_usage",
                provider="manual",
                captured_at="2026-08-01T12:00:00Z",
                source_url="https://example.test/older",
                rights_class="local_use_only",
            )
            catalog.publish_frame(
                pd.DataFrame({"player_code": [1], "preseason_minutes": [180]}),
                season_name="2026-27",
                topic="preseason_player_usage",
                provider="manual",
                captured_at="2026-08-10T12:00:00Z",
                source_url="https://example.test/future",
                rights_class="local_use_only",
            )

            resolved = catalog.resolve(
                season_name="2026-27",
                topic="preseason_player_usage",
                cutoff=datetime(2026, 8, 5, tzinfo=timezone.utc),
            )

            self.assertEqual(resolved, older)
            self.assertEqual(catalog.load_frame(resolved)["preseason_minutes"].tolist(), [90])

    def test_strict_resolution_rejects_future_only_snapshot(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            catalog.publish_frame(
                pd.DataFrame({"team_code": [9]}),
                season_name="2026-27",
                topic="team_priors",
                provider="football_data_co_uk",
                captured_at="2026-08-10T12:00:00Z",
                source_url="https://example.test/future",
                rights_class="local_use_only",
            )

            with self.assertRaisesRegex(FileNotFoundError, "cutoff"):
                catalog.resolve(
                    season_name="2026-27",
                    topic="team_priors",
                    cutoff=datetime(2026, 8, 5, tzinfo=timezone.utc),
                    required=True,
                )

    def test_manifest_integrity_detects_sidecar_tampering(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            snapshot = catalog.publish_frame(
                pd.DataFrame({"player_code": [1], "preseason_minutes": [90]}),
                season_name="2026-27",
                topic="preseason_player_usage",
                provider="manual",
                captured_at="2026-08-01T12:00:00Z",
                source_url="https://example.test/older",
                rights_class="local_use_only",
            )
            manifest_path = catalog.manifest_path(snapshot)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["captured_at"] = "2026-08-02T12:00:00Z"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "integrity validation"):
                catalog.read_manifest(snapshot)

    def test_context_merges_player_topics_and_skips_wrong_provider_gameweek(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            for topic, frame in (
                (
                    "preseason_player_usage",
                    pd.DataFrame(
                        {
                            "player_code": [1001],
                            "team_preseason_matches": [4],
                            "preseason_starts": [4],
                        }
                    ),
                ),
                (
                    "tournament_workload",
                    pd.DataFrame(
                        {
                            "player_code": [1001],
                            "tournament_minutes": [420],
                            "tournament_matches": [5],
                            "tournament_starts": [5],
                            "last_tournament_appearance_at": ["2026-07-19T20:00:00Z"],
                            "readiness_multiplier": [0.8],
                            "workload_evidence_confidence": [0.9],
                            "workload_valid_through_gameweek": [1],
                        }
                    ),
                ),
                (
                    "provider_expected_points",
                    pd.DataFrame(
                        {
                            "player_code": [1001],
                            "gameweek": [2],
                            "provider_expected_points": [6.5],
                        }
                    ),
                ),
            ):
                catalog.publish_frame(
                    frame,
                    season_name="2026-27",
                    topic=topic,
                    provider="test_provider",
                    captured_at="2026-08-01T12:00:00Z",
                    source_url=f"https://example.test/{topic}",
                    rights_class="local_use_only",
                )

            context = load_preseason_snapshot_context(
                catalog,
                season_name="2026-27",
                cutoff="2026-08-20T12:00:00Z",
                target_gameweek=1,
            )

        player = context.player_enrichment.iloc[0]
        self.assertEqual(int(player["player_code"]), 1001)
        self.assertEqual(float(player["tournament_minutes"]), 420.0)
        self.assertNotIn("provider_expected_points", context.player_enrichment.columns)
        self.assertIn("preseason_data_source", context.player_enrichment.columns)
        self.assertIn("workload_data_source", context.player_enrichment.columns)

    def test_context_uses_latest_provider_snapshot_matching_target_gameweek(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            catalog.publish_frame(
                pd.DataFrame({"player_code": [1001], "team_preseason_matches": [4], "preseason_starts": [4]}),
                season_name="2026-27",
                topic="preseason_player_usage",
                provider="manual",
                captured_at="2026-08-01T12:00:00Z",
                source_url="https://example.test/usage",
                rights_class="local_use_only",
            )
            catalog.publish_frame(
                pd.DataFrame({"player_code": [1001], "gameweek": [2], "provider_expected_points": [6.0]}),
                season_name="2026-27",
                topic="provider_expected_points",
                provider="provider",
                captured_at="2026-08-03T12:00:00Z",
                source_url="https://example.test/gw2",
                rights_class="local_use_only",
            )
            catalog.publish_frame(
                pd.DataFrame({"player_code": [1001], "gameweek": [1], "provider_expected_points": [7.0]}),
                season_name="2026-27",
                topic="provider_expected_points",
                provider="provider",
                captured_at="2026-08-02T12:00:00Z",
                source_url="https://example.test/gw1",
                rights_class="local_use_only",
            )

            context = load_preseason_snapshot_context(
                catalog,
                season_name="2026-27",
                cutoff="2026-08-20T12:00:00Z",
                target_gameweek=1,
            )

        self.assertEqual(
            float(context.player_enrichment.loc[0, "provider_expected_points"]),
            7.0,
        )

    def test_context_uses_role_evidence_for_the_target_gameweek_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            for captured_at, gameweek, score in (
                ("2026-08-02T12:00:00Z", 1, 1.0),
                ("2026-08-03T12:00:00Z", 2, -1.0),
            ):
                catalog.publish_frame(
                    pd.DataFrame(
                        {
                            "player_code": [1001],
                            "role_target_gameweek": [gameweek],
                            "attacking_role_delta": [score],
                            "penalty_role_delta": [pd.NA],
                            "corner_role_delta": [pd.NA],
                            "direct_fk_role_delta": [pd.NA],
                            "role_evidence_matches": [3],
                            "role_evidence_confidence": [0.8],
                            "role_last_observed_at": [captured_at],
                            "role_notes": ["observed"],
                        }
                    ),
                    season_name="2026-27",
                    topic="preseason_role_evidence",
                    provider="manual",
                    captured_at=captured_at,
                    source_url="https://example.test/role",
                    rights_class="local_use_only",
                )

            context = load_preseason_snapshot_context(
                catalog,
                season_name="2026-27",
                cutoff="2026-08-20T12:00:00Z",
                target_gameweek=1,
            )

        player = context.player_enrichment.iloc[0]
        self.assertEqual(int(player["role_target_gameweek"]), 1)
        self.assertAlmostEqual(float(player["role_change_score"]), 0.45)
        self.assertAlmostEqual(float(player["role_evidence_quality"]), 0.8)
        self.assertEqual(
            player["role_data_source"],
            "manual:2026-08-02T12:00:00Z",
        )

    def test_context_rejects_role_observation_after_its_snapshot_capture(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            catalog.publish_frame(
                pd.DataFrame(
                    {
                        "player_code": [1001],
                        "role_target_gameweek": [1],
                        "attacking_role_delta": [0.5],
                        "role_evidence_matches": [2],
                        "role_evidence_confidence": [0.8],
                        "role_last_observed_at": ["2026-08-11T12:00:00Z"],
                    }
                ),
                season_name="2026-27",
                topic="preseason_role_evidence",
                provider="manual",
                captured_at="2026-08-10T12:00:00Z",
                source_url="https://example.test/role",
                rights_class="local_use_only",
            )

            with self.assertRaisesRegex(ValueError, "after snapshot capture"):
                load_preseason_snapshot_context(
                    catalog,
                    season_name="2026-27",
                    cutoff="2026-08-20T12:00:00Z",
                    target_gameweek=1,
                )

    def test_newer_expired_workload_does_not_revive_an_older_snapshot(self) -> None:
        def workload(valid_through: int, readiness: float) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "player_code": [1001],
                    "tournament_minutes": [420],
                    "tournament_matches": [5],
                    "tournament_starts": [5],
                    "last_tournament_appearance_at": ["2026-07-19T20:00:00Z"],
                    "readiness_multiplier": [readiness],
                    "workload_evidence_confidence": [0.9],
                    "workload_valid_through_gameweek": [valid_through],
                }
            )

        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            for captured_at, frame in (
                ("2026-08-01T12:00:00Z", workload(38, 0.8)),
                ("2026-08-10T12:00:00Z", workload(1, 1.0)),
            ):
                catalog.publish_frame(
                    frame,
                    season_name="2026-27",
                    topic="tournament_workload",
                    provider="manual",
                    captured_at=captured_at,
                    source_url="https://example.test/workload",
                    rights_class="local_use_only",
                )

            context = load_preseason_snapshot_context(
                catalog,
                season_name="2026-27",
                cutoff="2026-08-25T12:00:00Z",
                target_gameweek=2,
            )

        self.assertTrue(context.player_enrichment.empty)

    def test_context_merges_shared_identity_and_readiness_without_conflict(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            catalog.publish_frame(
                pd.DataFrame(
                    {
                        "player_code": [1001],
                        "player_id": [7],
                        "full_name": ["Test Player"],
                        "team_name": ["Example FC"],
                        "team_preseason_matches": [4],
                        "preseason_starts": [4],
                    }
                ),
                season_name="2026-27",
                topic="preseason_player_usage",
                provider="manual",
                captured_at="2026-08-01T12:00:00Z",
                source_url="https://example.test/usage",
                rights_class="local_use_only",
            )
            catalog.publish_frame(
                pd.DataFrame(
                    {
                        "player_code": [1001],
                        "player_id": [7],
                        "full_name": ["Test Player"],
                        "team_name": ["Example FC"],
                        "readiness_multiplier": [0.8],
                        "tournament_minutes": [210],
                        "tournament_matches": [3],
                        "tournament_starts": [2],
                        "last_tournament_appearance_at": ["2026-07-19T20:00:00Z"],
                        "workload_evidence_confidence": [0.9],
                        "workload_valid_through_gameweek": [1],
                    }
                ),
                season_name="2026-27",
                topic="tournament_workload",
                provider="manual",
                captured_at="2026-08-02T12:00:00Z",
                source_url="https://example.test/workload",
                rights_class="local_use_only",
            )

            context = load_preseason_snapshot_context(
                catalog,
                season_name="2026-27",
                cutoff="2026-08-20T12:00:00Z",
                target_gameweek=1,
            )

        player = context.player_enrichment.iloc[0]
        self.assertEqual(player["full_name"], "Test Player")
        self.assertEqual(float(player["readiness_multiplier"]), 0.8)
        self.assertEqual(float(player["tournament_minutes"]), 210.0)

    def test_context_rejects_conflicting_nonowned_player_columns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            catalog.publish_frame(
                pd.DataFrame({"player_code": [1001], "source_confidence": [0.4]}),
                season_name="2026-27",
                topic="preseason_player_usage",
                provider="manual",
                captured_at="2026-08-01T12:00:00Z",
                source_url="https://example.test/usage",
                rights_class="local_use_only",
            )
            catalog.publish_frame(
                pd.DataFrame(
                    {
                        "player_code": [1001],
                        "source_confidence": [0.8],
                        "tournament_minutes": [210],
                        "tournament_matches": [3],
                        "tournament_starts": [2],
                        "last_tournament_appearance_at": ["2026-07-19T20:00:00Z"],
                        "readiness_multiplier": [0.8],
                        "workload_evidence_confidence": [0.9],
                        "workload_valid_through_gameweek": [1],
                    }
                ),
                season_name="2026-27",
                topic="tournament_workload",
                provider="manual",
                captured_at="2026-08-02T12:00:00Z",
                source_url="https://example.test/workload",
                rights_class="local_use_only",
            )

            with self.assertRaisesRegex(ValueError, "ambiguous columns"):
                load_preseason_snapshot_context(
                    catalog,
                    season_name="2026-27",
                    cutoff="2026-08-20T12:00:00Z",
                    target_gameweek=1,
                )

    def test_strict_gw1_requires_full_player_and_team_evidence(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            usage = pd.DataFrame(
                {
                    "player_code": [1001],
                    "prior_p_start": [pd.NA],
                    "prior_p_appearance": [pd.NA],
                }
            )
            team_priors = pd.DataFrame(
                {
                    "team_code": list(range(1, 20)),
                    "team_id": list(range(1, 20)),
                    "team_goals_for_ma5": [1.0] * 19,
                    "team_goals_against_ma5": [1.0] * 19,
                    "team_prior_source": ["manual"] * 19,
                }
            )
            for topic, frame in (
                ("preseason_player_usage", usage),
                ("team_priors", team_priors),
            ):
                catalog.publish_frame(
                    frame,
                    season_name="2026-27",
                    topic=topic,
                    provider="manual",
                    captured_at="2026-08-01T12:00:00Z",
                    source_url=f"https://example.test/{topic}",
                    rights_class="local_use_only",
                )

            with self.assertRaisesRegex(ValueError, "non-null player evidence"):
                load_preseason_snapshot_context(
                    catalog,
                    season_name="2026-27",
                    cutoff="2026-08-20T12:00:00Z",
                    mode="strict",
                    required_topics=("preseason_player_usage", "team_priors"),
                    target_gameweek=1,
                )

    def test_strict_provider_expected_points_requires_matching_gameweek(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            catalog.publish_frame(
                pd.DataFrame({"player_code": [1001], "gameweek": [2], "provider_expected_points": [6.0]}),
                season_name="2026-27",
                topic="provider_expected_points",
                provider="provider",
                captured_at="2026-08-03T12:00:00Z",
                source_url="https://example.test/gw2",
                rights_class="local_use_only",
            )

            with self.assertRaisesRegex(FileNotFoundError, "gameweek 1"):
                load_preseason_snapshot_context(
                    catalog,
                    season_name="2026-27",
                    cutoff="2026-08-20T12:00:00Z",
                    mode="strict",
                    required_topics=("provider_expected_points",),
                    target_gameweek=1,
                )

    def test_strict_gw1_team_priors_must_match_the_current_team_codes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            catalog.publish_frame(
                pd.DataFrame(
                    {
                        "player_code": [1001],
                        "prior_p_start": [0.8],
                        "prior_p_appearance": [0.9],
                    }
                ),
                season_name="2026-27",
                topic="preseason_player_usage",
                provider="manual",
                captured_at="2026-08-01T12:00:00Z",
                source_url="https://example.test/usage",
                rights_class="local_use_only",
            )
            catalog.publish_frame(
                pd.DataFrame(
                    {
                        "team_code": list(range(1, 21)),
                        "team_id": list(range(1, 21)),
                        "team_goals_for_ma5": [1.0] * 20,
                        "team_goals_against_ma5": [1.0] * 20,
                        "team_goal_diff_ma5": [0.0] * 20,
                        "team_clean_sheet_ma5": [0.2] * 20,
                        "team_conceded_two_plus_ma5": [0.3] * 20,
                        "team_match_points_ma5": [1.5] * 20,
                        "team_prior_matches": [5] * 20,
                        "team_prior_source": ["manual"] * 20,
                    }
                ),
                season_name="2026-27",
                topic="team_priors",
                provider="manual",
                captured_at="2026-08-01T12:00:00Z",
                source_url="https://example.test/team-priors",
                rights_class="local_use_only",
            )

            with self.assertRaisesRegex(ValueError, "current team codes"):
                load_preseason_snapshot_context(
                    catalog,
                    season_name="2026-27",
                    cutoff="2026-08-20T12:00:00Z",
                    mode="strict",
                    required_topics=("preseason_player_usage", "team_priors"),
                    target_gameweek=1,
                    expected_team_codes=set(range(2, 22)),
                )

    def test_strict_gw1_rejects_an_incomplete_team_feature_schema(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            catalog.publish_frame(
                pd.DataFrame(
                    {
                        "player_code": [1001],
                        "prior_p_start": [0.8],
                        "prior_p_appearance": [0.9],
                    }
                ),
                season_name="2026-27",
                topic="preseason_player_usage",
                provider="manual",
                captured_at="2026-08-01T12:00:00Z",
                source_url="https://example.test/usage",
                rights_class="local_use_only",
            )
            catalog.publish_frame(
                pd.DataFrame(
                    {
                        "team_code": list(range(1, 21)),
                        "team_id": list(range(1, 21)),
                        "team_goals_for_ma5": [1.0] * 20,
                        "team_goals_against_ma5": [1.0] * 20,
                        "team_prior_source": ["manual"] * 20,
                    }
                ),
                season_name="2026-27",
                topic="team_priors",
                provider="manual",
                captured_at="2026-08-01T12:00:00Z",
                source_url="https://example.test/team-priors",
                rights_class="local_use_only",
            )

            with self.assertRaisesRegex(ValueError, "team-prior columns"):
                load_preseason_snapshot_context(
                    catalog,
                    season_name="2026-27",
                    cutoff="2026-08-20T12:00:00Z",
                    mode="strict",
                    required_topics=("preseason_player_usage", "team_priors"),
                    target_gameweek=1,
                    expected_team_codes=set(range(1, 21)),
                )

    def test_strict_gw1_accepts_fresh_full_roster_and_meaningful_club_coverage(self) -> None:
        players = _strict_player_universe()
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            for topic, frame in (
                ("preseason_player_usage", _complete_usage(players)),
                ("team_priors", _complete_team_priors()),
            ):
                catalog.publish_frame(
                    frame,
                    season_name="2026-27",
                    topic=topic,
                    provider="manual",
                    captured_at="2026-08-16T12:00:00Z",
                    source_url=f"https://example.test/{topic}",
                    rights_class="local_use_only",
                )

            context = load_preseason_snapshot_context(
                catalog,
                season_name="2026-27",
                cutoff="2026-08-20T12:00:00Z",
                mode="strict",
                required_topics=("preseason_player_usage", "team_priors"),
                target_gameweek=1,
                expected_team_codes=set(players["team_code"]),
                expected_players=players,
            )

        self.assertEqual(len(context.player_enrichment), 220)

    def test_strict_gw1_rejects_stale_or_thin_usage_evidence(self) -> None:
        players = _strict_player_universe()

        def load_with_usage(usage: pd.DataFrame, captured_at: str) -> None:
            with TemporaryDirectory() as tmpdir:
                catalog = SnapshotCatalog(Path(tmpdir))
                for topic, frame in (
                    ("preseason_player_usage", usage),
                    ("team_priors", _complete_team_priors()),
                ):
                    catalog.publish_frame(
                        frame,
                        season_name="2026-27",
                        topic=topic,
                        provider="manual",
                        captured_at=captured_at,
                        source_url=f"https://example.test/{topic}",
                        rights_class="local_use_only",
                    )
                load_preseason_snapshot_context(
                    catalog,
                    season_name="2026-27",
                    cutoff="2026-08-20T12:00:00Z",
                    mode="strict",
                    required_topics=("preseason_player_usage", "team_priors"),
                    target_gameweek=1,
                    expected_team_codes=set(players["team_code"]),
                    expected_players=players,
                )

        with self.assertRaisesRegex(ValueError, "older than 7 days"):
            load_with_usage(_complete_usage(players), "2026-08-12T11:59:59Z")

        thin_usage = _complete_usage(players)
        team_one_player = thin_usage["team_code"].eq(10).idxmax()
        thin_usage.loc[
            team_one_player,
            [
                "preseason_appearances",
                "preseason_starts",
                "preseason_minutes",
                "preseason_last_two_minutes",
            ],
        ] = pd.NA
        with self.assertRaisesRegex(ValueError, r"team 10: 10/11"):
            load_with_usage(thin_usage, "2026-08-16T12:00:00Z")

    def test_strict_gw1_rejects_stale_or_mismatched_team_priors(self) -> None:
        players = _strict_player_universe()
        expected_teams = players[["team_id", "team_code"]].drop_duplicates()

        def load_with_priors(priors: pd.DataFrame, prior_season: str) -> None:
            with TemporaryDirectory() as tmpdir:
                catalog = SnapshotCatalog(Path(tmpdir))
                catalog.publish_frame(
                    _complete_usage(players),
                    season_name="2026-27",
                    topic="preseason_player_usage",
                    provider="manual",
                    captured_at="2026-08-16T12:00:00Z",
                    source_url="https://example.test/usage",
                    rights_class="local_use_only",
                )
                catalog.publish_frame(
                    priors,
                    season_name="2026-27",
                    topic="team_priors",
                    provider="manual",
                    captured_at="2026-08-16T12:00:00Z",
                    source_url="https://example.test/priors",
                    rights_class="local_use_only",
                    extra_manifest={"prior_season": prior_season},
                )
                load_preseason_snapshot_context(
                    catalog,
                    season_name="2026-27",
                    cutoff="2026-08-20T12:00:00Z",
                    mode="strict",
                    required_topics=("preseason_player_usage", "team_priors"),
                    target_gameweek=1,
                    expected_team_codes=set(players["team_code"]),
                    expected_players=players,
                    expected_teams=expected_teams,
                )

        swapped = _complete_team_priors()
        swapped.loc[[0, 1], "team_id"] = swapped.loc[[1, 0], "team_id"].to_numpy()
        with self.assertRaisesRegex(ValueError, "team_code-to-team_id"):
            load_with_priors(swapped, "2025-26")

        stale = _complete_team_priors()
        stale["team_prior_as_of"] = "2026-04-01"
        with self.assertRaisesRegex(ValueError, "older than 120 days"):
            load_with_priors(stale, "2025-26")

        with self.assertRaisesRegex(ValueError, "prior_season"):
            load_with_priors(_complete_team_priors(), "2024-25")

        wrong_window = _complete_team_priors()
        wrong_window["team_prior_as_of"] = "2026-08-01"
        with self.assertRaisesRegex(ValueError, "outside 2025-26"):
            load_with_priors(wrong_window, "2025-26")

    def test_strict_context_loads_all_six_snapshot_topics_together(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            topics = {
                "preseason_player_usage": pd.DataFrame(
                    {
                        "player_code": [1001],
                        "prior_p_start": [0.75],
                        "prior_p_appearance": [0.9],
                    }
                ),
                "tournament_workload": pd.DataFrame(
                    {
                        "player_code": [1001],
                        "tournament_minutes": [420],
                        "tournament_matches": [5],
                        "tournament_starts": [5],
                        "last_tournament_appearance_at": ["2026-07-19T20:00:00Z"],
                        "readiness_multiplier": [0.85],
                        "workload_evidence_confidence": [0.9],
                        "workload_valid_through_gameweek": [1],
                    }
                ),
                "preseason_role_evidence": pd.DataFrame(
                    {
                        "player_code": [1001],
                        "role_target_gameweek": [1],
                        "attacking_role_delta": [0.5],
                        "role_evidence_matches": [3],
                        "role_evidence_confidence": [0.8],
                        "role_last_observed_at": ["2026-08-01T10:00:00Z"],
                    }
                ),
                "provider_expected_points": pd.DataFrame(
                    {
                        "player_code": [1001],
                        "gameweek": [1],
                        "provider_expected_points": [6.4],
                    }
                ),
                "fixture_odds": pd.DataFrame(
                    {"provider_event_id": ["gw1-a"], "home_win_probability": [0.5]}
                ),
                "team_priors": pd.DataFrame(
                    {
                        "team_code": list(range(1, 21)),
                        "team_id": list(range(1, 21)),
                        "team_goals_for_ma5": [1.0] * 20,
                        "team_goals_against_ma5": [1.0] * 20,
                        "team_goal_diff_ma5": [0.0] * 20,
                        "team_clean_sheet_ma5": [0.2] * 20,
                        "team_conceded_two_plus_ma5": [0.3] * 20,
                        "team_match_points_ma5": [1.5] * 20,
                        "team_prior_matches": [5] * 20,
                        "team_prior_source": ["manual"] * 20,
                    }
                ),
            }
            for topic, frame in topics.items():
                catalog.publish_frame(
                    frame,
                    season_name="2026-27",
                    topic=topic,
                    provider="manual",
                    captured_at="2026-08-01T12:00:00Z",
                    source_url=f"https://example.test/{topic}",
                    rights_class="local_use_only",
                )

            context = load_preseason_snapshot_context(
                catalog,
                season_name="2026-27",
                cutoff="2026-08-20T12:00:00Z",
                mode="strict",
                required_topics=("preseason_player_usage", "team_priors"),
                target_gameweek=1,
                expected_team_codes=set(range(1, 21)),
            )

        self.assertEqual(len(context.snapshot_paths), 6)
        self.assertEqual(float(context.player_enrichment.loc[0, "readiness_multiplier"]), 0.85)
        self.assertEqual(float(context.player_enrichment.loc[0, "provider_expected_points"]), 6.4)
        self.assertAlmostEqual(float(context.player_enrichment.loc[0, "role_change_score"]), 0.225)
        self.assertEqual(len(context.team_priors), 20)
        self.assertEqual(len(context.fixture_odds), 1)


class PreseasonPriorTests(unittest.TestCase):
    def test_usage_priors_reward_recent_starts_and_report_evidence_quality(self) -> None:
        usage = pd.DataFrame(
            {
                "player_code": [1001, 1002],
                "team_preseason_matches": [4, 4],
                "preseason_appearances": [4, 4],
                "preseason_starts": [4, 1],
                "preseason_minutes": [300, 150],
                "preseason_last_two_minutes": [170, 20],
                "source_confidence": [1.0, 1.0],
            }
        )

        priors = derive_preseason_player_priors(usage).set_index("player_code")

        self.assertGreater(priors.loc[1001, "prior_p_start"], priors.loc[1002, "prior_p_start"])
        self.assertGreaterEqual(priors.loc[1001, "prior_p_appearance"], priors.loc[1001, "prior_p_start"])
        self.assertEqual(float(priors.loc[1001, "preseason_prior_quality"]), 1.0)
        self.assertTrue(priors["prior_p_start"].between(0.0, 1.0).all())

    def test_role_evidence_derives_a_bounded_auditable_signal(self) -> None:
        evidence = pd.DataFrame(
            {
                "player_code": [1001],
                "role_target_gameweek": [1],
                "attacking_role_delta": [1.0],
                "penalty_role_delta": [1.0],
                "corner_role_delta": [0.0],
                "direct_fk_role_delta": [pd.NA],
                "role_evidence_matches": [3],
                "role_evidence_confidence": [0.8],
                "role_last_observed_at": ["2026-08-16T12:00:00Z"],
            }
        )

        result = derive_preseason_role_evidence(evidence)

        self.assertAlmostEqual(float(result.loc[0, "role_change_score"]), 0.8)
        self.assertAlmostEqual(float(result.loc[0, "role_evidence_quality"]), 0.8)
        self.assertEqual(int(result.loc[0, "role_evidence_components"]), 3)

    def test_role_and_workload_validation_rejects_unsafe_values(self) -> None:
        bad_role = pd.DataFrame(
            {
                "player_code": [1001],
                "role_target_gameweek": [1],
                "attacking_role_delta": [1.2],
                "role_evidence_matches": [2],
                "role_evidence_confidence": [0.8],
                "role_last_observed_at": ["2026-08-16T12:00:00Z"],
            }
        )
        with self.assertRaisesRegex(ValueError, "attacking_role_delta"):
            derive_preseason_role_evidence(bad_role)

        bad_workload = pd.DataFrame(
            {
                "player_code": [1001],
                "tournament_minutes": [300],
                "tournament_matches": [3],
                "tournament_starts": [3],
                "readiness_multiplier": [1.1],
                "workload_evidence_confidence": [0.8],
                "last_tournament_appearance_at": ["2026-07-19T20:00:00Z"],
                "workload_valid_through_gameweek": [1],
            }
        )
        with self.assertRaisesRegex(ValueError, "readiness_multiplier"):
            normalize_tournament_workload(bad_workload)

        no_expiry = bad_workload.assign(
            tournament_starts=3,
            readiness_multiplier=0.9,
        ).drop(columns="workload_valid_through_gameweek")
        with self.assertRaisesRegex(ValueError, "workload_valid_through_gameweek"):
            normalize_tournament_workload(no_expiry)

    def test_usage_snapshot_cannot_supply_readiness_multiplier(self) -> None:
        usage = pd.DataFrame(
            {
                "player_code": [1001],
                "team_preseason_matches": [4],
                "preseason_appearances": [4],
                "preseason_starts": [4],
                "preseason_minutes": [320],
                "preseason_last_two_minutes": [170],
                "source_confidence": [0.8],
                "readiness_multiplier": [0.7],
            }
        )

        with TemporaryDirectory() as tmpdir:
            catalog = SnapshotCatalog(Path(tmpdir))
            catalog.publish_frame(
                usage,
                season_name="2026-27",
                topic="preseason_player_usage",
                provider="manual",
                captured_at="2026-08-16T12:00:00Z",
                source_url="https://example.test/usage",
                rights_class="local_use_only",
            )
            with self.assertRaisesRegex(ValueError, "readiness_multiplier"):
                load_preseason_snapshot_context(
                    catalog,
                    season_name="2026-27",
                    cutoff="2026-08-20T12:00:00Z",
                    target_gameweek=1,
                )

    def test_workload_normalizer_discards_blank_template_rows(self) -> None:
        workload = pd.DataFrame(
            {
                "player_code": [1001, 1002],
                "tournament_minutes": [420, pd.NA],
                "tournament_matches": [5, pd.NA],
                "tournament_starts": [5, pd.NA],
                "last_tournament_appearance_at": ["2026-07-19T20:00:00Z", pd.NA],
                "readiness_multiplier": [0.8, pd.NA],
                "workload_evidence_confidence": [0.9, pd.NA],
                "workload_valid_through_gameweek": [1, 1],
            }
        )

        result = normalize_tournament_workload(workload)

        self.assertEqual(result["player_code"].tolist(), [1001])


class PreseasonTemplateTests(unittest.TestCase):
    def test_templates_include_stable_team_identity_and_topic_specific_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            bootstrap_path = Path(tmpdir) / "bootstrap.json"
            bootstrap_path.write_text(
                json.dumps(
                    {
                        "teams": [{"id": 1, "code": 10, "name": "Alpha"}],
                        "elements": [
                            {
                                "id": 7,
                                "code": 1001,
                                "first_name": "Test",
                                "second_name": "Player",
                                "team": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def args(name: str) -> object:
                return type(
                    "Args",
                    (),
                    {
                        "bootstrap": str(bootstrap_path),
                        "output": str(Path(tmpdir) / name),
                        "target_gameweek": 1,
                    },
                )()

            usage = pd.read_csv(create_player_usage_template(args("usage.csv")))
            role = pd.read_csv(create_role_evidence_template(args("role.csv")))
            workload = pd.read_csv(create_workload_template(args("workload.csv")))

        self.assertEqual(int(usage.loc[0, "team_code"]), 10)
        self.assertIn("attacking_role_delta", role.columns)
        self.assertEqual(int(role.loc[0, "role_target_gameweek"]), 1)
        self.assertIn("workload_valid_through_gameweek", workload.columns)
        self.assertIn("workload_evidence_confidence", workload.columns)

    def test_template_rejects_duplicate_bootstrap_player_codes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            bootstrap_path = Path(tmpdir) / "bootstrap.json"
            bootstrap_path.write_text(
                json.dumps(
                    {
                        "teams": [{"id": 1, "code": 10, "name": "Alpha"}],
                        "elements": [
                            {
                                "id": player_id,
                                "code": 1001,
                                "first_name": "Test",
                                "second_name": str(player_id),
                                "team": 1,
                            }
                            for player_id in (7, 8)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "bootstrap": str(bootstrap_path),
                    "output": str(Path(tmpdir) / "usage.csv"),
                },
            )()

            with self.assertRaisesRegex(ValueError, "unique player_code"):
                create_player_usage_template(args)


class TeamPriorTests(unittest.TestCase):
    def test_builds_last_five_team_features_keyed_by_stable_team_code(self) -> None:
        results = pd.DataFrame(
            [
                {"Date": f"0{idx + 1}/05/2026", "HomeTeam": "Coventry", "AwayTeam": "Hull", "FTHG": idx + 1, "FTAG": 1}
                for idx in range(5)
            ]
        )
        teams = pd.DataFrame(
            [
                {"team_id": 7, "code": 9, "name": "Coventry City"},
                {"team_id": 11, "code": 88, "name": "Hull City"},
            ]
        )

        priors = build_team_priors_from_results(
            results,
            teams,
            source_league="E1",
            source_label="football_data_co_uk:2025-26",
            window=5,
        ).set_index("team_code")

        self.assertEqual(set(priors.index), {9, 88})
        self.assertAlmostEqual(float(priors.loc[9, "team_goals_for_ma5"]), 3.0)
        self.assertAlmostEqual(float(priors.loc[88, "team_goals_against_ma5"]), 3.0)
        self.assertEqual(priors.loc[9, "team_prior_source"], "football_data_co_uk:2025-26:E1")


class TeamPriorSnapshotScriptTests(unittest.TestCase):
    def test_validates_prior_season_urls(self) -> None:
        self.assertEqual(
            _validated_football_data_url(None, prior_season="2025-26", league="E0"),
            _default_football_data_url("2025-26", "E0"),
        )
        with self.assertRaisesRegex(ValueError, "prior season 2024-25"):
            _validated_football_data_url(
                "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
                prior_season="2024-25",
                league="E0",
            )

    def test_build_team_snapshot_uses_prior_season_derived_urls(self) -> None:
        with TemporaryDirectory() as tmpdir:
            bootstrap_path = Path(tmpdir) / "bootstrap-static.json"
            bootstrap = {
                "teams": (
                    [{"id": 1, "code": 9, "name": "Coventry City"}]
                    + [{"id": 2, "code": 88, "name": "Hull City"}]
                    + [
                        {"id": team_id, "code": 100 + team_id, "name": f"Club {team_id}"}
                        for team_id in range(3, 21)
                    ]
                )
            }
            bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "bootstrap": str(bootstrap_path),
                    "epl_url": None,
                    "championship_url": None,
                    "timeout": 5.0,
                    "prior_season": "2025-26",
                    "season": "2026-27",
                    "snapshot_root": tmpdir,
                    "captured_at": "2026-08-01T12:00:00Z",
                },
            )()
            csv_payload = (
                "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
                "E1,01/05/2026,Coventry,Hull,1,0,H\n"
                "E1,02/05/2026,Hull,Coventry,0,2,A\n"
                "E1,03/05/2026,Coventry,Hull,3,1,H\n"
                "E1,04/05/2026,Hull,Coventry,0,1,A\n"
                "E1,05/05/2026,Coventry,Hull,2,0,H\n"
            ).encode("utf-8")

            class FakeResponse:
                def __init__(self, content: bytes) -> None:
                    self.content = content

                def raise_for_status(self) -> None:
                    return None

            fake_priors_e0 = pd.DataFrame(
                {
                    "team_id": list(range(1, 11)),
                    "team_code": [9, 88] + [100 + team_id for team_id in range(3, 11)],
                    "team_goals_for_ma5": [1.0] * 10,
                    "team_goals_against_ma5": [1.0] * 10,
                    "team_prior_source": ["manual"] * 10,
                }
            )
            fake_priors_e1 = pd.DataFrame(
                {
                    "team_id": list(range(11, 21)),
                    "team_code": [100 + team_id for team_id in range(11, 21)],
                    "team_goals_for_ma5": [1.0] * 10,
                    "team_goals_against_ma5": [1.0] * 10,
                    "team_prior_source": ["manual"] * 10,
                }
            )

            with (
                patch("scripts.update_preseason_snapshots.requests.get", return_value=FakeResponse(csv_payload)) as mocked_get,
                patch(
                    "scripts.update_preseason_snapshots.build_team_priors_from_results",
                    side_effect=[fake_priors_e0, fake_priors_e1],
                ),
            ):
                snapshot_path = build_team_prior_snapshot(args)

            called_urls = [call.args[0] for call in mocked_get.call_args_list]
            self.assertEqual(
                called_urls,
                [
                    "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
                    "https://www.football-data.co.uk/mmz4281/2526/E1.csv",
                ],
            )
            self.assertTrue(Path(snapshot_path).exists())


if __name__ == "__main__":
    unittest.main()
