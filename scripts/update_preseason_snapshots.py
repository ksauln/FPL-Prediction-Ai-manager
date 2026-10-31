#!/usr/bin/env python3
"""Build private point-in-time preseason enrichment snapshots.

The default team-prior command downloads final prior-season league results from
Football-Data.co.uk, reduces them to last-five aggregate features, and stores
only the normalized local snapshot plus provenance. Player usage, tournament
workload, licensed projections, and odds are published from explicit local CSV
exports so credentials and heterogeneous raw evidence never enter the pipeline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.parse import parse_qsl, urlsplit
import re

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fplmodel.config import PRESEASON_SNAPSHOT_DIR, RAW_DIR  # noqa: E402
from fplmodel.preseason import derive_preseason_player_priors  # noqa: E402
from fplmodel.snapshot_catalog import SnapshotCatalog, _safe_source_url  # noqa: E402
from fplmodel.team_priors import build_team_priors_from_results  # noqa: E402


FOOTBALL_DATA_LISTING = "https://www.football-data.co.uk/englandm.php"
REQUIRED_RESULT_COLUMNS = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
PLAYER_TOPICS = {
    "preseason_player_usage",
    "preseason_role_evidence",
    "tournament_workload",
    "provider_expected_points",
}
ALL_IMPORT_TOPICS = PLAYER_TOPICS | {"fixture_odds", "team_priors"}


def _captured_at(value: str | None) -> str:
    if value:
        return value
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _season_code(prior_season: str) -> str:
    text = str(prior_season).strip()
    match = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if not match:
        raise ValueError("--prior-season must look like YYYY-YY, e.g. 2025-26")
    start_year = int(match.group(1))
    end_suffix = int(match.group(2))
    if (start_year + 1) % 100 != end_suffix:
        raise ValueError("--prior-season must describe adjacent seasons, e.g. 2025-26")
    return f"{start_year % 100:02d}{end_suffix:02d}"


def _default_football_data_url(prior_season: str, league: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{_season_code(prior_season)}/{league}.csv"


def _validated_football_data_url(url: str | None, *, prior_season: str, league: str) -> str:
    candidate = _safe_source_url(url or _default_football_data_url(prior_season, league))
    parsed = urlsplit(candidate)
    expected_suffix = f"/mmz4281/{_season_code(prior_season)}/{league}.csv"
    if parsed.netloc.casefold() != "www.football-data.co.uk" or parsed.path != expected_suffix:
        raise ValueError(
            f"{league} URL must match prior season {prior_season} at {expected_suffix}"
        )
    return candidate


def _load_bootstrap_teams(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        bootstrap = json.load(handle)
    teams = pd.DataFrame(bootstrap.get("teams", [])).rename(columns={"id": "team_id"})
    required = {"team_id", "code", "name"}
    if not required.issubset(teams.columns) or len(teams) != 20:
        raise ValueError(f"Bootstrap must contain 20 teams with {sorted(required)}")
    return teams


def _bootstrap_player_identity_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        bootstrap = json.load(handle)
    teams = {
        int(team["id"]): {
            "team_name": str(team["name"]),
            "team_code": int(team["code"]),
        }
        for team in bootstrap.get("teams", [])
    }
    rows: list[dict[str, object]] = []
    players = bootstrap.get("elements", [])
    for player in players:
        try:
            team = teams[int(player["team"])]
            rows.append(
                {
                    "player_code": int(player["code"]),
                    "player_id": int(player["id"]),
                    "full_name": f"{player['first_name']} {player['second_name']}",
                    "team_name": team["team_name"],
                    "team_code": team["team_code"],
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Bootstrap contains a malformed player identity") from exc
    if not rows:
        raise ValueError("Bootstrap contains no usable players")
    frame = pd.DataFrame(rows)
    if frame["player_code"].duplicated().any():
        raise ValueError("Bootstrap players must have unique player_code values")
    if frame["player_id"].duplicated().any():
        raise ValueError("Bootstrap players must have unique player_id values")
    return rows


def _write_template(rows: list[dict[str, object]], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["team_name", "full_name"]).to_csv(output, index=False)
    return output


def create_player_usage_template(args: argparse.Namespace) -> Path:
    bootstrap_path = Path(args.bootstrap)
    rows = [
        {
            **player,
            "team_preseason_matches": pd.NA,
            "preseason_appearances": pd.NA,
            "preseason_starts": pd.NA,
            "preseason_minutes": pd.NA,
            "preseason_last_two_minutes": pd.NA,
            "source_confidence": pd.NA,
            "notes": "",
        }
        for player in _bootstrap_player_identity_rows(bootstrap_path)
    ]
    return _write_template(rows, Path(args.output))


def create_role_evidence_template(args: argparse.Namespace) -> Path:
    bootstrap_path = Path(args.bootstrap)
    rows = [
        {
            **player,
            "role_target_gameweek": args.target_gameweek,
            "attacking_role_delta": pd.NA,
            "penalty_role_delta": pd.NA,
            "corner_role_delta": pd.NA,
            "direct_fk_role_delta": pd.NA,
            "role_evidence_matches": pd.NA,
            "role_evidence_confidence": pd.NA,
            "role_last_observed_at": pd.NA,
            "role_notes": "",
        }
        for player in _bootstrap_player_identity_rows(bootstrap_path)
    ]
    return _write_template(rows, Path(args.output))


def create_workload_template(args: argparse.Namespace) -> Path:
    bootstrap_path = Path(args.bootstrap)
    rows = [
        {
            **player,
            "tournament_minutes": pd.NA,
            "tournament_matches": pd.NA,
            "tournament_starts": pd.NA,
            "last_tournament_appearance_at": pd.NA,
            "readiness_multiplier": pd.NA,
            "workload_evidence_confidence": pd.NA,
            "workload_valid_through_gameweek": args.target_gameweek,
            "workload_notes": "",
        }
        for player in _bootstrap_player_identity_rows(bootstrap_path)
    ]
    return _write_template(rows, Path(args.output))


def _download_results(url: str, *, timeout: float) -> tuple[pd.DataFrame, str]:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "FPL-Prediction preseason snapshot builder"},
    )
    response.raise_for_status()
    payload = response.content
    frame = pd.read_csv(BytesIO(payload), encoding="utf-8-sig")
    missing = REQUIRED_RESULT_COLUMNS - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(
            f"Football-Data response at {url} is invalid; missing {sorted(missing)}"
        )
    return frame, hashlib.sha256(payload).hexdigest()


def build_team_prior_snapshot(args: argparse.Namespace) -> Path:
    teams = _load_bootstrap_teams(Path(args.bootstrap))
    epl_url = _validated_football_data_url(args.epl_url, prior_season=args.prior_season, league="E0")
    championship_url = _validated_football_data_url(
        args.championship_url,
        prior_season=args.prior_season,
        league="E1",
    )
    epl, epl_hash = _download_results(epl_url, timeout=args.timeout)
    championship, championship_hash = _download_results(
        championship_url, timeout=args.timeout
    )
    prior_season = args.prior_season
    frames = [
        build_team_priors_from_results(
            epl,
            teams,
            source_league="E0",
            source_label=f"football_data_co_uk:{prior_season}",
            window=5,
        ),
        build_team_priors_from_results(
            championship,
            teams,
            source_league="E1",
            source_label=f"football_data_co_uk:{prior_season}",
            window=5,
        ),
    ]
    priors = pd.concat(frames, ignore_index=True).drop_duplicates(
        "team_code", keep=False
    )
    expected_codes = set(pd.to_numeric(teams["code"], errors="coerce").dropna().astype(int))
    actual_codes = set(pd.to_numeric(priors["team_code"], errors="coerce").dropna().astype(int))
    missing_codes = expected_codes - actual_codes
    if missing_codes:
        missing_names = teams[teams["code"].isin(missing_codes)]["name"].tolist()
        raise ValueError(f"No prior result mapping for current teams: {missing_names}")
    priors = priors.sort_values("team_id").reset_index(drop=True)

    catalog = SnapshotCatalog(Path(args.snapshot_root))
    return catalog.publish_frame(
        priors,
        season_name=args.season,
        topic="team_priors",
        provider="football_data_co_uk",
        captured_at=_captured_at(args.captured_at),
        source_url=FOOTBALL_DATA_LISTING,
        rights_class="local_use_only",
        paid=False,
        schema_version="1.0.0",
        extra_manifest={
            "source_urls": [epl_url, championship_url],
            "source_sha256": {"E0": epl_hash, "E1": championship_hash},
            "license_url": "https://www.football-data.co.uk/data.php",
            "raw_payload_persisted": False,
            "prior_season": prior_season,
        },
    )


def publish_local_csv(args: argparse.Namespace) -> Path:
    topic = str(args.topic)
    if topic not in ALL_IMPORT_TOPICS:
        raise ValueError(f"Unsupported topic: {topic}")
    frame = pd.read_csv(args.input)
    if frame.empty:
        raise ValueError("Input snapshot is empty")
    if topic in PLAYER_TOPICS:
        if "player_code" not in frame.columns:
            raise ValueError(f"{topic} requires player_code")
        player_codes = pd.to_numeric(frame["player_code"], errors="coerce")
        if (
            player_codes.isna().any()
            or player_codes.duplicated().any()
            or player_codes.mod(1).ne(0).any()
        ):
            raise ValueError("player_code must be unique and integral")
    if topic == "preseason_player_usage":
        if (
            "readiness_multiplier" in frame.columns
            and frame["readiness_multiplier"].notna().any()
        ):
            raise ValueError(
                "preseason_player_usage cannot supply readiness_multiplier; "
                "publish tournament_workload instead"
            )
        frame = frame.drop(columns=["readiness_multiplier"], errors="ignore")
        frame = derive_preseason_player_priors(frame)
    elif topic == "preseason_role_evidence":
        from fplmodel.preseason import derive_preseason_role_evidence

        frame = derive_preseason_role_evidence(frame)
    elif topic == "tournament_workload":
        from fplmodel.preseason import normalize_tournament_workload

        frame = normalize_tournament_workload(frame)
    elif topic == "team_priors" and "team_code" not in frame.columns:
        raise ValueError("team_priors requires team_code")
    elif topic == "provider_expected_points":
        required = {"provider_expected_points", "gameweek"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                "provider_expected_points topic requires: " + ", ".join(sorted(required))
            )

    catalog = SnapshotCatalog(Path(args.snapshot_root))
    return catalog.publish_frame(
        frame,
        season_name=args.season,
        topic=topic,
        provider=args.provider,
        captured_at=_captured_at(args.captured_at),
        source_url=_safe_source_url(args.source_url),
        rights_class=args.rights_class,
        paid=args.paid,
        schema_version="1.0.0",
        extra_manifest={"raw_payload_persisted": False},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-root", default=str(PRESEASON_SNAPSHOT_DIR), help="Private snapshot root"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    team = subparsers.add_parser("team-priors", help="Build free prior-season team form")
    team.add_argument("--season", required=True, help="Target FPL season, e.g. 2026-27")
    team.add_argument("--prior-season", required=True, help="Source season, e.g. 2025-26")
    team.add_argument("--bootstrap", default=str(RAW_DIR / "bootstrap-static.json"))
    team.add_argument("--epl-url")
    team.add_argument("--championship-url")
    team.add_argument("--timeout", type=float, default=30.0)
    team.add_argument("--captured-at")
    team.set_defaults(handler=build_team_prior_snapshot)

    template = subparsers.add_parser(
        "player-template", help="Create a current-player friendly-usage entry template"
    )
    template.add_argument("--bootstrap", default=str(RAW_DIR / "bootstrap-static.json"))
    template.add_argument("--output", required=True)
    template.set_defaults(handler=create_player_usage_template)

    role_template = subparsers.add_parser(
        "role-template", help="Create a current-player role-evidence entry template"
    )
    role_template.add_argument("--bootstrap", default=str(RAW_DIR / "bootstrap-static.json"))
    role_template.add_argument("--output", required=True)
    role_template.add_argument("--target-gameweek", type=int, default=1)
    role_template.set_defaults(handler=create_role_evidence_template)

    workload_template = subparsers.add_parser(
        "workload-template", help="Create a current-player workload entry template"
    )
    workload_template.add_argument("--bootstrap", default=str(RAW_DIR / "bootstrap-static.json"))
    workload_template.add_argument("--output", required=True)
    workload_template.add_argument("--target-gameweek", type=int, default=1)
    workload_template.set_defaults(handler=create_workload_template)

    publish = subparsers.add_parser("publish-csv", help="Publish a normalized local export")
    publish.add_argument("--season", required=True)
    publish.add_argument("--topic", required=True, choices=sorted(ALL_IMPORT_TOPICS))
    publish.add_argument("--input", type=Path, required=True)
    publish.add_argument("--provider", required=True)
    publish.add_argument("--source-url", required=True)
    publish.add_argument("--rights-class", default="local_use_only")
    publish.add_argument("--paid", action="store_true")
    publish.add_argument("--captured-at")
    publish.set_defaults(handler=publish_local_csv)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = args.handler(args)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
