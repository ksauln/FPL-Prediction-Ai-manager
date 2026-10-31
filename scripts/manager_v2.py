"""Offline commands for the read-only upgraded manager.

Run: python -m scripts.manager_v2 --help
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from fplmodel.config import MODELS_DIR, OUTPUTS_DIR, RAW_DIR
from fplmodel.manager_backtest import historical_competition, load_vintage
from fplmodel.manager_forecasts import build_cached_forecasts
from fplmodel.manager_v2 import run_v2_simulations
from fplmodel.season_manager import SeasonManagerConfig, load_manager_state, load_prediction_files


def _report(payload, label):
    directory = OUTPUTS_DIR / "manager_v2" / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{label}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    forecast = commands.add_parser("forecast", help="Build individual-fixture projections from cached inputs and fitted models")
    forecast.add_argument("--start-gw", type=int, required=True)
    forecast.add_argument("--end-gw", type=int, required=True)
    forecast.add_argument("--model-context", type=Path, required=True)
    forecast.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    forecast.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    trial = commands.add_parser("trial", help="Compare actions without committing live state")
    trial.add_argument("--predictions", type=Path, required=True)
    trial.add_argument("--state", type=Path, required=True)
    trial.add_argument("--end-gw", type=int)
    audit = commands.add_parser("audit", help="Check whether saved vintage manifests support a historical competition")
    audit.add_argument("--vintages", type=Path, required=True)
    backtest = commands.add_parser("backtest", help="Run both managers and a hold baseline on deadline-safe vintages")
    backtest.add_argument("--vintages", type=Path, required=True)
    backtest.add_argument("--actuals", type=Path, required=True,
                          help='JSON: {"GW": {"player_id": [appeared, points]}}; DNPs must be explicit')
    backtest.add_argument("--state", type=Path, required=True)
    backtest.add_argument("--legacy-request", type=Path,
                          help="Original job request.json, to reproduce saved legacy strategy settings")
    for command in (trial, backtest):
        command.add_argument("--simulations", type=int, default=128)
        command.add_argument("--seed", type=int, default=90)
        command.add_argument("--horizon", type=int, default=4)
        command.add_argument("--beam-width", type=int, default=4)
        command.add_argument("--max-transfers", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "forecast":
        destination = build_cached_forecasts(raw_dir=args.raw_dir, models_dir=args.models_dir,
            output_dir=OUTPUTS_DIR / "manager_v2" / "forecasts",
            start_gw=args.start_gw, end_gw=args.end_gw, model_context=args.model_context)
        print(f"Created individual-fixture forecasts: {destination}")
        return 0
    if args.command in {"backtest", "audit"}:
        paths = sorted(args.vintages.rglob("vintage.json"))
        valid, rejected = {}, []
        for path in paths:
            try:
                vintage = load_vintage(path)
                # Keep the latest deadline-safe vintage for each origin.
                previous = valid.get(vintage.origin_gw)
                if previous is None or vintage.created_at > previous.created_at:
                    valid[vintage.origin_gw] = vintage
            except (ValueError, KeyError, OSError) as exc:
                rejected.append({"path": str(path), "reason": str(exc)})
        if args.command == "audit":
            payload = {"valid_gameweeks": sorted(valid), "rejected": rejected,
                       "manifests_found": len(paths),
                       "note": "Unversioned prediction CSVs cannot prove pre-deadline provenance."}
            path = _report(payload, "vintage_audit")
            print(json.dumps(payload, indent=2))
            print(f"Report: {path}")
            return 0
        if not valid:
            parser.error("No verified pre-deadline vintage manifests found. Generate and retain vintages prospectively.")
    config = SeasonManagerConfig(manager_version="v2", max_transfers_per_gw=args.max_transfers,
        transfer_horizon=args.horizon, chip_lookahead=args.horizon, planning_beam_width=args.beam_width)
    state = load_manager_state(args.state)
    if args.command == "trial":
        data = load_prediction_files(args.predictions, start_gw=state.last_processed_gameweek + 1,
                                     end_gw=args.end_gw, expected_season_name=state.season_name)
        payload = run_v2_simulations(data, config=config, initial_state=state,
            simulations=args.simulations, random_seed=args.seed,
            progress_callback=lambda e: print(f"{e.get('phase', 'planning')}: GW{e.get('current_gameweek', '')}", flush=True))
    else:
        actuals = {int(gw): {int(pid): (value[0], float(value[1])) for pid, value in rows.items()}
                   for gw, rows in json.loads(args.actuals.read_text()).items()}
        baseline = None
        if args.legacy_request:
            from fplmodel.manager_jobs import _build_manager_config
            baseline = _build_manager_config(json.loads(args.legacy_request.read_text()))
        payload = historical_competition(valid, actuals, initial_state=state,
            upgraded_config=config, legacy_config=baseline, simulations=args.simulations, seed=args.seed)
    path = _report(payload, args.command)
    print(f"Saved {args.command} result: {path}. Live state was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
