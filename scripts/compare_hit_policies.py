"""Compare the frozen pre-fix v2 and revised hit policy from the same GW4 state.

This is sensitivity analysis on already observed gameweeks, not a new holdout.
No defaults are selected by the best realized score.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

from fplmodel import config
from fplmodel.manager_backtest import historical_competition
from fplmodel.season_manager import ManagerState, SeasonManagerConfig
from scripts.backtest_current_season import audit_origin, actual_outcomes, digest, load_reconstruction


def load_before_policy(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    source = directory / "manager_v2.py"
    expected = manifest["source_sha256"]["fplmodel/manager_v2.py"]
    if digest(source) != expected:
        raise ValueError("Frozen pre-fix policy hash mismatch")
    name = "fplmodel._before_paid_transfer_fix"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.run_v2_simulations, expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--previous-report", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[90, 91, 92])
    parser.add_argument("--buffers", type=float, nargs="+", default=[1., 2., 3.])
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--reuse-experiments", type=Path,
                        help="Reuse completed frozen-before cases with the same state and source hash")
    args = parser.parse_args(argv)
    before, source_hash = load_before_policy(args.baseline)
    previous = json.loads(args.previous_report.read_text())
    if previous["engine_sha256"]["manager_v2.py"] != source_hash:
        raise ValueError("Previous report does not match the frozen pre-fix policy")
    initial = ManagerState.from_dict(previous["managers"]["v2"]["weekly"][3]["decision"]["manager_state_after"])
    if initial.last_processed_gameweek != 4:
        raise ValueError("Controlled comparison requires state through GW4")
    baseline_config = SeasonManagerConfig()
    revised_config = SeasonManagerConfig(manager_version="v2", max_transfers_per_gw=5)
    archive = config.OUTPUTS_DIR / "seasons" / initial.season_name
    audits = {gw: audit_origin(archive, gw) for gw in range(5, 9)}
    stored = {gw: pd.read_csv(audit["paths"]["predictions"]) for gw, audit in audits.items()}
    vintages = {gw: load_reconstruction(args.reconstruction, audit, stored[gw], 4) for gw, audit in audits.items()}
    actuals = {gw: actual_outcomes(frame, gw, config.RAW_DIR) for gw, frame in stored.items()}
    directory = config.OUTPUTS_DIR / "manager_v2" / "hit_policy_experiments" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    protected = {str(p): digest(p) for p in list(config.MODELS_DIR.glob("*"))
                 + list(config.OUTPUTS_DIR.glob("manager_state_*.json")) + list(archive.glob("*")) if p.is_file()}
    (directory / "protocol.json").write_text(json.dumps({
        "initial_state": initial.to_dict(), "gameweeks": [5, 6, 7, 8],
        "seeds": args.seeds, "buffers": args.buffers, "simulations_per_phase": args.simulations,
        "before_source_sha256": source_hash,
        "after_source_sha256": digest(config.PROJECT_ROOT / "fplmodel" / "manager_v2.py"),
        "before_report": str(args.previous_report), "manager_config": asdict(revised_config),
        "reconstruction": str(args.reconstruction),
        "note": "Observed-period sensitivity from a common state. Defaults remain buffer=2 and win probability=.75 regardless of realized scores."
    }, indent=2))
    print(f"Controlled GW5–8 comparison from identical state. Reports: {directory}", flush=True)
    jobs = [("before", seed, 2.) for seed in args.seeds]
    jobs += [("after", seed, 2.) for seed in args.seeds]
    jobs += [("after", args.seeds[0], value) for value in args.buffers if value != 2.]
    rows = []
    for label, seed, buffer in jobs:
        print(f"Starting {label}, seed={seed}, gain buffer={buffer}", flush=True)

        def progress(event):
            if event["manager"] == "v2" and event["phase"] == "scored":
                print(f"  GW{event['gameweek']}: {event['actual_points']:.0f} points, {event['transfers']} transfers", flush=True)

        reuse = None
        if args.reuse_experiments and label == "before":
            prior = json.loads((args.reuse_experiments / "protocol.json").read_text())
            candidate_path = args.reuse_experiments / f"{label}_seed{seed}_buffer{buffer:g}.json"
            if (candidate_path.is_file() and prior["before_source_sha256"] == source_hash
                    and prior["initial_state"] == initial.to_dict()
                    and prior["simulations_per_phase"] == args.simulations):
                reuse = json.loads(candidate_path.read_text())
        result = reuse or historical_competition(vintages, actuals, initial_state=initial,
            upgraded_config=replace(revised_config, paid_transfer_gain_buffer=buffer),
            legacy_config=baseline_config, simulations=args.simulations, seed=seed,
            allow_reconstructed=True, progress_callback=progress,
            v2_runner=before if label == "before" else None)
        manager = result["managers"]["v2"]
        row = {"policy": label, "seed": seed, "gain_buffer": buffer if label == "after" else None,
               "net_points": manager["net_actual_points"], "hit_points": manager["transfer_hit_cost"],
               "transfers": manager["transfers"]}
        rows.append(row)
        with (directory / f"{label}_seed{seed}_buffer{buffer:g}.json").open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        pd.DataFrame(rows).to_csv(directory / "summary.csv", index=False)
        print("Completed", row, flush=True)
    if any(digest(path) != old for path, old in protected.items()):
        raise RuntimeError("Protected inputs changed during comparison")
    (directory / "preservation.json").write_text(json.dumps({"files_checked": len(protected), "unchanged": True}, indent=2))
    print(f"Complete: {directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
