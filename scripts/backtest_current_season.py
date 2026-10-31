"""Evaluate retained season forecasts, with explicitly reconstructed lookahead.

Run with ``python -m scripts.backtest_current_season --help``. All writes are
under outputs/manager_v2; original archives, models and live state are read only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from fplmodel import config
from fplmodel.data_cleaning import normalize_bootstrap
from fplmodel.external_history import load_external_histories
from fplmodel.features import build_training_and_pred_frames
from fplmodel.history_universe import build_history_universe
from fplmodel.manager_backtest import ForecastVintage, ReconstructedVintage, historical_competition
from fplmodel.manager_forecasts import forecast_gameweek
from fplmodel.model import predict_expected_points
from fplmodel.preseason import load_preseason_snapshot_context
from fplmodel.season_manager import ManagerState, SeasonManagerConfig, _records_to_squad_ids, pick_best_xi
from fplmodel.snapshot_catalog import SnapshotCatalog
from fplmodel.state import ModelState


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_origin(archive, gw):
    archive = Path(archive)
    paths = {name: archive / f"{stem}_gw{gw}.{suffix}" for name, stem, suffix in (
        ("predictions", "predictions", "csv"), ("bootstrap", "bootstrap", "json"),
        ("fixtures", "fixtures", "json"), ("context", "context", "json"))}
    bootstrap = json.loads(paths["bootstrap"].read_text())
    context = json.loads(paths["context"].read_text())
    deadline = pd.Timestamp(next(e["deadline_time"] for e in bootstrap["events"] if e["id"] == gw))
    generated = datetime.fromtimestamp(paths["predictions"].stat().st_mtime, timezone.utc)
    if generated >= deadline or int(context["last_finished_gw"]) >= gw:
        raise ValueError(f"GW{gw} archive is post-deadline or trained on evaluation results")
    records = context["reproducibility"]["files"]
    for name in ("ensemble_models.joblib", "model_state.json", "source_features.py", "source_model.py"):
        record = records[name]
        source = archive / record["archive_path"]
        if digest(source) != record["sha256"]:
            raise ValueError(f"GW{gw} archived {name} hash mismatch")
        if datetime.fromtimestamp(source.stat().st_mtime, timezone.utc) >= deadline:
            raise ValueError(f"GW{gw} {name} was saved after the deadline")
    if digest(config.PROJECT_ROOT / "fplmodel" / "features.py") != records["source_features.py"]["sha256"]:
        raise ValueError("Reconstruction requires the original feature implementation")
    # Corroborate the CSV filesystem timestamp with its original pipeline log.
    log_evidence = None
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*Predictions saved to .*predictions_gw" + str(gw) + r"\.csv")
    for log in sorted(config.LOGS_DIR.glob(f"run_gw{gw}_*.log")):
        for line in log.read_text(errors="replace").splitlines():
            match = pattern.match(line)
            if match:
                observed = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=ZoneInfo("America/Chicago"))
                if abs((observed - generated).total_seconds()) < 3:
                    log_evidence = str(log)
    if log_evidence is None:
        raise ValueError(f"GW{gw} forecast time could not be corroborated by its original run log")
    return {"gameweek": gw, "deadline": deadline.isoformat(),
        "original_generated_at": generated.isoformat(), "log": log_evidence,
        "trained_through_gw": int(context["last_finished_gw"]),
        "sha256": {str(p): digest(p) for p in paths.values()},
        "context": context, "paths": paths}


def actual_outcomes(predictions, gw, raw_dir):
    outcomes = {}
    for pid in predictions.player_id.astype(int):
        path = Path(raw_dir) / f"player_{pid}.json"
        matches = [r for r in json.loads(path.read_text())["history"] if int(r["round"]) == gw]
        if not matches:
            raise ValueError(f"Missing explicit actual history for GW{gw} player {pid}")
        fixtures = [r["fixture"] for r in matches]
        if len(fixtures) != len(set(fixtures)):
            raise ValueError("Duplicate actual fixture rows")
        outcomes[pid] = (any(float(r["minutes"]) > 0 for r in matches),
                         float(sum(float(r["total_points"]) for r in matches)))
    return outcomes


def retarget_features(base, fixture, gw, origin):
    """Reuse frozen player form; replace every match-dependent model input."""
    clubs = {int(fixture["team_h"]), int(fixture["team_a"])}
    result = base[base.team_id.isin(clubs)].copy()
    own_columns = [c for c in base if c.startswith("team_") and f"opp_{c}" in base]
    context = base.drop_duplicates("team_id").set_index("team_id")
    for team in clubs:
        opponent = next(c for c in clubs if c != team)
        mask = result.team_id.eq(team)
        home = team == int(fixture["team_h"])
        result.loc[mask, "was_home"] = float(home)
        result.loc[mask, "fixture_difficulty"] = float(fixture.get("team_h_difficulty" if home else "team_a_difficulty", 3.))
        for column in own_columns:
            result.loc[mask, f"opp_{column}"] = context.loc[opponent, column]
        if "opponent_team_prior_source" in result and "team_prior_source" in context:
            result.loc[mask, "opponent_team_prior_source"] = context.loc[opponent, "team_prior_source"]
        if "opponent_team_context_prior_weight" in result:
            result.loc[mask, "opponent_team_context_prior_weight"] = context.loc[opponent, "team_context_prior_weight"]
    result["fixture_count"] = 1
    result["target_fixture_kickoff"] = pd.to_datetime(fixture.get("kickoff_time"), utc=True).tz_localize(None)
    if gw != origin:
        result["official_ep_next"] = np.nan
        persistent = pd.to_numeric(result.get("status_persistent_flag", 0), errors="coerce").eq(1)
        returns = pd.to_datetime(result.get("status_return_date"), utc=True, errors="coerce")
        kickoff = pd.to_datetime(fixture.get("kickoff_time"), utc=True)
        known_absence = persistent | (returns.notna() & (returns > kickoff))
        for column in ("availability_this_round", "availability_next_round", "status_availability"):
            if column in result:
                result.loc[~known_absence, column] = 1.
        for column in ("status_injury_flag", "status_persistent_flag", "injury_risk_flag", "future_status_applied_flag"):
            if column in result:
                result.loc[~known_absence, column] = 0.
    return result


def reconstruct_origin(audit, source, external, *, horizon, raw_dir, destination):
    """Rebuild lookahead using that origin's model, bootstrap, bias and prior data.

    The origin's next-GW forecast stays exactly as retained. Newly reconstructed
    future forecasts use per-fixture opponents/venues and freeze prices/identity.
    """
    gw, paths, context = audit["gameweek"], audit["paths"], audit["context"]
    bootstrap = json.loads(paths["bootstrap"].read_text())
    norms = normalize_bootstrap(bootstrap)
    fixtures = pd.DataFrame(json.loads(paths["fixtures"].read_text()))
    rows = []
    for pid in norms["elements"].player_id.astype(int):
        for match in json.loads((Path(raw_dir) / f"player_{pid}.json").read_text())["history"]:
            if int(match["round"]) < gw and pd.Timestamp(match["kickoff_time"]) < pd.Timestamp(audit["deadline"]):
                rows.append({**match, "player_id": pid, "current_player_id": pid, "season_name": context["season_name"]})
    prior = build_history_universe(external, norms["elements"])
    history = pd.concat([prior, pd.DataFrame(rows)], ignore_index=True, sort=False)
    snapshot = load_preseason_snapshot_context(SnapshotCatalog(config.PRESEASON_SNAPSHOT_DIR),
        season_name=context["season_name"], cutoff=audit["original_generated_at"],
        target_gameweek=gw, mode="auto")
    archive = paths["context"].parent
    records = context["reproducibility"]["files"]
    models = joblib.load(archive / records["ensemble_models.joblib"]["archive_path"])
    bias = ModelState(archive / records["model_state.json"]["archive_path"], season_name=context["season_name"])
    if bias.last_evaluated_gw >= gw:
        raise ValueError("Reconstruction bias includes evaluation-week results")
    _, _, base, _ = build_training_and_pred_frames(
        norms["elements"], norms["teams"], history,
        next_gw=gw, last_finished_gw=gw - 1, state=bias,
        fixtures_df=fixtures, current_season_name=context["season_name"], official_next_gw=gw,
        preseason_player_df=snapshot.player_enrichment, team_priors_df=snapshot.team_priors)
    (destination / f"features_origin_gw{gw}.csv").write_text(base.to_csv(index=False))

    def predict(single, target_gw):
        feature_rows = retarget_features(base, single.iloc[0].to_dict(), target_gw, gw)
        forecasts = [predict_expected_points(feature_rows, m.classifier, m.regressor, bias,
                     appearance_clf=m.appearance_classifier,
                     cameo_points_by_position=m.cameo_points_by_position) for m in models]
        result = forecasts[0].copy()
        ep = np.stack([f.expected_points.to_numpy() for f in forecasts])
        result["expected_points"] = ep.mean(axis=0)
        gate = pd.to_numeric(result.availability_next_round, errors="coerce").fillna(1.).clip(0, 1)
        readiness = pd.to_numeric(result.get("readiness_multiplier", pd.Series(1., index=result.index)), errors="coerce").fillna(1.).clip(0, 1)
        result["start_probability"] = np.mean([f.p_start.to_numpy() for f in forecasts], axis=0) * gate * readiness
        result["appearance_probability"] = np.mean([f.p_appearance.to_numpy() for f in forecasts], axis=0) * gate * readiness
        result["cameo_points"] = np.mean([result.element_type.map(m.cameo_points_by_position).fillna(1.).to_numpy() for m in models], axis=0)
        sigma = ep.std(axis=0) + 1.
        result["expected_points_lower_80"] = np.maximum(0., result.expected_points - 1.2815515655 * sigma)
        result["expected_points_upper_80"] = result.expected_points + 1.2815515655 * sigma
        return result

    data = {gw: source.copy()}
    for target in range(gw + 1, min(38, gw + horizon - 1) + 1):
        data[target] = forecast_gameweek(source, fixtures, target, predict)
        data[target]["season_name"] = context["season_name"]
        # Per-fixture probabilities already include availability/readiness.
        # Neutral metadata lets the legacy engine consume the same probability
        # representation without multiplying the original status gate twice.
        for column in ("availability_this_round", "availability_next_round", "status_availability"):
            data[target][column] = 1.
    folder = destination / f"origin_gw{gw}"
    folder.mkdir()
    for target, frame in data.items():
        frame.to_csv(folder / f"predictions_gw{target}.csv", index=False)
    rebuilt = datetime.now(timezone.utc).isoformat()
    (folder / "reconstruction.json").write_text(json.dumps({
        "origin_gw": gw, "reconstructed_at": rebuilt, "source_generated_at": audit["original_generated_at"],
        "history_through_gw": gw - 1, "target_gameweeks": sorted(data),
        "model_sha256": records["ensemble_models.joblib"]["sha256"],
        "note": "Origin forecast is original. Future horizons are reconstructed now from frozen historical models and past-only data."}, indent=2))
    return ReconstructedVintage(origin_gw=gw, season_name=context["season_name"],
        created_at=audit["original_generated_at"], inputs_as_of=audit["original_generated_at"],
        trained_through_gw=audit["trained_through_gw"], predictions=data, deadline=audit["deadline"],
        reconstructed_at=rebuilt, history_through_gw=gw - 1)


def load_reconstruction(root, audit, source, horizon):
    folder = Path(root) / f"origin_gw{audit['gameweek']}"
    manifest = json.loads((folder / "reconstruction.json").read_text())
    gw = audit["gameweek"]
    expected_model = audit["context"]["reproducibility"]["files"]["ensemble_models.joblib"]["sha256"]
    if (manifest["origin_gw"] != gw or manifest["history_through_gw"] >= gw
            or manifest["model_sha256"] != expected_model):
        raise ValueError("Cached reconstruction does not match the archived model/cutoff")
    data = {gw: source.copy()}
    for target in range(gw + 1, min(38, gw + horizon - 1) + 1):
        frame = pd.read_csv(folder / f"predictions_gw{target}.csv")
        for column in ("availability_this_round", "availability_next_round", "status_availability"):
            frame[column] = 1.
        data[target] = frame
    result = ReconstructedVintage(gw, audit["context"]["season_name"], audit["original_generated_at"],
        audit["original_generated_at"], audit["trained_through_gw"], data, audit["deadline"],
        reconstructed_at=manifest["reconstructed_at"], history_through_gw=manifest["history_through_gw"])
    result.validate()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--end-gw", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=90)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--archived-only", action="store_true")
    parser.add_argument("--reuse-reconstruction", type=Path,
                        help="Reuse frozen-model forecast outputs from a previous audited run")
    args = parser.parse_args(argv)
    archive = config.OUTPUTS_DIR / "seasons" / args.season
    destination = config.OUTPUTS_DIR / "manager_v2" / "season_backtests" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True)
    protected = list(config.MODELS_DIR.glob("*")) + list(config.OUTPUTS_DIR.glob("manager_state_*.json")) + list(archive.glob("*"))
    protected = {str(p): digest(p) for p in protected if p.is_file()}
    audits = {gw: audit_origin(archive, gw) for gw in range(1, args.end_gw + 1)}
    stored = {gw: pd.read_csv(a["paths"]["predictions"]) for gw, a in audits.items()}
    actuals = {gw: actual_outcomes(frame, gw, config.RAW_DIR) for gw, frame in stored.items()}
    initial = pick_best_xi(stored[1], budget_m=100., formations=config.FORMATION_OPTIONS)
    ids = _records_to_squad_ids(initial)
    prices = stored[1].set_index("player_id").now_cost_millions.to_dict()
    start = ManagerState(ids, round(100. - sum(prices[p] for p in ids), 2), 0,
                         {p: prices[p] for p in ids}, last_processed_gameweek=0, season_name=args.season)
    (destination / "source_audit.json").write_text(json.dumps({
        gw: {k: v for k, v in audit.items() if k not in {"context", "paths"}} for gw, audit in audits.items()}, indent=2))
    print(f"Audited GW1–{args.end_gw}; all original forecast times corroborated by logs. Results: {destination}", flush=True)
    vintages = {gw: ForecastVintage(gw, args.season, a["original_generated_at"],
        a["original_generated_at"], a["trained_through_gw"], {gw: stored[gw]}, a["deadline"]) for gw, a in audits.items()}
    upgraded = SeasonManagerConfig(manager_version="v2", max_transfers_per_gw=5,
        transfer_horizon=args.horizon, chip_lookahead=args.horizon, planning_beam_width=args.beam_width)
    baseline = SeasonManagerConfig()

    def progress(event):
        if event["phase"] == "scored":
            print(f"GW{event['gameweek']} {event['manager']}: {event['actual_points']:.0f} actual points, "
                  f"{event['transfers']} transfers, chip={event['chip']}", flush=True)
        else:
            print(f"Planning GW{event['gameweek']} {event['manager']}...", flush=True)

    retained = historical_competition(vintages, actuals, initial_state=start,
        upgraded_config=upgraded, legacy_config=baseline, simulations=args.simulations,
        seed=args.seed, progress_callback=progress)
    retained["method"] = "Original archived next-week forecasts only; effective planning horizon is one GW."
    retained["provenance_note"] = "Source generation times inferred from preserved CSV timestamps and corroborated against original run logs, not original vintage manifests."
    (destination / "archived_one_week.json").write_text(json.dumps(retained, indent=2, allow_nan=False))
    print("ARCHIVED RESULT", {name: result["net_actual_points"] for name, result in retained["managers"].items()}, flush=True)
    if not args.archived_only:
        rebuilt = {}
        if args.reuse_reconstruction:
            for gw, audit in audits.items():
                rebuilt[gw] = load_reconstruction(args.reuse_reconstruction, audit, stored[gw], args.horizon)
        else:
            print("Loading completed prior-season history for frozen-model reconstruction...", flush=True)
            external = load_external_histories(config.EXTERNAL_HISTORY_SEASONS)
            for gw, audit in audits.items():
                print(f"Rebuilding GW{gw} lookahead using its own archived model and data through GW{gw - 1}...", flush=True)
                rebuilt[gw] = reconstruct_origin(audit, stored[gw], external, horizon=args.horizon,
                                                 raw_dir=config.RAW_DIR, destination=destination)
        result = historical_competition(rebuilt, actuals, initial_state=start,
            upgraded_config=upgraded, legacy_config=baseline, simulations=args.simulations,
            seed=args.seed, allow_reconstructed=True, progress_callback=progress)
        result["method"] = "Original next-week forecasts plus reconstructed per-fixture lookahead from each origin's archived model, metadata, bias, and past-only history."
        result["settings"] = {"horizon": args.horizon, "simulations_per_phase": args.simulations,
                              "seed": args.seed, "beam_width": args.beam_width,
                              "legacy_max_transfers": 2, "v2_max_transfers": 5}
        result["manager_configs"] = {"legacy": asdict(baseline), "v2": asdict(upgraded)}
        result["engine_sha256"] = {name: digest(config.PROJECT_ROOT / "fplmodel" / name)
            for name in ("manager_v2.py", "manager_backtest.py", "season_manager.py", "team_picker.py")}
        (destination / "reconstructed_four_week.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print("RECONSTRUCTED RESULT", {name: r["net_actual_points"] for name, r in result["managers"].items()}, flush=True)
    changed = [p for p, old in protected.items() if digest(p) != old]
    if changed:
        raise RuntimeError(f"Protected inputs changed during backtest: {changed}")
    (destination / "preservation.json").write_text(json.dumps({"files_checked": len(protected), "unchanged": True}, indent=2))
    print(f"Backtest complete; {len(protected)} protected files unchanged. Reports: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
