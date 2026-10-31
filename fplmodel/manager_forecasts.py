"""Per-fixture forecasts for v2, kept separate from the live pipeline outputs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def forecast_gameweek(players, fixtures, gameweek, predict_fixture):
    """Call the model separately for each match, then aggregate player totals.

    ``predict_fixture`` receives a one-row fixture frame and the target GW. It
    must rebuild opponent/venue features for that match and return player EP.
    Within-week any-appearance probabilities use an independence approximation;
    scenario evaluation retains fixture detail and correlated rotation.
    """
    if players.player_id.duplicated().any():
        raise ValueError("Duplicate player IDs in fixture forecast universe")
    rows = {int(p["player_id"]): {**p, "fixture_forecasts": []}
            for p in players.to_dict("records")}
    target = fixtures[pd.to_numeric(fixtures["event"], errors="coerce").eq(gameweek)]
    for _, fixture in target.sort_values("id").iterrows():
        single = pd.DataFrame([fixture])
        prediction = predict_fixture(single, gameweek)
        if prediction.player_id.duplicated().any():
            raise ValueError("Duplicate per-fixture prediction IDs")
        clubs = {int(fixture.team_h), int(fixture.team_a)}
        needed = {pid for pid, p in rows.items() if int(p["team_id"]) in clubs}
        if needed - set(prediction.player_id.astype(int)):
            raise ValueError("Missing per-fixture predictions for scheduled players")
        for player in prediction.to_dict("records"):
            pid = int(player["player_id"])
            if pid not in needed:
                continue
            points = float(player["expected_points"])
            if not np.isfinite(points):
                raise ValueError("Non-finite fixture expected points")
            home = int(rows[pid]["team_id"]) == int(fixture.team_h)
            detail = {
                "fixture_id": int(fixture.id), "was_home": home,
                "opponent_id": int(fixture.team_a if home else fixture.team_h),
                "expected_points": points,
                "start_probability": float(player.get("start_probability", 1.)),
                "appearance_probability": float(player.get("appearance_probability", 1.)),
                "cameo_points": float(player.get("cameo_points", 1.)),
                "expected_points_lower_80": float(player.get("expected_points_lower_80", max(0., points - 2.))),
                "expected_points_upper_80": float(player.get("expected_points_upper_80", points + 2.)),
            }
            rows[pid]["fixture_forecasts"].append(detail)
    for row in rows.values():
        detail = row["fixture_forecasts"]
        row["expected_points"] = sum(f["expected_points"] for f in detail)
        row["fixture_multiplier"] = len(detail)
        for column in ("appearance_probability", "start_probability"):
            row[column] = 1. - float(np.prod([1. - f[column] for f in detail]))
        # Conservative common-team uncertainty; do not imply independent matches.
        for column in ("expected_points_lower_80", "expected_points_upper_80"):
            row[column] = sum(f[column] for f in detail)
        row["gameweek"] = gameweek
        row["probabilities_include_availability"] = True
        row["fixture_forecasts"] = json.dumps(detail, allow_nan=False)
    return pd.DataFrame(rows.values())


def build_cached_forecasts(*, raw_dir, models_dir, output_dir, start_gw, end_gw, model_context=None):
    """Read cached data/models only; create a new immutable v2 forecast vintage.

    This deliberately never fetches, fits, updates bias state, or overwrites a
    previous forecast. Use the normal live pipeline first when fresh data is
    needed. Current cache snapshots are not retroactive backtest evidence.
    """
    import hashlib
    import joblib
    from .data_cleaning import normalize_bootstrap
    from .features import build_training_and_pred_frames
    from .model import predict_expected_points
    from .prediction_artifacts import infer_bootstrap_season_name
    from .state import ModelState
    from .utils import get_current_and_last_finished_gw

    raw_dir, models_dir, output_dir = map(Path, (raw_dir, models_dir, output_dir))
    if not 1 <= start_gw <= end_gw <= 38:
        raise ValueError("Expected 1 <= start_gw <= end_gw <= 38")
    bootstrap_path = raw_dir / "bootstrap-static.json"
    bootstrap = json.loads(bootstrap_path.read_text())
    season = infer_bootstrap_season_name(bootstrap_path)
    if not season:
        raise ValueError("Cannot identify cached season")
    events = bootstrap["events"]
    official_next, last_finished = get_current_and_last_finished_gw(pd.DataFrame(events))
    if official_next is None or start_gw < int(official_next):
        raise ValueError("Cached forecast builder only supports the upcoming/future gameweeks")
    norms = normalize_bootstrap(bootstrap)
    fixtures = pd.DataFrame(json.loads((raw_dir / "fixtures-all.json").read_text()))
    rows, sources = [], [bootstrap_path, raw_dir / "fixtures-all.json"]
    for pid in norms["elements"].player_id.astype(int):
        path = raw_dir / f"player_{pid}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached history: {path.name}")
        sources.append(path)
        for match in json.loads(path.read_text()).get("history", []):
            rows.append({**match, "player_id": pid, "current_player_id": pid, "season_name": season})
    histories = pd.DataFrame(rows)
    if not histories.empty:
        histories = histories[pd.to_numeric(histories["round"], errors="coerce") <= last_finished]
    model_path = models_dir / "ensemble_models.joblib"
    if model_context is None:
        raise ValueError("A matching archived model context is required to verify the training cutoff")
    model_context = Path(model_context)
    context = json.loads(model_context.read_text())
    expected_hash = context.get("reproducibility", {}).get("files", {}).get("ensemble_models.joblib", {}).get("sha256")
    if context.get("season_name") != season or int(context.get("last_finished_gw", 99)) > last_finished:
        raise ValueError("Archived model context season/training cutoff does not match cached inputs")
    if expected_hash != hashlib.sha256(model_path.read_bytes()).hexdigest():
        raise ValueError("Current models do not match the archived model context")
    models = joblib.load(model_path)
    sources.extend([model_path, models_dir / "state.json", model_context])
    bias = ModelState(models_dir / "state.json", season_name=season)
    if bias.last_evaluated_gw > last_finished:
        raise ValueError("Bias state includes future results")
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}

    def predict_fixture(single, gw):
        _, _, features, _ = build_training_and_pred_frames(
            norms["elements"], norms["teams"], histories,
            next_gw=gw, last_finished_gw=last_finished, state=bias,
            fixtures_df=single, current_season_name=season, official_next_gw=official_next,
        )
        forecasts = [predict_expected_points(features, m.classifier, m.regressor, bias,
                     appearance_clf=m.appearance_classifier,
                     cameo_points_by_position=m.cameo_points_by_position) for m in models]
        if not forecasts:
            raise ValueError("No fitted ensemble candidates available")
        result = forecasts[0].copy()
        points = np.stack([f.expected_points.to_numpy() for f in forecasts])
        result["expected_points"] = points.mean(axis=0)
        # Model EP already includes availability/readiness. Apply that same gate
        # to appearance probabilities once, so captain fallback is consistent.
        availability = pd.to_numeric(result.get("availability_next_round", pd.Series(1., index=result.index)), errors="coerce")
        availability = availability.fillna(pd.to_numeric(result.get("status_availability", pd.Series(1., index=result.index)), errors="coerce")).fillna(1.).clip(0, 1)
        readiness = pd.to_numeric(result.get("readiness_multiplier", pd.Series(1., index=result.index)), errors="coerce").fillna(1.).clip(0, 1)
        gate = availability.to_numpy() * readiness.to_numpy()
        result["start_probability"] = np.mean([f.p_start.to_numpy() for f in forecasts], axis=0) * gate
        result["appearance_probability"] = np.mean([f.p_appearance.to_numpy() for f in forecasts], axis=0) * gate
        result["cameo_points"] = np.mean([result.element_type.map(m.cameo_points_by_position).fillna(1.).to_numpy() for m in models], axis=0)
        sigma = points.std(axis=0) + 1.
        result["expected_points_lower_80"] = np.maximum(0, result.expected_points - 1.2815515655 * sigma)
        result["expected_points_upper_80"] = result.expected_points + 1.2815515655 * sigma
        return result

    now = datetime.now(timezone.utc)
    destination = output_dir / season / now.strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    metadata = norms["elements"].copy()
    teams = norms["teams"].set_index("team_id")["name"].to_dict()
    metadata["team_name"] = metadata.team_id.map(teams)
    if "now_cost_millions" not in metadata:
        metadata["now_cost_millions"] = metadata.now_cost / 10.
    paths = {}
    for gw in range(start_gw, end_gw + 1):
        result = forecast_gameweek(metadata, fixtures, gw, predict_fixture)
        result["season_name"] = season
        path = destination / f"predictions_gw{gw}.csv"
        result.to_csv(path, index=False)
        paths[str(gw)] = path.name
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p, digest in source_hashes.items()):
        raise ValueError("Forecast inputs changed during the build; discard this incomplete vintage and retry")
    manifest = {"season_name": season, "origin_gw": int(official_next),
                "created_at": datetime.now(timezone.utc).isoformat(), "inputs_as_of": now.isoformat(),
                "trained_through_gw": int(context["last_finished_gw"]),
                "deadline": next(e["deadline_time"] for e in events if e["id"] == official_next),
                "predictions": paths, "forecast_method": "individual-fixtures-v2",
                "uncertainty": "heuristic; not empirically calibrated",
                "history_scope": "cached current season; no retraining",
                "sources": source_hashes}
    (destination / "vintage.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination
