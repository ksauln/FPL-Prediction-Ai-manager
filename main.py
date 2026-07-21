from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
import pandas as pd

from fplmodel.config import (
    RAW_DIR,
    PROCESSED_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    FORMATION_OPTIONS,
    MAX_TRAIN_GW,
    USE_EXTERNAL_HISTORY,
    EXTERNAL_HISTORY_SEASONS,
)
from fplmodel.data_pull import fetch_bootstrap_static, fetch_fixtures_all, bulk_fetch_player_histories
from fplmodel.data_cleaning import normalize_bootstrap
from fplmodel.features import (
    build_training_and_pred_frames,
    expand_for_double_gw,
    log_model_feature_weights,
)
from fplmodel.model import train_models, predict_expected_points
from fplmodel.evaluation import evaluate_last_finished_gw_and_update_state
from fplmodel.state import ModelState
from fplmodel.utils import get_current_and_last_finished_gw
from fplmodel.logging_utils import configure_run_logger, update_log_filename_for_gameweek, log_timed_step
from fplmodel.external_history import load_external_histories
from fplmodel.prediction_artifacts import archive_prediction_file

from fplmodel.team_picker import pick_best_xi
try:
    from fplmodel.display import create_best_xi_graphic
except ImportError as exc:  # pragma: no cover - depends on optional plotting stack
    _DISPLAY_IMPORT_ERROR = exc

    def create_best_xi_graphic(*args, **kwargs):
        raise ImportError(
            "Rendering the best XI graphic requires the optional plotting dependencies."
        ) from _DISPLAY_IMPORT_ERROR


def build_fixture_labels(fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, next_gw: int) -> dict[int, str]:
    """
    Map each team_id to a formatted opponent string for the upcoming gameweek.
    Handles blank and double gameweeks by joining multiple fixtures with ' / '.
    """
    if fixtures_df is None or fixtures_df.empty:
        return {}
    if "event" not in fixtures_df.columns:
        return {}
    gw_fixtures = fixtures_df[fixtures_df["event"] == next_gw]
    if gw_fixtures.empty:
        return {}

    if "team_id" not in teams_df.columns:
        return {}
    name_col = "short_name" if "short_name" in teams_df.columns else "name"
    team_name_map = teams_df.set_index("team_id")[name_col].to_dict()

    fixtures_map: dict[int, list[str]] = defaultdict(list)
    for _, fixture in gw_fixtures.iterrows():
        team_h = fixture.get("team_h")
        team_a = fixture.get("team_a")
        if pd.isna(team_h) or pd.isna(team_a):
            continue
        team_h = int(team_h)
        team_a = int(team_a)
        opponent_for_home = team_name_map.get(team_a)
        opponent_for_away = team_name_map.get(team_h)
        if opponent_for_home:
            fixtures_map[team_h].append(f"{opponent_for_home} (H)")
        if opponent_for_away:
            fixtures_map[team_a].append(f"{opponent_for_away} (A)")

    return {team_id: " / ".join(parts) for team_id, parts in fixtures_map.items()}


def add_prediction_confidence(
    predictions: pd.DataFrame,
    per_model_corrected_cols: list[str],
    per_model_start_cols: list[str],
) -> pd.DataFrame:
    """
    Add confidence diagnostics for each player prediction.

    Confidence blends ensemble agreement, player-history reliability, availability,
    and start-probability certainty. Intervals are approximate 80% prediction ranges.
    """
    out = predictions.copy()
    if per_model_start_cols:
        out["start_probability"] = out[per_model_start_cols].mean(axis=1).clip(lower=0.0, upper=1.0)
    elif "start_probability" not in out.columns:
        out["start_probability"] = pd.NA

    if per_model_corrected_cols:
        model_std = out[per_model_corrected_cols].std(axis=1, ddof=0).fillna(0.0)
        model_count = len(per_model_corrected_cols)
    else:
        model_std = pd.Series(0.0, index=out.index)
        model_count = 1

    def numeric_col(name: str, default: float) -> pd.Series:
        values = out[name] if name in out.columns else pd.Series(default, index=out.index)
        return pd.to_numeric(values, errors="coerce")

    reliability = numeric_col("reliability_weight", 1.0).fillna(1.0).clip(0.0, 1.0)
    availability = numeric_col("availability_next_round", 1.0)
    if availability.isna().all():
        availability = numeric_col("availability_this_round", 1.0)
    availability = availability.fillna(numeric_col("status_availability", 1.0))
    availability = availability.fillna(1.0).clip(0.0, 1.0)

    start_probability = pd.to_numeric(out["start_probability"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    start_certainty = ((start_probability - 0.5).abs() * 2.0).clip(0.0, 1.0)
    model_agreement = (1.0 / (1.0 + model_std)).clip(0.0, 1.0)

    confidence = (
        0.35 * reliability
        + 0.25 * availability
        + 0.25 * model_agreement
        + 0.15 * start_certainty
    ).clip(0.0, 1.0)

    expected_points = pd.to_numeric(out["expected_points"], errors="coerce").fillna(0.0)
    single_model_penalty = 0.35 if model_count <= 1 else 0.0
    interval_sigma = (
        model_std
        + (1.0 - confidence) * (0.40 + 0.20 * expected_points.clip(lower=0.0))
        + single_model_penalty
    ).clip(lower=0.05)
    z_80 = 1.2815515655446004

    out["expected_points_std"] = model_std.round(4)
    out["confidence_score"] = (confidence * 100.0).round(1)
    out["confidence_level"] = pd.cut(
        out["confidence_score"],
        bins=[-0.1, 49.9, 74.9, 100.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    out["expected_points_lower_80"] = (expected_points - z_80 * interval_sigma).clip(lower=0.0).round(3)
    out["expected_points_upper_80"] = (expected_points + z_80 * interval_sigma).round(3)
    return out


def _combine_ensemble_expected_points(
    predictions: pd.DataFrame,
    per_model_raw_cols: list[str],
    per_model_corrected_cols: list[str],
) -> pd.DataFrame:
    """Combine per-model outputs without reapplying bias corrections."""
    out = predictions.copy()
    if per_model_raw_cols:
        out["expected_points_raw"] = out[per_model_raw_cols].mean(axis=1)
    else:
        out["expected_points_raw"] = 0.0

    if per_model_corrected_cols:
        out["expected_points"] = out[per_model_corrected_cols].mean(axis=1).clip(lower=0.0)
    else:
        out["expected_points"] = out["expected_points_raw"].clip(lower=0.0)
    return out


def list_current_player_history_files(raw_dir, player_ids: list[int]) -> list[str]:
    """Return cached player history files that match the current bootstrap player list."""
    current_ids = {int(player_id) for player_id in player_ids}
    files: list[str] = []
    for filename in os.listdir(raw_dir):
        if not filename.startswith("player_") or not filename.endswith(".json"):
            continue
        player_id_text = filename.removeprefix("player_").removesuffix(".json")
        if not player_id_text.isdigit():
            continue
        if int(player_id_text) in current_ids:
            files.append(filename)
    return sorted(files, key=lambda name: int(name.split("_")[1].split(".")[0]))


def infer_season_name(events_df: pd.DataFrame) -> str | None:
    """Infer an FPL season code such as ``2026-27`` from event deadlines."""
    if events_df is None or events_df.empty or "deadline_time" not in events_df.columns:
        return None
    deadlines = pd.to_datetime(events_df["deadline_time"], errors="coerce", utc=True).dropna()
    if deadlines.empty:
        return None
    start_year = int(deadlines.min().year)
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def expected_season_name_for_date(now: datetime | None = None) -> str:
    """Return the season that should be live for a calendar date."""
    now = now or datetime.now()
    start_year = now.year if now.month >= 7 else now.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def validate_season_name(
    actual_season_name: str | None,
    expected_season_name: str | None = None,
) -> str:
    """Reject stale or malformed bootstrap data before it reaches the model."""
    expected = expected_season_name or expected_season_name_for_date()
    if actual_season_name != expected:
        actual_label = actual_season_name or "unknown"
        raise RuntimeError(
            f"FPL API season mismatch: expected {expected}, but bootstrap data is "
            f"for {actual_label}. The new game may not be live yet, or the cache may "
            "be stale. Retry with --force-refetch after the FPL site launches; use "
            "--expected-season only when intentionally replaying another season."
        )
    return expected


def previous_season_name(season_name: str) -> str:
    start_year = int(season_name.split("-", 1)[0]) - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def validate_external_history_coverage(
    external_histories: pd.DataFrame,
    season_name: str,
    expected_gameweeks: int = 38,
) -> None:
    """Require a complete prior season before a GW1 cold-start run."""
    if "season_name" not in external_histories.columns or "round" not in external_histories.columns:
        raise RuntimeError("External history is missing season_name or round metadata.")
    season_rows = external_histories[external_histories["season_name"].eq(season_name)]
    rounds = {
        int(value)
        for value in pd.to_numeric(season_rows["round"], errors="coerce").dropna().unique()
    }
    missing = sorted(set(range(1, expected_gameweeks + 1)) - rounds)
    if missing:
        missing_labels = ", ".join(f"GW{gameweek}" for gameweek in missing)
        raise RuntimeError(
            f"External history for {season_name} is incomplete; missing {missing_labels}. "
            "Update data/external/Fantasy-Premier-League before the preseason run."
        )


def remap_external_histories_to_current_players(
    external_histories: pd.DataFrame,
    elements_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map season-specific historical element ids through stable player codes."""
    if external_histories is None or external_histories.empty:
        return pd.DataFrame()
    if "player_code" not in external_histories.columns or "code" not in elements_df.columns:
        return pd.DataFrame()
    current = elements_df[["player_id", "code"]].copy()
    current["player_code"] = pd.to_numeric(current["code"], errors="coerce")
    current = current.dropna(subset=["player_code"]).drop_duplicates("player_code")
    current["player_code"] = current["player_code"].astype(int)
    current = current[["player_code", "player_id"]].rename(
        columns={"player_id": "current_player_id"}
    )
    remapped = external_histories.merge(current, on="player_code", how="inner")
    remapped = remapped.drop(columns=["player_id"], errors="ignore").rename(
        columns={"current_player_id": "player_id"}
    )
    remapped["player_id"] = remapped["player_id"].astype(int)
    return remapped


def run_pipeline(
    force_refetch: bool = False,
    override_next_gw: int | None = None,
    override_last_finished_gw: int | None = None,
    expected_season_name: str | None = None,
):
    logger, file_handler, log_path = configure_run_logger()
    logger.info("Starting pipeline run (force_refetch=%s)", force_refetch)

    try:
        # 1) Pull data
        with log_timed_step(logger, "Fetching bootstrap static data"):
            bootstrap = fetch_bootstrap_static(force=force_refetch)

        with log_timed_step(logger, "Fetching fixtures data"):
            fixtures = fetch_fixtures_all(force=force_refetch)

        with log_timed_step(logger, "Normalising bootstrap data"):
            norms = normalize_bootstrap(bootstrap)
        elements_df, teams_df, events_df = norms["elements"], norms["teams"], norms["events"]
        fixtures_df = pd.DataFrame(fixtures)
        current_season_name = infer_season_name(events_df)
        validate_season_name(current_season_name, expected_season_name)
        logger.info(
            "Loaded normalised frames: %d elements, %d teams, %d events",
            len(elements_df),
            len(teams_df),
            len(events_df),
        )

        # Which GW?
        next_gw, last_finished_gw = get_current_and_last_finished_gw(events_df)
        if MAX_TRAIN_GW is not None:
            capped = min(last_finished_gw, int(MAX_TRAIN_GW))
            if capped != last_finished_gw:
                logger.info(
                    "Capping last finished GW from %s to MAX_TRAIN_GW=%s",
                    last_finished_gw,
                    MAX_TRAIN_GW,
                )
                last_finished_gw = capped
        override_applied = False
        if override_next_gw is not None:
            next_gw = int(override_next_gw)
            override_applied = True
        if override_last_finished_gw is not None:
            last_finished_gw = int(override_last_finished_gw)
            override_applied = True
        elif override_next_gw is not None:
            last_finished_gw = max(int(override_next_gw) - 1, 0)
            override_applied = True
        if override_applied:
            if last_finished_gw >= next_gw:
                adjusted = max(next_gw - 1, 0)
                logger.warning(
                    "Override produced last_finished_gw >= next_gw; adjusting last_finished_gw to %s",
                    adjusted,
                )
                last_finished_gw = adjusted
            logger.info(
                "Gameweek override applied -> next_gw=%s | last_finished_gw=%s",
                next_gw,
                last_finished_gw,
            )
        logger.info("Next gameweek: %s | Last finished gameweek: %s", next_gw, last_finished_gw)

        file_handler, log_path = update_log_filename_for_gameweek(logger, file_handler, log_path, next_gw)

        # 2) Player histories (bulk)
        player_ids = elements_df["player_id"].tolist()
        with log_timed_step(
            logger,
            f"Fetching player histories for {len(player_ids)} players (force_refetch={force_refetch})",
        ):
            bulk_fetch_player_histories(player_ids, force=force_refetch, sleep_s=0.0)

        # 3) Load histories from disk into one DF
        #    Avoid re-reading thousands of files into memory at once by streaming
        raw_files = list_current_player_history_files(RAW_DIR, player_ids)
        logger.info(
            "Loading %d current player history files from %s",
            len(raw_files),
            RAW_DIR,
        )

        rows = []
        with log_timed_step(logger, "Collating player history JSON into dataframe"):
            for fn in raw_files:
                with open(RAW_DIR / fn, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pid = int(fn.split("_")[1].split(".")[0])
                for h in data.get("history", []):
                    h["player_id"] = pid
                    rows.append(h)
        histories_df = pd.DataFrame(rows)
        logger.info("Built player histories dataframe with %d rows", len(histories_df))

        if USE_EXTERNAL_HISTORY:
            with log_timed_step(logger, "Loading external history data"):
                external_histories = load_external_histories(EXTERNAL_HISTORY_SEASONS)
            if external_histories is not None and not external_histories.empty:
                if last_finished_gw == 0 and current_season_name is not None:
                    validate_external_history_coverage(
                        external_histories,
                        previous_season_name(current_season_name),
                    )
                external_histories = remap_external_histories_to_current_players(
                    external_histories,
                    elements_df,
                )
                logger.info(
                    "Loaded %d external history rows for seasons %s",
                    len(external_histories),
                    ", ".join(EXTERNAL_HISTORY_SEASONS),
                )
                histories_df = pd.concat(
                    [histories_df, external_histories],
                    ignore_index=True,
                    sort=False,
                )
                histories_df = histories_df.drop_duplicates(
                    subset=["player_id", "fixture", "round", "season_name"],
                    keep="first",
                )
                logger.info("Combined history dataset now has %d rows", len(histories_df))
            else:
                logger.info("No external history rows loaded.")

        if histories_df.empty:
            logger.error("No current or mapped historical player data found")
            raise RuntimeError(
                "No usable player history data found. Ensure the external history dataset "
                "is installed for a pre-season GW1 run."
            )

        # 4) Build training and next-gw prediction frames
        state = ModelState(season_name=current_season_name)
        with log_timed_step(logger, "Building training and prediction feature frames"):
            X_train, y_train, X_pred, train_metadata = build_training_and_pred_frames(
                elements_df,
                teams_df,
                histories_df,
                next_gw,
                last_finished_gw,
                state,
                fixtures_df=fixtures_df,
                current_season_name=current_season_name,
            )
        logger.info(
            "Prepared features: X_train=%s, y_train=%d, X_pred=%s",
            tuple(X_train.shape),
            len(y_train),
            tuple(X_pred.shape),
        )

        # 5) Train (or retrain) models
        train_features = X_train.drop(
            columns=["player_id", "full_name", "now_cost_millions", "team_id", "element_type"],
            errors="ignore",
        )
        feature_columns = list(train_features.columns)
        logger.info("Training models with %d features", train_features.shape[1])
        logger.info(
            "Training feature columns: %s",
            ", ".join(train_features.columns.astype(str)),
        )
        with log_timed_step(logger, "Training prediction models"):
            clf, appearance_clf, reg, cameo_points, candidate_models = train_models(
                train_features,
                y_train,
                train_metadata,
            )
        log_model_feature_weights(logger, train_features.columns, reg, model_label="regressor")
        log_model_feature_weights(logger, train_features.columns, clf, model_label="classifier")
        log_model_feature_weights(
            logger,
            train_features.columns,
            appearance_clf,
            model_label="appearance classifier",
        )
        logger.info(
            "Model training complete; fitted %d candidate pair(s).",
            len(candidate_models),
        )

        # 6) Predict EP for next GW
        meta_cols = [
            "player_id",
            "full_name",
            "team_name",
            "now_cost_millions",
            "team_id",
            "element_type",
            "reliability_weight",
        ]
        meta_cols.extend(
            col
            for col in (
                "availability_this_round",
                "availability_next_round",
                "status_availability",
                "status_injury_flag",
                "injury_risk_flag",
            )
            if col in X_pred.columns
        )
        per_model_raw_cols: list[str] = []
        per_model_corrected_cols: list[str] = []
        per_model_start_cols: list[str] = []
        per_model_appearance_cols: list[str] = []
        per_model_points_hat_cols: list[str] = []
        per_model_cameo_cols: list[str] = []
        ensemble_predictions = None

        with log_timed_step(logger, "Generating next gameweek predictions"):
            for bundle in candidate_models:
                preds = predict_expected_points(
                    X_pred,
                    bundle.classifier,
                    bundle.regressor,
                    state,
                    appearance_clf=bundle.appearance_classifier,
                    cameo_points_by_position=bundle.cameo_points_by_position,
                )
                suffix = bundle.name
                raw_col = f"expected_points_raw__{suffix}"
                corrected_col = f"expected_points__{suffix}"
                start_col = f"start_probability__{suffix}"
                appearance_col = f"appearance_probability__{suffix}"
                points_hat_col = f"points_hat__{suffix}"
                cameo_col = f"cameo_points__{suffix}"
                if ensemble_predictions is None:
                    ensemble_predictions = preds[meta_cols].copy()
                ensemble_predictions[raw_col] = preds["expected_points_raw"].values
                ensemble_predictions[corrected_col] = preds["expected_points"].values
                ensemble_predictions[start_col] = preds["p_start"].values
                ensemble_predictions[appearance_col] = preds["p_appearance"].values
                ensemble_predictions[points_hat_col] = preds["points_hat"].values
                ensemble_predictions[cameo_col] = preds["element_type"].map(
                    bundle.cameo_points_by_position
                ).fillna(1.0).values
                per_model_raw_cols.append(raw_col)
                per_model_corrected_cols.append(corrected_col)
                per_model_start_cols.append(start_col)
                per_model_appearance_cols.append(appearance_col)
                per_model_points_hat_cols.append(points_hat_col)
                per_model_cameo_cols.append(cameo_col)

        if ensemble_predictions is None:
            raise RuntimeError("No candidate predictions were generated for the ensemble.")

        predictions = _combine_ensemble_expected_points(
            ensemble_predictions,
            per_model_raw_cols=per_model_raw_cols,
            per_model_corrected_cols=per_model_corrected_cols,
        )
        predictions["appearance_probability"] = predictions[
            per_model_appearance_cols
        ].mean(axis=1).clip(lower=0.0, upper=1.0)
        predictions["cameo_points"] = predictions[per_model_cameo_cols].mean(axis=1)
        logger.info(
            "Generated ensemble predictions for %d players using %d model(s): %s",
            len(predictions),
            len(candidate_models),
            ", ".join(bundle.display_name for bundle in candidate_models),
        )

        # 7) Double/Blank GW scaling (approximate): multiply EP by number of fixtures
        predictions = expand_for_double_gw(predictions, fixtures_df, next_gw)
        if "fixture_multiplier" in predictions.columns:
            predictions["expected_points"] = predictions["expected_points"] * predictions["fixture_multiplier"]
            for col in per_model_corrected_cols:
                predictions[col] = predictions[col] * predictions["fixture_multiplier"]
            for col in per_model_points_hat_cols:
                predictions[col] = predictions[col] * predictions["fixture_multiplier"]
            for col in per_model_cameo_cols:
                predictions[col] = predictions[col] * predictions["fixture_multiplier"]
            predictions["cameo_points"] = (
                predictions["cameo_points"] * predictions["fixture_multiplier"]
            )
        predictions = add_prediction_confidence(
            predictions,
            per_model_corrected_cols=per_model_corrected_cols,
            per_model_start_cols=per_model_start_cols,
        )
        predictions["season_name"] = current_season_name
        predictions["gameweek"] = int(next_gw)
        logger.info("Applied fixture multipliers; average EP now %.2f", predictions["expected_points"].mean())

        top_preds = (
            predictions.sort_values("expected_points", ascending=False)
            .head(5)
            .apply(lambda row: f"{row['full_name']} ({row['expected_points']:.2f})", axis=1)
            .tolist()
        )
        if top_preds:
            logger.info("Top 5 expected point predictions (ensemble): %s", "; ".join(top_preds))

        # 8) Evaluate last finished GW and update biases (EMA)
        train_like = X_pred[["player_id", "element_type"] + feature_columns].copy()
        with log_timed_step(logger, "Evaluating last finished gameweek residuals"):
            res_df = evaluate_last_finished_gw_and_update_state(
                clf,
                appearance_clf,
                reg,
                cameo_points,
                train_like,
                histories_df,
                last_finished_gw,
                state,
            )
        if res_df is not None and len(res_df):
            logger.info("Residuals computed for %d players in GW %s", len(res_df), last_finished_gw)
        else:
            logger.info("No residuals available for GW %s", last_finished_gw)

        # 9) Best XI selection
        logger.info("Selecting best XI from %d candidates", len(predictions))
        picker_cols = [
            "player_id",
            "full_name",
            "team_name",
            "team_id",
            "element_type",
            "now_cost_millions",
            "expected_points",
            "start_probability",
            "appearance_probability",
            "availability_this_round",
            "availability_next_round",
            "status_availability",
            "fixture_multiplier",
            "cameo_points",
            "confidence_score",
            "confidence_level",
            "expected_points_lower_80",
            "expected_points_upper_80",
        ]
        picker_cols = [col for col in picker_cols if col in predictions.columns]
        with log_timed_step(logger, "Optimising best XI selection"):
            team = pick_best_xi(
                predictions[picker_cols],
                formations=FORMATION_OPTIONS,
            )
        team["season_name"] = current_season_name
        team["gameweek"] = int(next_gw)
        logger.info(
            "Selected squad: total cost %.1fM | starting cost %.1fM | bench cost %.1fM",
            team["total_cost"],
            team["starting_cost"],
            team["bench_cost"],
        )
        fixture_labels = build_fixture_labels(fixtures_df, teams_df, next_gw)
        if fixture_labels:
            for section in ("squad", "bench"):
                for player in team.get(section, []):
                    team_id = player.get("team_id")
                    if team_id in fixture_labels:
                        player["next_fixture"] = fixture_labels[team_id]
        logger.info(
            "Projected points: XI %.2f | Captain included %.2f | Bench %.2f",
            team["expected_points_without_captain"],
            team["total_expected_points_with_captain"],
            team["bench_expected_points"],
        )
        logger.info("Captain: %s", team.get("captain"))

        for player in team.get("squad", []):
            logger.info(
                "XI | %s (%s) - %.2f pts | %.1fM",
                player["full_name"],
                player["team_name"],
                player["expected_points"],
                player["now_cost_millions"],
            )
        for bench_player in team.get("bench", []):
            logger.info(
                "Bench %d | %s (%s) - %.2f pts",
                bench_player.get("bench_order", 0),
                bench_player["full_name"],
                bench_player["team_name"],
                bench_player["expected_points"],
            )

        # 10) Save artifacts
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        predictions_csv = OUTPUTS_DIR / f"predictions_gw{next_gw}.csv"
        predictions.sort_values("expected_points", ascending=False).to_csv(predictions_csv, index=False)
        logger.info("Predictions saved to %s", predictions_csv)
        predictions_archive_csv = archive_prediction_file(
            predictions_path=predictions_csv,
            output_root=OUTPUTS_DIR,
            bootstrap_path=RAW_DIR / "bootstrap-static.json",
            season_name=current_season_name,
            gameweek=int(next_gw),
        )
        logger.info("Season prediction archive saved to %s", predictions_archive_csv)

        xi_csv = OUTPUTS_DIR / f"starting_xi_gw{next_gw}.csv"
        squad_df = pd.DataFrame(team.get("squad", []))
        if not squad_df.empty:
            squad_df.to_csv(xi_csv, index=False)
            logger.info("Starting XI saved to %s", xi_csv)
        else:
            xi_csv = None
            logger.warning("No starting XI to save for GW %s", next_gw)

        bench_csv = OUTPUTS_DIR / f"bench_gw{next_gw}.csv"
        bench_df = pd.DataFrame(team.get("bench", []))
        if not bench_df.empty:
            bench_df.to_csv(bench_csv, index=False)
            logger.info("Bench saved to %s", bench_csv)
        else:
            bench_csv = None
            logger.warning("No bench to save for GW %s", next_gw)

        team_json = OUTPUTS_DIR / f"best_xi_gw{next_gw}.json"
        with open(team_json, "w", encoding="utf-8") as f:
            json.dump(team, f, indent=2)
        logger.info("Best XI JSON saved to %s", team_json)

        with log_timed_step(logger, "Rendering best XI graphic"):
            team_image = create_best_xi_graphic(team, gameweek=next_gw)
        logger.info("Best XI graphic generated at %s", team_image)

        # Residuals summary
        residuals_csv = None
        if res_df is not None and len(res_df):
            residuals_csv = OUTPUTS_DIR / f"residuals_gw{last_finished_gw}.csv"
            res_df.to_csv(residuals_csv, index=False)
            logger.info("Residuals saved to %s", residuals_csv)

        logger.info("Pipeline complete for GW %s", next_gw)
        return {
            "season_name": current_season_name,
            "next_gw": int(next_gw),
            "last_finished_gw": int(last_finished_gw),
            "predictions_csv": str(predictions_csv),
            "predictions_archive_csv": str(predictions_archive_csv),
            "team_json": str(team_json),
            "team_graphic": str(team_image),
            "starting_xi_csv": str(xi_csv) if xi_csv is not None else None,
            "bench_csv": str(bench_csv) if bench_csv is not None else None,
            "residuals_csv": str(residuals_csv) if residuals_csv is not None else None,
            "log_file": str(log_path),
        }
    except Exception:
        logger.exception("Pipeline execution failed")
        raise
    finally:
        try:
            logger.removeHandler(file_handler)
        except ValueError:
            pass
        file_handler.close()

def replay_gameweeks(
    start_gw: int,
    end_gw: int | None = None,
    force_refetch: bool = False,
    expected_season_name: str | None = None,
) -> list[dict[str, object]]:
    """
    Sequentially run the pipeline for a range of gameweeks using GW overrides.
    """
    if start_gw < 1:
        raise ValueError("start_gw must be >= 1")
    end_gw = start_gw if end_gw is None else end_gw
    if end_gw < start_gw:
        raise ValueError("end_gw must be >= start_gw")

    results: list[dict[str, object]] = []
    for idx, gw in enumerate(range(start_gw, end_gw + 1)):
        run_force_refetch = force_refetch if idx == 0 else False
        result = run_pipeline(
            force_refetch=run_force_refetch,
            override_next_gw=gw,
            override_last_finished_gw=max(gw - 1, 0),
            expected_season_name=expected_season_name,
        )
        results.append(result)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FPL prediction pipeline.")
    parser.add_argument(
        "--force-refetch",
        action="store_true",
        help="Ignore cached API responses and refetch all upstream data.",
    )
    parser.add_argument(
        "--override-next-gw",
        type=int,
        help="Manually set the next gameweek for a single pipeline run.",
    )
    parser.add_argument(
        "--override-last-finished-gw",
        type=int,
        help="Manually set the last finished gameweek for a single pipeline run.",
    )
    parser.add_argument(
        "--expected-season",
        help=(
            "Expected season code, for example 2026-27. Defaults to the season implied "
            "by today's date and rejects stale bootstrap data."
        ),
    )
    parser.add_argument(
        "--replay-start-gw",
        type=int,
        help="Start gameweek for sequential replay using overrides.",
    )
    parser.add_argument(
        "--replay-end-gw",
        type=int,
        help="End gameweek (inclusive) for sequential replay. Defaults to start.",
    )
    args = parser.parse_args()

    if args.replay_start_gw is not None:
        results = replay_gameweeks(
            start_gw=args.replay_start_gw,
            end_gw=args.replay_end_gw,
            force_refetch=args.force_refetch,
            expected_season_name=args.expected_season,
        )
        print(json.dumps(results, indent=2))
    else:
        out = run_pipeline(
            force_refetch=args.force_refetch,
            override_next_gw=args.override_next_gw,
            override_last_finished_gw=args.override_last_finished_gw,
            expected_season_name=args.expected_season,
        )
        print(json.dumps(out, indent=2))
