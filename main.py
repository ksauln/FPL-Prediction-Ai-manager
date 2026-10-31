from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd

from fplmodel import config as model_config
from fplmodel import model as model_module
from fplmodel.config import (
    RAW_DIR,
    PROCESSED_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    FORMATION_OPTIONS,
    MAX_TRAIN_GW,
    USE_EXTERNAL_HISTORY,
    EXTERNAL_HISTORY_SEASONS,
    OFFICIAL_EP_BLEND_WEIGHT,
    OFFICIAL_EP_COLD_START_BLEND_WEIGHT,
    PROVIDER_EP_BLEND_WEIGHT,
    PRESEASON_SNAPSHOT_DIR,
)
from fplmodel.data_pull import fetch_bootstrap_static, fetch_fixtures_all, bulk_fetch_player_histories
from fplmodel.data_cleaning import normalize_bootstrap
from fplmodel.features import (
    build_training_and_pred_frames,
    expand_for_double_gw,
    log_model_feature_weights,
)
from fplmodel.model import train_models, predict_expected_points
from fplmodel.evaluation import (
    evaluate_last_finished_gw_and_update_state,
    prequential_training_mask,
)
from fplmodel.state import ModelState
from fplmodel.utils import get_current_and_last_finished_gw
from fplmodel.logging_utils import configure_run_logger, update_log_filename_for_gameweek, log_timed_step
from fplmodel.external_history import load_external_histories
from fplmodel.history_universe import build_history_universe
from fplmodel.preseason import load_preseason_snapshot_context
from fplmodel.snapshot_catalog import SnapshotCatalog
from fplmodel.live_rules import validate_live_fpl_configuration
from fplmodel.prediction_artifacts import (
    archive_prediction_file,
    load_archived_gameweek_inputs,
)

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

    preseason_quality = numeric_col("preseason_prior_quality", 0.0).fillna(0.0).clip(0.0, 1.0)
    evidence = np.maximum(reliability, 0.8 * preseason_quality)
    confidence = (
        0.40 * evidence
        + 0.20 * availability
        + 0.25 * model_agreement * evidence
        + 0.15 * start_certainty * evidence
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


def scale_prediction_totals_for_fixtures(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Convert every per-fixture points total to the row's Gameweek basis."""
    out = predictions.copy()
    if "fixture_multiplier" not in out.columns:
        return out
    multiplier = pd.to_numeric(
        out["fixture_multiplier"], errors="coerce"
    ).fillna(0.0)
    exact_totals = {"expected_points_raw", "expected_points", "cameo_points"}
    total_prefixes = (
        "expected_points_raw__",
        "expected_points__",
        "points_hat__",
        "cameo_points__",
    )
    total_columns = [
        column
        for column in out.columns
        if column in exact_totals or column.startswith(total_prefixes)
    ]
    for column in total_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce") * multiplier
    return out


def blend_official_expected_points(
    predictions: pd.DataFrame,
    *,
    weight: float = OFFICIAL_EP_BLEND_WEIGHT,
    cold_start_weight: float = OFFICIAL_EP_COLD_START_BLEND_WEIGHT,
) -> pd.DataFrame:
    """Reliability-weight live ``ep_next`` where it is available."""

    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("Official EP blend weight must be between 0 and 1")
    if not 0.0 <= float(cold_start_weight) <= 1.0:
        raise ValueError("Cold-start official EP blend weight must be between 0 and 1")
    out = predictions.copy()
    model_ep = pd.to_numeric(out["expected_points"], errors="coerce").fillna(0.0)
    official_ep = pd.to_numeric(
        out.get("official_ep_next", pd.Series(pd.NA, index=out.index)),
        errors="coerce",
    ).clip(lower=0.0)
    available = official_ep.notna()
    reliability = pd.to_numeric(
        out.get("reliability_weight", pd.Series(1.0, index=out.index)),
        errors="coerce",
    ).fillna(1.0).clip(0.0, 1.0)
    applied_weight = pd.Series(0.0, index=out.index)
    adaptive_weight = float(weight) + (
        float(cold_start_weight) - float(weight)
    ) * (1.0 - reliability)
    applied_weight.loc[available] = adaptive_weight.loc[available]
    out["model_expected_points"] = model_ep
    out["official_ep_blend_weight"] = applied_weight
    out["expected_points"] = (
        (1.0 - applied_weight) * model_ep
        + applied_weight * official_ep.fillna(model_ep)
    ).clip(lower=0.0)
    return out


def blend_provider_expected_points(
    predictions: pd.DataFrame,
    *,
    weight: float = PROVIDER_EP_BLEND_WEIGHT,
) -> pd.DataFrame:
    """Blend an optional cutoff-safe licensed/free projection snapshot."""
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("Provider EP blend weight must be between 0 and 1")
    out = predictions.copy()
    current_ep = pd.to_numeric(out["expected_points"], errors="coerce").fillna(0.0)
    provider_ep = pd.to_numeric(
        out.get("provider_expected_points", pd.Series(pd.NA, index=out.index)),
        errors="coerce",
    ).clip(lower=0.0)
    row_weight = pd.to_numeric(
        out.get("provider_expected_points_weight", pd.Series(weight, index=out.index)),
        errors="coerce",
    ).fillna(float(weight)).clip(0.0, 1.0)
    applied = row_weight.where(provider_ep.notna(), 0.0)
    out["pre_provider_expected_points"] = current_ep
    out["provider_ep_blend_weight"] = applied
    out["expected_points"] = (
        (1.0 - applied) * current_ep + applied * provider_ep.fillna(current_ep)
    ).clip(lower=0.0)
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


def target_gameweek_deadline(events_df: pd.DataFrame, gameweek: int) -> datetime:
    """Return the target deadline used as the point-in-time snapshot cutoff."""
    if events_df is not None and not events_df.empty:
        event_col = "event_id" if "event_id" in events_df.columns else "id"
        if event_col in events_df.columns and "deadline_time" in events_df.columns:
            target = events_df[
                pd.to_numeric(events_df[event_col], errors="coerce").eq(int(gameweek))
            ]
            if not target.empty:
                deadline = pd.to_datetime(
                    target.iloc[0]["deadline_time"], errors="coerce", utc=True
                )
                if pd.notna(deadline):
                    return deadline.to_pydatetime()
    raise ValueError(
        f"Cannot resolve the GW{int(gameweek)} deadline from bootstrap events; "
        "refusing to use the current clock as a point-in-time snapshot cutoff."
    )


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
    """Compatibility wrapper for building the full historical training universe."""
    if external_histories is None or external_histories.empty:
        return pd.DataFrame()
    if "player_code" not in external_histories.columns or "code" not in elements_df.columns:
        return pd.DataFrame()
    return build_history_universe(external_histories, elements_df)


def _update_training_digest(
    digest,
    label: str,
    values: pd.DataFrame | pd.Series,
) -> None:
    """Add a pandas object's full ordered contents and schema to a digest."""
    frame = values.to_frame() if isinstance(values, pd.Series) else values
    schema = [
        (str(column), str(dtype))
        for column, dtype in zip(frame.columns, frame.dtypes, strict=True)
    ]
    digest.update(label.encode("utf-8"))
    digest.update(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    digest.update(str(tuple(frame.shape)).encode("ascii"))
    digest.update(
        pd.util.hash_pandas_object(frame, index=True, categorize=True)
        .to_numpy(dtype=np.uint64, copy=False)
        .tobytes()
    )


def training_input_fingerprint(
    train_features: pd.DataFrame,
    y_train: pd.Series,
    train_metadata: pd.DataFrame,
) -> str:
    """Return a stable fingerprint for the full ordered model-fitting dataset."""
    digest = hashlib.sha256()
    _update_training_digest(digest, "features", train_features)
    _update_training_digest(digest, "targets", y_train)
    _update_training_digest(digest, "metadata", train_metadata)
    return digest.hexdigest()


_MODEL_TRAINING_CONFIG_NAMES = (
    "REG_PARAMS",
    "CLF_PARAMS",
    "RANDOM_SEED",
    "FEATURE_CORRELATION_THRESHOLD",
    "FEATURE_MIN_VARIANCE",
    "ENABLE_HYPERPARAM_TUNING",
    "HYPERPARAM_TUNING_MIN_SAMPLES",
    "HYPERPARAM_TUNING_ITER",
    "HYPERPARAM_TUNING_CV",
    "MODEL_SELECTION_MAX_SAMPLES",
    "MODEL_TRAINING_N_JOBS",
    "SEASON_WEIGHT_DECAY",
    "SEASON_WEIGHT_MIN",
    "REG_PARAM_DISTRIBUTIONS",
    "CLF_PARAM_DISTRIBUTIONS",
    "RF_REG_PARAM_DISTRIBUTIONS",
    "RF_CLF_PARAM_DISTRIBUTIONS",
    "MLP_REG_PARAM_DISTRIBUTIONS",
    "MLP_CLF_PARAM_DISTRIBUTIONS",
    "XGB_REG_PARAM_DISTRIBUTIONS",
    "XGB_CLF_PARAM_DISTRIBUTIONS",
    "ENABLE_GPU_TRAINING",
    "ENABLE_RANDOM_FOREST_MODELS",
    "ENABLE_MLP_MODELS",
)


def _callable_training_signature(value) -> dict[str, object]:
    code = getattr(value, "__code__", None)
    if code is None:
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }
    return {
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", None),
        "bytecode": code.co_code.hex(),
        "constants": repr(code.co_consts),
        "names": list(code.co_names),
    }


def model_trainer_signature(train_fn) -> str:
    """Fingerprint the effective in-process trainer and model-fit configuration."""
    payload = {
        "config": {
            name: getattr(model_module, name)
            for name in _MODEL_TRAINING_CONFIG_NAMES
        },
        "trainer": _callable_training_signature(train_fn),
        "components": {
            name: _callable_training_signature(getattr(model_module, name))
            for name in (
                "_build_model_candidates",
                "_evaluate_model_candidates",
                "_fit_with_optional_tuning",
            )
        },
        "xgboost": {
            "classifier": repr(getattr(model_module, "XGBClassifier", None)),
            "regressor": repr(getattr(model_module, "XGBRegressor", None)),
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def train_prediction_models_for_replay(
    train_features: pd.DataFrame,
    y_train: pd.Series,
    train_metadata: pd.DataFrame,
    *,
    model_cache: dict[str, object] | None = None,
    train_fn=None,
    gameweek: int | None = None,
) -> tuple[tuple[object, object, object, object, object], bool, str, int | None]:
    """Fit models, or reuse the immediately previous exact training result."""
    fit = train_models if train_fn is None else train_fn
    data_fingerprint = training_input_fingerprint(
        train_features,
        y_train,
        train_metadata,
    )
    trainer_fingerprint = model_trainer_signature(fit)
    fingerprint = hashlib.sha256(
        f"{data_fingerprint}:{trainer_fingerprint}".encode("ascii")
    ).hexdigest()
    if model_cache is not None and model_cache.get("fingerprint") == fingerprint:
        cached_models = model_cache.get("models")
        if isinstance(cached_models, tuple) and len(cached_models) == 5:
            source_gameweek = model_cache.get("source_gameweek")
            return (
                cached_models,
                True,
                fingerprint,
                int(source_gameweek) if source_gameweek is not None else None,
            )

    fitted_models = fit(train_features, y_train, train_metadata)
    if model_cache is not None:
        # Keep only the immediately previous fit. This guarantees that the model
        # files on disk still match the objects copied into this GW's archive.
        model_cache.clear()
        model_cache.update(
            {
                "fingerprint": fingerprint,
                "data_fingerprint": data_fingerprint,
                "trainer_fingerprint": trainer_fingerprint,
                "models": fitted_models,
                "source_gameweek": gameweek,
            }
        )
    return fitted_models, False, fingerprint, gameweek


def run_pipeline(
    force_refetch: bool = False,
    override_next_gw: int | None = None,
    override_last_finished_gw: int | None = None,
    expected_season_name: str | None = None,
    snapshot_mode: str = "auto",
    snapshot_root=None,
    model_cache: dict[str, object] | None = None,
):
    snapshot_mode = str(snapshot_mode).strip().lower()
    logger, file_handler, log_path = configure_run_logger()
    logger.info("Starting pipeline run (force_refetch=%s)", force_refetch)

    try:
        # 1) Pull data
        with log_timed_step(logger, "Fetching bootstrap static data"):
            bootstrap = fetch_bootstrap_static(force=force_refetch)

        with log_timed_step(logger, "Fetching fixtures data"):
            fixtures = fetch_fixtures_all(force=force_refetch)

        bootstrap_source_path = RAW_DIR / "bootstrap-static.json"
        fixtures_source_path = RAW_DIR / "fixtures-all.json"

        with log_timed_step(logger, "Normalising bootstrap data"):
            norms = normalize_bootstrap(bootstrap)
        elements_df, teams_df, events_df = norms["elements"], norms["teams"], norms["events"]
        fixtures_df = pd.DataFrame(fixtures)
        current_season_name = infer_season_name(events_df)
        validate_season_name(current_season_name, expected_season_name)
        live_rules = validate_live_fpl_configuration(bootstrap)
        logger.info(
            "Loaded normalised frames: %d elements, %d teams, %d events",
            len(elements_df),
            len(teams_df),
            len(events_df),
        )
        logger.info(
            "Validated live FPL rules: £%.1fm budget, %d max free transfers, "
            "first chip half through GW%d",
            live_rules.budget_m,
            live_rules.max_free_transfers,
            live_rules.first_half_end_gw,
        )

        # Which GW?
        next_gw, last_finished_gw = get_current_and_last_finished_gw(events_df)
        official_next_gw = next_gw
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

        # Historical replays must use the exact metadata archived for that
        # target.  Live bootstrap fields can otherwise leak later transfers,
        # injuries, positions, prices, and set-piece roles into the past.
        if override_applied and int(next_gw) < int(official_next_gw):
            (
                bootstrap,
                fixtures,
                bootstrap_source_path,
                fixtures_source_path,
            ) = load_archived_gameweek_inputs(
                output_root=OUTPUTS_DIR,
                season_name=str(current_season_name),
                gameweek=int(next_gw),
            )
            replay_norms = normalize_bootstrap(bootstrap)
            elements_df = replay_norms["elements"]
            teams_df = replay_norms["teams"]
            events_df = replay_norms["events"]
            fixtures_df = pd.DataFrame(fixtures)
            replay_season_name = infer_season_name(events_df)
            validate_season_name(replay_season_name, str(current_season_name))
            validate_live_fpl_configuration(bootstrap)
            logger.info(
                "Loaded archived as-of bootstrap and fixtures for %s GW%s replay.",
                current_season_name,
                next_gw,
            )
        logger.info("Next gameweek: %s | Last finished gameweek: %s", next_gw, last_finished_gw)

        snapshot_cutoff = target_gameweek_deadline(events_df, next_gw)
        snapshot_catalog = SnapshotCatalog(snapshot_root or PRESEASON_SNAPSHOT_DIR)
        required_snapshot_topics = (
            ("preseason_player_usage", "team_priors")
            if snapshot_mode == "strict" and int(next_gw) == 1
            else ()
        )
        expected_players = None
        if {"code", "player_id", "team_id"}.issubset(elements_df.columns) and {
            "team_id",
            "code",
            "name",
        }.issubset(teams_df.columns):
            player_columns = [
                column
                for column in (
                    "code",
                    "player_id",
                    "full_name",
                    "team_id",
                    "can_select",
                    "removed",
                )
                if column in elements_df.columns
            ]
            expected_players = elements_df[player_columns].rename(
                columns={"code": "player_code"}
            )
            team_identity = teams_df[["team_id", "code", "name"]].rename(
                columns={"code": "team_code", "name": "team_name"}
            )
            expected_players = expected_players.merge(
                team_identity,
                on="team_id",
                how="left",
                validate="many_to_one",
            )
        with log_timed_step(logger, "Resolving point-in-time enrichment snapshots"):
            preseason_context = load_preseason_snapshot_context(
                snapshot_catalog,
                season_name=current_season_name,
                cutoff=snapshot_cutoff,
                mode=snapshot_mode,
                required_topics=required_snapshot_topics,
                target_gameweek=next_gw,
                expected_team_codes=(
                    set(pd.to_numeric(teams_df["code"], errors="coerce").dropna().astype(int))
                    if "code" in teams_df.columns
                    else None
                ),
                expected_players=expected_players,
                expected_teams=(
                    teams_df[["team_id", "code"]].copy()
                    if {"team_id", "code"}.issubset(teams_df.columns)
                    else None
                ),
            )
        logger.info(
            "Resolved %d enrichment snapshot(s) at cutoff %s",
            len(preseason_context.snapshot_paths),
            snapshot_cutoff.isoformat(),
        )

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
                    row = dict(h)
                    row["player_id"] = pid
                    row["current_player_id"] = pid
                    row["season_name"] = current_season_name
                    rows.append(row)
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
                if (
                    "position_changed" in external_histories.columns
                    and current_season_name is not None
                ):
                    prior_season = previous_season_name(current_season_name)
                    changed = external_histories[
                        external_histories["season_name"].eq(prior_season)
                        & external_histories["position_changed"].fillna(False)
                    ][
                        [
                            "player_code",
                            "element_type",
                            "current_element_type",
                        ]
                    ].drop_duplicates("player_code")
                    logger.info(
                        "Detected %d mapped player position change(s) from %s to %s.",
                        len(changed),
                        prior_season,
                        current_season_name,
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
        if state.last_evaluated_gw > last_finished_gw:
            logger.warning(
                "Resetting model bias state evaluated through GW%d because the "
                "requested %s run is final only through GW%d.",
                state.last_evaluated_gw,
                current_season_name,
                last_finished_gw,
            )
            state.reset_biases()
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
                official_next_gw=official_next_gw,
                preseason_player_df=preseason_context.player_enrichment,
                team_priors_df=preseason_context.team_priors,
            )
        logger.info(
            "Prepared features: X_train=%s, y_train=%d, X_pred=%s",
            tuple(X_train.shape),
            len(y_train),
            tuple(X_pred.shape),
        )

        # 5) Build the exact numeric feature matrix used by both calibration and
        # the final fit.  Identity/display columns must never reach estimators.
        train_features = X_train.drop(
            columns=["player_id", "full_name", "now_cost_millions", "team_id", "element_type"],
            errors="ignore",
        )
        logger.info("Training models with %d features", train_features.shape[1])
        logger.info(
            "Training feature columns: %s",
            ", ".join(train_features.columns.astype(str)),
        )

        # Fit the residual-calibration model only on rows available before the
        # Gameweek being evaluated.  Training on the target GW itself would make
        # the persistent bias update in-sample and systematically too optimistic.
        evaluation_frame = train_features.copy()
        for column in train_metadata.columns:
            evaluation_frame[column] = train_metadata[column]
        res_df = pd.DataFrame()
        evaluation_models = None
        evaluation_due = (
            int(last_finished_gw) > 0
            and state.last_evaluated_gw < int(last_finished_gw)
        )
        target_rows = pd.to_numeric(
            train_metadata.get("round", pd.Series(index=train_metadata.index, dtype=float)),
            errors="coerce",
        ).eq(int(last_finished_gw))
        if "season_name" in train_metadata.columns:
            target_rows &= train_metadata["season_name"].astype(str).eq(
                str(current_season_name)
            )
        if evaluation_due and bool(target_rows.any()):
            calibration_mask = prequential_training_mask(
                train_metadata,
                evaluation_gw=int(last_finished_gw),
                current_season_name=str(current_season_name),
            )
            if int(calibration_mask.sum()) >= 3:
                with log_timed_step(
                    logger,
                    "Training prequential residual-calibration models",
                ):
                    eval_clf, eval_appearance_clf, eval_reg, eval_cameo, _ = train_models(
                        train_features.loc[calibration_mask].reset_index(drop=True),
                        y_train.loc[calibration_mask].reset_index(drop=True),
                        train_metadata.loc[calibration_mask].reset_index(drop=True),
                    )
                evaluation_models = (
                    eval_clf,
                    eval_appearance_clf,
                    eval_reg,
                    eval_cameo,
                )
            else:
                logger.warning(
                    "Skipping GW%s residual calibration: only %d pre-GW training rows.",
                    last_finished_gw,
                    int(calibration_mask.sum()),
                )
        # 6) Fit the final models. A sequential planning replay may reuse the
        # immediately previous fit only when every ordered fitting input is exact.
        if evaluation_models is not None and model_cache is not None:
            # The calibration fit persists temporary estimators to the shared
            # model paths. Force a final fit so this GW archives prediction models.
            model_cache.clear()
        with log_timed_step(logger, "Training or reusing prediction models"):
            (
                fitted_models,
                reused_fitted_models,
                model_training_fingerprint,
                model_training_source_gameweek,
            ) = train_prediction_models_for_replay(
                train_features,
                y_train,
                train_metadata,
                model_cache=model_cache,
                gameweek=int(next_gw),
            )
        clf, appearance_clf, reg, cameo_points, candidate_models = fitted_models
        if reused_fitted_models:
            logger.info(
                "Reused fitted prediction models for identical data and trainer %s.",
                model_training_fingerprint[:12],
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

        # Only persist the prequential residual update after the final model fit
        # has succeeded, so a failed run cannot advance calibration state.
        if evaluation_models is not None:
            eval_clf, eval_appearance_clf, eval_reg, eval_cameo = evaluation_models
            with log_timed_step(logger, "Evaluating last finished gameweek residuals"):
                res_df = evaluate_last_finished_gw_and_update_state(
                    eval_clf,
                    eval_appearance_clf,
                    eval_reg,
                    eval_cameo,
                    evaluation_frame,
                    histories_df,
                    last_finished_gw,
                    state,
                    current_season_name=current_season_name,
                )
        if len(res_df):
            logger.info("Residuals computed for %d rows in GW %s", len(res_df), last_finished_gw)
        else:
            logger.info("No new residuals available for GW %s", last_finished_gw)

        # 7) Predict EP for next GW
        meta_cols = [
            "player_id",
            "full_name",
            "team_name",
            "now_cost_millions",
            "team_id",
            "element_type",
            "reliability_weight",
            "start_prior_weight",
            "appearance_prior_weight",
            "cold_start_bucket",
        ]
        meta_cols.extend(
            col
            for col in (
                "availability_this_round",
                "availability_next_round",
                "status_availability",
                "status_injury_flag",
                "status_persistent_flag",
                "status_return_date",
                "target_fixture_kickoff",
                "future_status_applied_flag",
                "injury_risk_flag",
                "official_ep_next",
                "price_change_percent",
                "player_code",
                "season_minutes",
                "history_match_count",
                "current_season_match_count",
                "set_piece_duty_score",
                "has_corners_duty",
                "primary_corners_taker",
                "has_direct_fk_duty",
                "primary_direct_fk_taker",
                "has_penalty_duty",
                "primary_penalty_taker",
                "preseason_minutes",
                "preseason_starts",
                "preseason_appearances",
                "preseason_last_two_minutes",
                "prior_p_start",
                "prior_p_appearance",
                "preseason_prior_quality",
                "preseason_data_source",
                "source_confidence",
                "preseason_appearance_share",
                "preseason_start_share",
                "preseason_minutes_share",
                "preseason_recent_minutes_share",
                "preseason_evidence_components",
                "preseason_evidence_depth",
                "role_target_gameweek",
                "attacking_role_delta",
                "penalty_role_delta",
                "corner_role_delta",
                "direct_fk_role_delta",
                "role_evidence_matches",
                "role_evidence_confidence",
                "role_last_observed_at",
                "role_change_score",
                "role_evidence_components",
                "role_evidence_quality",
                "role_data_source",
                "role_notes",
                "tournament_minutes",
                "tournament_matches",
                "tournament_starts",
                "last_tournament_appearance_at",
                "late_return_flag",
                "readiness_multiplier",
                "workload_evidence_confidence",
                "workload_valid_through_gameweek",
                "workload_notes",
                "workload_data_source",
                "provider_expected_points",
                "provider_expected_points_weight",
                "provider_expected_points_source",
                "team_prior_source",
                "opponent_team_prior_source",
                "team_context_prior_weight",
                "opponent_team_context_prior_weight",
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
                    available_meta_cols = [
                        column for column in meta_cols if column in preds.columns
                    ]
                    available_meta_cols.extend(
                        column
                        for column in (
                            "role_adjustment_weight",
                            "role_ep_multiplier",
                        )
                        if column in preds.columns and column not in available_meta_cols
                    )
                    ensemble_predictions = preds[available_meta_cols].copy()
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

        # 8) Double/Blank GW scaling (approximate): multiply EP by number of fixtures
        predictions = expand_for_double_gw(predictions, fixtures_df, next_gw)
        predictions = scale_prediction_totals_for_fixtures(predictions)
        predictions = blend_official_expected_points(predictions)
        predictions = blend_provider_expected_points(predictions)
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
            "status_persistent_flag",
            "status_return_date",
            "target_fixture_kickoff",
            "future_status_applied_flag",
            "fixture_multiplier",
            "cameo_points",
            "confidence_score",
            "confidence_level",
            "expected_points_lower_80",
            "expected_points_upper_80",
            "official_ep_next",
            "price_change_percent",
            "history_match_count",
            "current_season_match_count",
            "reliability_weight",
            "cold_start_bucket",
            "model_expected_points",
            "official_ep_blend_weight",
            "provider_expected_points",
            "provider_ep_blend_weight",
            "preseason_prior_quality",
            "start_prior_weight",
            "appearance_prior_weight",
            "role_change_score",
            "role_evidence_quality",
            "role_adjustment_weight",
            "role_ep_multiplier",
            "set_piece_duty_score",
            "primary_corners_taker",
            "primary_direct_fk_taker",
            "primary_penalty_taker",
            "team_prior_source",
            "opponent_team_prior_source",
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
        run_context = preseason_context.manifest_payload(
            cutoff=snapshot_cutoff.isoformat(),
            mode=snapshot_mode,
        )
        historical_only_rows = (
            int(histories_df["is_current_player_mapped"].eq(False).sum())
            if "is_current_player_mapped" in histories_df.columns
            else 0
        )
        run_context.update(
            {
                "feature_contract": "preseason-enrichment-v3",
                "next_gw": int(next_gw),
                "last_finished_gw": int(last_finished_gw),
                "history_rows": int(len(histories_df)),
                "historical_only_rows": historical_only_rows,
                "training_rows": int(len(X_train)),
                "prediction_players": int(len(predictions)),
                "reproducibility": {
                    "model_training_fingerprint": model_training_fingerprint,
                    "reused_fitted_models": reused_fitted_models,
                    "model_training_source_gameweek": model_training_source_gameweek,
                    "effective_config": {
                        "rolling_windows": list(model_config.ROLLING_WINDOWS),
                        "min_matches_for_features": int(
                            model_config.MIN_MATCHES_FOR_FEATURES
                        ),
                        "prediction_reliability_matches": int(
                            model_config.PREDICTION_RELIABILITY_MATCHES
                        ),
                        "preseason_prior_max_weight": float(
                            model_config.PRESEASON_PRIOR_MAX_WEIGHT
                        ),
                        "preseason_role_max_ep_adjustment": float(
                            model_config.PRESEASON_ROLE_MAX_EP_ADJUSTMENT
                        ),
                        "strict_gw1_player_evidence_max_age_days": int(
                            model_config.STRICT_GW1_PLAYER_EVIDENCE_MAX_AGE_DAYS
                        ),
                        "strict_gw1_team_prior_max_age_days": int(
                            model_config.STRICT_GW1_TEAM_PRIOR_MAX_AGE_DAYS
                        ),
                        "strict_gw1_min_evidenced_players_per_team": int(
                            model_config.STRICT_GW1_MIN_EVIDENCED_PLAYERS_PER_TEAM
                        ),
                        "strict_gw1_min_preseason_matches": int(
                            model_config.STRICT_GW1_MIN_PRESEASON_MATCHES
                        ),
                        "strict_gw1_min_source_confidence": float(
                            model_config.STRICT_GW1_MIN_SOURCE_CONFIDENCE
                        ),
                        "strict_gw1_min_player_evidence_quality": float(
                            model_config.STRICT_GW1_MIN_PLAYER_EVIDENCE_QUALITY
                        ),
                        "official_ep_blend_weight": float(
                            model_config.OFFICIAL_EP_BLEND_WEIGHT
                        ),
                        "official_ep_cold_start_blend_weight": float(
                            model_config.OFFICIAL_EP_COLD_START_BLEND_WEIGHT
                        ),
                        "provider_ep_blend_weight": float(
                            model_config.PROVIDER_EP_BLEND_WEIGHT
                        ),
                        "ema_alpha": float(model_config.EMA_ALPHA),
                        "season_weight_decay": float(
                            model_config.SEASON_WEIGHT_DECAY
                        ),
                        "season_weight_min": float(model_config.SEASON_WEIGHT_MIN),
                        "external_history_seasons": list(
                            model_config.EXTERNAL_HISTORY_SEASONS
                        ),
                    },
                    "runtime": {
                        "python": sys.version.split()[0],
                        "pandas": pd.__version__,
                        "numpy": np.__version__,
                    },
                },
            }
        )
        reproducibility_candidates = {
            "classifier.joblib": MODELS_DIR / "classifier.joblib",
            "appearance_classifier.joblib": MODELS_DIR
            / "appearance_classifier.joblib",
            "regressor.joblib": MODELS_DIR / "regressor.joblib",
            "cameo_points_by_position.joblib": MODELS_DIR
            / "cameo_points_by_position.joblib",
            "ensemble_models.joblib": MODELS_DIR / "ensemble_models.joblib",
            "model_state.json": MODELS_DIR / "state.json",
            "model_selection_summary.csv": MODELS_DIR
            / "model_selection_summary.csv",
            "source_main.py": model_config.PROJECT_ROOT / "main.py",
            "source_config.py": model_config.PROJECT_ROOT
            / "fplmodel"
            / "config.py",
            "source_features.py": model_config.PROJECT_ROOT
            / "fplmodel"
            / "features.py",
            "source_model.py": model_config.PROJECT_ROOT / "fplmodel" / "model.py",
            "source_evaluation.py": model_config.PROJECT_ROOT
            / "fplmodel"
            / "evaluation.py",
            "source_preseason.py": model_config.PROJECT_ROOT
            / "fplmodel"
            / "preseason.py",
            "source_snapshot_catalog.py": model_config.PROJECT_ROOT
            / "fplmodel"
            / "snapshot_catalog.py",
            "source_team_priors.py": model_config.PROJECT_ROOT
            / "fplmodel"
            / "team_priors.py",
            "source_history_universe.py": model_config.PROJECT_ROOT
            / "fplmodel"
            / "history_universe.py",
            "requirements.txt": model_config.PROJECT_ROOT / "requirements.txt",
        }
        reproducibility_files = {
            name: path
            for name, path in reproducibility_candidates.items()
            if path.is_file()
        }
        for snapshot_path, manifest in zip(
            preseason_context.snapshot_paths,
            preseason_context.manifests,
        ):
            topic = str(manifest.get("topic", "snapshot")).replace("/", "_")
            provider = str(manifest.get("provider", "provider")).replace("/", "_")
            reproducibility_files[f"snapshot_{topic}_{provider}.csv"] = snapshot_path
            manifest_path = SnapshotCatalog.manifest_path(snapshot_path)
            if manifest_path.is_file():
                reproducibility_files[
                    f"snapshot_{topic}_{provider}.manifest.json"
                ] = manifest_path
        predictions_archive_csv = archive_prediction_file(
            predictions_path=predictions_csv,
            output_root=OUTPUTS_DIR,
            bootstrap_path=bootstrap_source_path,
            season_name=current_season_name,
            gameweek=int(next_gw),
            fixtures_path=fixtures_source_path,
            context_manifest=run_context,
            reproducibility_files=reproducibility_files,
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
    snapshot_mode: str = "auto",
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
    model_cache: dict[str, object] = {}
    for idx, gw in enumerate(range(start_gw, end_gw + 1)):
        run_force_refetch = force_refetch if idx == 0 else False
        result = run_pipeline(
            force_refetch=run_force_refetch,
            override_next_gw=gw,
            override_last_finished_gw=max(gw - 1, 0),
            expected_season_name=expected_season_name,
            snapshot_mode=snapshot_mode,
            model_cache=model_cache,
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
    parser.add_argument(
        "--snapshot-mode",
        choices=("auto", "strict", "off"),
        default="auto",
        help=(
            "Use the latest enrichment snapshots before the target deadline (auto), "
            "require GW1 player/team snapshots (strict), or disable enrichment (off)."
        ),
    )
    args = parser.parse_args()

    if args.replay_start_gw is not None:
        results = replay_gameweeks(
            start_gw=args.replay_start_gw,
            end_gw=args.replay_end_gw,
            force_refetch=args.force_refetch,
            expected_season_name=args.expected_season,
            snapshot_mode=args.snapshot_mode,
        )
        print(json.dumps(results, indent=2))
    else:
        out = run_pipeline(
            force_refetch=args.force_refetch,
            override_next_gw=args.override_next_gw,
            override_last_finished_gw=args.override_last_finished_gw,
            expected_season_name=args.expected_season,
            snapshot_mode=args.snapshot_mode,
        )
        print(json.dumps(out, indent=2))
