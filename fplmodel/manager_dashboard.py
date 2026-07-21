"""Streamlit page for running and reviewing the stateful AI Manager."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from .config import OUTPUTS_DIR, RAW_DIR
from .manager_jobs import (
    cancel_manager_job,
    commit_recommended_state,
    create_manager_job,
    list_manager_jobs,
    load_manager_job,
    load_manager_job_result,
    start_manager_job,
)
from .prediction_artifacts import (
    PredictionSeasonSource,
    discover_prediction_season_sources,
    infer_bootstrap_season_name,
    inspect_prediction_artifacts,
    migrate_legacy_prediction_files,
)
from .season_manager import ManagerState, load_manager_state, load_prediction_files


POSITION_LABELS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SIMULATION_MODE_LABELS = {
    "periodic_reoptimization": "Periodic reoptimization",
    "fixed_policy": "Fixed policy",
    "full_reoptimization": "Full reoptimization",
}


def _default_season_name(
    now: Optional[datetime] = None,
    *,
    bootstrap_path: Path = RAW_DIR / "bootstrap-static.json",
) -> str:
    bootstrap_season = infer_bootstrap_season_name(bootstrap_path)
    if bootstrap_season is not None:
        return bootstrap_season
    now = now or datetime.now()
    start_year = now.year if now.month >= 7 else now.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _format_duration(seconds: object) -> str:
    if seconds is None:
        return "Calculating"
    try:
        total = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "Calculating"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _prediction_gameweeks(output_dir: Path = OUTPUTS_DIR) -> list[int]:
    gameweeks: list[int] = []
    for path in Path(output_dir).glob("predictions_gw*.csv"):
        match = re.fullmatch(r"predictions_gw(\d+)", path.stem)
        if match and 1 <= int(match.group(1)) <= 38:
            gameweeks.append(int(match.group(1)))
    return sorted(set(gameweeks))


@st.cache_data(ttl=30)
def _season_sources() -> dict[str, PredictionSeasonSource]:
    return discover_prediction_season_sources(
        output_root=OUTPUTS_DIR,
        current_bootstrap_path=RAW_DIR / "bootstrap-static.json",
    )


def _select_season_source(
    sources: dict[str, PredictionSeasonSource],
    *,
    default_season: str,
    container,
) -> tuple[str, PredictionSeasonSource]:
    season_options = list(sources)
    default_index = (
        season_options.index(default_season)
        if default_season in season_options
        else 0
    )
    season_name = container.selectbox(
        "Season to simulate",
        options=season_options,
        index=default_index,
    )
    return season_name, sources[season_name]


@st.cache_data(ttl=30)
def _artifact_inspection(
    season_name: str,
    predictions_dir: str,
    bootstrap_path: str,
):
    return inspect_prediction_artifacts(
        output_dir=Path(predictions_dir),
        bootstrap_path=Path(bootstrap_path),
        expected_season_name=season_name,
    )


def _state_path(season_name: str) -> Path:
    return OUTPUTS_DIR / f"manager_state_{season_name}.json"


def _load_state_if_available(season_name: str):
    path = _state_path(season_name)
    if not path.exists():
        return None
    return load_manager_state(path)


def _status_label(job: dict[str, object]) -> str:
    request = dict(job.get("request", {}))
    created = str(job.get("created_at", ""))[:19].replace("T", " ")
    return (
        f"{created} | {request.get('season_name', 'season')} | "
        f"GW{request.get('start_gw', '?')}-{request.get('end_gw', '?')} | "
        f"{int(request.get('simulations', 0)):,} | {job.get('status', 'unknown')}"
    )


def _decision_frame(decisions: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for decision in decisions:
        transfers = list(decision.get("transfers", []))
        rows.append(
            {
                "GW": int(decision["gameweek"]),
                "Expected pts": round(float(decision.get("expected_points", 0.0)), 2),
                "Captain": decision.get("captain") or "",
                "Vice-captain": decision.get("vice_captain") or "",
                "Chip": str(decision.get("chip") or "").replace("_", " ").title(),
                "Lineup context": (
                    "Free Hit"
                    if decision.get("team_context") == "free_hit"
                    else "Owned squad"
                ),
                "Transfers": len(transfers),
                "Hit": round(float(decision.get("transfer_hit_cost", 0.0)), 1),
                "Free transfers": int(decision.get("free_transfers_after", 0)),
                "Owned bank (£m)": round(float(decision.get("bank_m", 0.0)), 1),
                "Owned team value (£m)": round(float(decision.get("team_value_m", 0.0)), 1),
            }
        )
    return pd.DataFrame(rows)


def _team_frame(
    players: list[dict[str, object]],
    decision: dict[str, object],
    *,
    bench: bool,
) -> pd.DataFrame:
    captain_id = decision.get("captain_id")
    vice_id = decision.get("vice_captain_id")
    rows = []
    for index, player in enumerate(players, start=1):
        player_id = int(player["player_id"])
        role = "Starter"
        if player_id == captain_id:
            role = "Captain"
        elif player_id == vice_id:
            role = "Vice-captain"
        elif bench:
            order = player.get("bench_order") or index
            role = f"Bench {int(order)}"
        rows.append(
            {
                "Pos": POSITION_LABELS.get(int(player.get("element_type", 0)), ""),
                "Player": player.get("full_name", ""),
                "Club": player.get("team_name", ""),
                "Role": role,
                "Cost (£m)": round(float(player.get("now_cost_millions", 0.0)), 1),
                "Expected pts": round(float(player.get("expected_points", 0.0)), 2),
                "Start %": round(float(player.get("start_probability", 0.0)) * 100.0),
                "Confidence": player.get("confidence_level", ""),
            }
        )
    return pd.DataFrame(rows)


def _transfer_frame(decision: dict[str, object]) -> pd.DataFrame:
    rows = []
    for transfer in decision.get("transfers", []):
        outgoing = dict(transfer.get("out_player", {}))
        incoming = dict(transfer.get("in_player", {}))
        rows.append(
            {
                "Sell": outgoing.get("full_name", ""),
                "Buy": incoming.get("full_name", ""),
                "Sell value (£m)": round(float(transfer.get("out_sale_value", 0.0)), 1),
                "Buy price (£m)": round(float(transfer.get("in_purchase_price", 0.0)), 1),
                "Projected gain": round(
                    float(transfer.get("expected_points_delta", 0.0)), 2
                ),
            }
        )
    return pd.DataFrame(rows)


def _parse_topups(value: str) -> dict[int, int]:
    value = value.strip()
    if not value:
        return {}
    topups: dict[int, int] = {}
    for item in value.split(","):
        try:
            gameweek, count = item.split(":", maxsplit=1)
            topups[int(gameweek.strip())] = int(count.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError("Free-transfer top-ups must use GW:count, for example 16:5") from exc
    return topups


def _render_data_readiness(
    season_name: str,
    source: PredictionSeasonSource,
) -> list[int]:
    gameweeks = _prediction_gameweeks(source.predictions_dir)
    bootstrap_path = source.bootstrap_path
    bootstrap_season = infer_bootstrap_season_name(bootstrap_path)
    inspection = None
    inspection_error = None
    try:
        inspection = _artifact_inspection(
            season_name,
            str(source.predictions_dir),
            str(source.bootstrap_path),
        )
    except (OSError, ValueError) as exc:
        inspection_error = str(exc)
    tagged_gameweeks = list(inspection.ready_gameweeks) if inspection else []
    legacy_gameweeks = list(inspection.migratable_gameweeks) if inspection else []
    state = None
    state_error = None
    try:
        state = _load_state_if_available(season_name)
    except (OSError, ValueError, KeyError) as exc:
        state_error = str(exc)

    columns = st.columns(4)
    columns[0].metric("Prediction files", len(gameweeks))
    columns[1].metric(
        "Prediction range",
        f"GW{gameweeks[0]}-GW{gameweeks[-1]}" if gameweeks else "None",
    )
    columns[2].metric("Season-ready files", f"{len(tagged_gameweeks)}/38")
    columns[3].metric(
        "Live manager state",
        f"Through GW{state.last_processed_gameweek}" if state else "Not started",
    )
    if state_error:
        st.error(f"Manager state could not be loaded: {state_error}")
    if inspection_error:
        st.error(f"Prediction files could not be validated: {inspection_error}")
    if bootstrap_season is not None and bootstrap_season != season_name:
        st.warning(
            f"Cached official FPL data is still for {bootstrap_season}. "
            f"Predictions cannot be prepared for {season_name} until the official "
            "game data changes."
        )
    elif legacy_gameweeks:
        st.warning(
            f"Found {len(legacy_gameweeks)} legacy prediction files without season "
            f"metadata. They can be validated against the cached {season_name} FPL data."
        )
        if st.button("Validate and tag legacy files", type="secondary"):
            try:
                with st.spinner("Validating player and team identities..."):
                    migration = migrate_legacy_prediction_files(
                        output_dir=source.predictions_dir,
                        bootstrap_path=bootstrap_path,
                        expected_season_name=season_name,
                    )
                _artifact_inspection.clear()
                _season_sources.clear()
                st.success(
                    f"Prepared {len(migration.migrated_gameweeks)} prediction files. "
                    f"Backups: {migration.backup_dir}"
                )
                st.rerun()
            except (FileNotFoundError, OSError, ValueError) as exc:
                st.error(str(exc))
    if (
        inspection is not None
        and inspection.issues
        and bootstrap_season == season_name
    ):
        preview = "; ".join(
            f"GW{gameweek}: {message}"
            for gameweek, message in inspection.issues[:3]
        )
        remaining = len(inspection.issues) - 3
        suffix = f" (+{remaining} more)" if remaining > 0 else ""
        st.error(
            f"Prediction file issues: {preview}{suffix} Regenerate the affected "
            "gameweeks with the current pipeline."
        )
    return tagged_gameweeks


@st.fragment(run_every="2s")
def _render_active_job(job_id: str) -> None:
    try:
        job = load_manager_job(job_id)
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(str(exc))
        return
    status = str(job["status"])
    progress = float(job.get("progress", 0.0))
    completed = int(job.get("completed_simulations", 0))
    total = int(job.get("total_simulations", 0))
    phase = str(job.get("phase", status)).replace("_", " ").title()
    st.progress(
        progress,
        text=f"{phase} | {completed:,}/{total:,} simulations | ETA {_format_duration(job.get('eta_seconds'))}",
    )
    columns = st.columns(3)
    columns[0].metric("Elapsed", _format_duration(job.get("elapsed_seconds")))
    columns[1].metric("Policy block", job.get("policy_block") or "-")
    columns[2].metric("Current gameweek", job.get("current_gameweek") or "-")
    if status == "running" and st.button("Cancel run", key=f"cancel_{job_id}"):
        cancel_manager_job(job_id)
        st.rerun()
    elif status == "failed":
        st.error(str(job.get("error", "The simulation worker failed.")))
    elif status == "completed":
        st.success("Simulation complete. Open Review results to inspect the policy.")
    elif status == "canceled":
        st.warning("Simulation canceled.")


def _latest_active_job(jobs: list[dict[str, object]]) -> Optional[dict[str, object]]:
    return next(
        (job for job in jobs if job.get("status") in {"queued", "running"}),
        None,
    )


def _render_run_view(
    season_name: str,
    source: PredictionSeasonSource,
) -> None:
    tagged_gameweeks = _render_data_readiness(season_name, source)
    jobs = list_manager_jobs()
    active_job = _latest_active_job(jobs)
    if active_job:
        st.subheader("Active run")
        _render_active_job(str(active_job["job_id"]))

    st.subheader("Simulation settings")
    run_mode = st.segmented_control(
        "Manager mode",
        options=["planning", "live"],
        format_func=lambda value: "Season planning" if value == "planning" else "Live gameweek",
        default="planning",
        key="ai_manager_run_mode",
    )
    state = None
    state_error = None
    if run_mode == "live":
        try:
            state = _load_state_if_available(season_name)
        except (OSError, ValueError, KeyError) as exc:
            state_error = str(exc)
            st.error(f"Manager state could not be loaded: {state_error}")

    expected_start = state.last_processed_gameweek + 1 if state else 1
    start_key = f"ai_manager_start_{season_name}_{run_mode}"
    if state is not None:
        st.session_state[start_key] = min(38, expected_start)
    settings = st.columns([1, 1, 1.4])
    start_gw = int(
        settings[0].number_input(
            "Start gameweek",
            min_value=1,
            max_value=38,
            value=min(38, expected_start),
            disabled=state is not None,
            key=start_key,
        )
    )
    default_end = 38 if run_mode == "planning" else min(38, start_gw + 3)
    end_gw = int(
        settings[1].number_input(
            "End gameweek",
            min_value=start_gw,
            max_value=38,
            value=default_end,
            key=f"ai_manager_end_{season_name}_{run_mode}",
        )
    )
    simulation_mode = settings[2].selectbox(
        "Simulation mode",
        options=list(SIMULATION_MODE_LABELS),
        format_func=SIMULATION_MODE_LABELS.get,
        index=0,
    )

    sizing = st.columns(3)
    simulations = int(
        sizing[0].number_input(
            "Simulations", min_value=1, max_value=1_000_000, value=50_000, step=1_000
        )
    )
    policy_refresh_interval = int(
        sizing[1].number_input(
            "Reoptimize every",
            min_value=1,
            max_value=max(1, simulations),
            value=min(1_000, simulations),
            step=min(100, simulations),
            disabled=simulation_mode != "periodic_reoptimization",
        )
    )
    random_seed = int(
        sizing[2].number_input(
            "Random seed", min_value=0, max_value=2_147_483_647, value=90
        )
    )

    with st.expander("Strategy settings"):
        horizons = st.columns(4)
        initial_horizon = int(
            horizons[0].number_input("Initial horizon", 1, 12, 4)
        )
        transfer_horizon = int(
            horizons[1].number_input("Transfer horizon", 1, 12, 4)
        )
        chip_lookahead = int(
            horizons[2].number_input("Chip lookahead", 1, 20, 4)
        )
        max_transfers = int(
            horizons[3].number_input("Max transfers/GW", 0, 15, 2)
        )
        thresholds = st.columns(4)
        transfer_threshold = float(
            thresholds[0].number_input("Transfer gain", 0.0, 20.0, 1.5, 0.1)
        )
        wildcard_threshold = float(
            thresholds[1].number_input("Wildcard gain", 0.0, 50.0, 16.0, 0.5)
        )
        free_hit_threshold = float(
            thresholds[2].number_input("Free Hit gain", 0.0, 50.0, 10.0, 0.5)
        )
        bench_boost_threshold = float(
            thresholds[3].number_input("Bench Boost gain", 0.0, 30.0, 8.0, 0.5)
        )
        chip_settings = st.columns(4)
        triple_captain_threshold = float(
            chip_settings[0].number_input("Triple Captain gain", 0.0, 30.0, 5.0, 0.5)
        )
        future_value_ratio = float(
            chip_settings[1].number_input("Future chip value", 0.0, 2.0, 0.95, 0.05)
        )
        noise_scale = float(
            chip_settings[2].number_input("Uncertainty scale", 0.0, 5.0, 1.0, 0.1)
        )
        enable_chips = chip_settings[3].toggle("Automatic chips", value=True)

    with st.expander("Season rules"):
        rules = st.columns(5)
        budget_m = float(rules[0].number_input("Starting budget (£m)", 50.0, 200.0, 100.0, 0.5))
        max_free_transfers = int(rules[1].number_input("Max free transfers", 1, 15, 5))
        transfer_hit_cost = float(rules[2].number_input("Transfer hit", 0.0, 20.0, 4.0, 1.0))
        first_half_end_gw = int(rules[3].number_input("First-half end GW", 1, 37, 19))
        free_transfers_per_gw = int(rules[4].number_input("Free transfers/GW", 0, 5, 1))
        free_transfer_topups = st.text_input("Free-transfer top-ups", value="")

    missing_ready_gameweeks = sorted(
        set(range(start_gw, end_gw + 1)) - set(tagged_gameweeks)
    )
    if missing_ready_gameweeks:
        st.info(
            "Prepare season-tagged prediction files for the selected range before "
            "starting the simulation."
        )
    run_disabled = (
        active_job is not None
        or state_error is not None
        or bool(missing_ready_gameweeks)
    )
    if st.button(
        "Run AI Manager",
        type="primary",
        width="stretch",
        disabled=run_disabled,
    ):
        try:
            predictions = load_prediction_files(
                source.predictions_dir,
                start_gw=start_gw,
                end_gw=end_gw,
                expected_season_name=season_name,
            )
            missing = [gw for gw in range(start_gw, end_gw + 1) if gw not in predictions]
            if missing:
                raise FileNotFoundError(f"Missing prediction files for gameweeks: {missing}")
            request = {
                "season_name": season_name,
                "run_mode": run_mode,
                "start_gw": start_gw,
                "end_gw": end_gw,
                "simulations": simulations,
                "simulation_mode": simulation_mode,
                "policy_refresh_interval": policy_refresh_interval,
                "random_seed": random_seed,
                "output_dir": str(source.predictions_dir),
                "manager_state_path": str(_state_path(season_name)),
                "manager_config": {
                    "initial_horizon": initial_horizon,
                    "transfer_horizon": transfer_horizon,
                    "chip_lookahead": chip_lookahead,
                    "transfer_gain_threshold": transfer_threshold,
                    "max_transfers_per_gw": max_transfers,
                    "enable_chips": enable_chips,
                    "bench_boost_gain_threshold": bench_boost_threshold,
                    "triple_captain_gain_threshold": triple_captain_threshold,
                    "free_hit_gain_threshold": free_hit_threshold,
                    "wildcard_gain_threshold": wildcard_threshold,
                    "chip_future_value_ratio": future_value_ratio,
                    "monte_carlo_noise_scale": noise_scale,
                },
                "rules": {
                    "budget_m": budget_m,
                    "max_free_transfers": max_free_transfers,
                    "transfer_hit_cost": transfer_hit_cost,
                    "first_half_end_gw": first_half_end_gw,
                    "free_transfer_per_gameweek": free_transfers_per_gw,
                    "free_transfer_topups": _parse_topups(free_transfer_topups),
                },
            }
            job = create_manager_job(request)
            start_manager_job(str(job["job_id"]))
            st.session_state["ai_manager_selected_job"] = job["job_id"]
            st.rerun()
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))


def _render_run_summary(result: dict[str, object]) -> None:
    summary = dict(result.get("summary", {}))
    columns = st.columns(5)
    columns[0].metric("Simulations", f"{int(summary.get('simulations', 0)):,}")
    columns[1].metric("Average points", f"{float(summary.get('average_total_expected_points', 0.0)):.1f}")
    columns[2].metric("Median points", f"{float(summary.get('median_total_expected_points', 0.0)):.1f}")
    columns[3].metric("Pessimistic", f"{float(summary.get('min_total_expected_points', 0.0)):.1f}")
    columns[4].metric("Optimistic", f"{float(summary.get('max_total_expected_points', 0.0)):.1f}")

    runs = list(result.get("runs", []))
    totals = [float(run["summary"]["total_expected_points"]) for run in runs]
    if totals:
        counts, edges = np.histogram(totals, bins=min(30, max(5, int(np.sqrt(len(totals))))))
        distribution = pd.DataFrame(
            {
                "Total points": (edges[:-1] + edges[1:]) / 2.0,
                "Simulations": counts,
            }
        )
        st.bar_chart(distribution, x="Total points", y="Simulations")


def _render_policy_comparison(result: dict[str, object]) -> None:
    rows = []
    for policy in result.get("policy_runs", []):
        evaluation = dict(policy.get("policy_evaluation", {}))
        rows.append(
            {
                "Policy block": policy.get("policy_block"),
                "Average points": round(float(evaluation.get("average_total_points", 0.0)), 2),
                "Median points": round(float(evaluation.get("median_total_points", 0.0)), 2),
                "P10": round(float(evaluation.get("p10_total_points", 0.0)), 2),
                "P90": round(float(evaluation.get("p90_total_points", 0.0)), 2),
                "Simulations": evaluation.get("simulations", 0),
            }
        )
    if rows:
        st.subheader("Policy comparison")
        st.dataframe(
            pd.DataFrame(rows).sort_values("Average points", ascending=False),
            width="stretch",
            hide_index=True,
        )


def _render_gameweek_decision(decision: dict[str, object]) -> None:
    is_free_hit = decision.get("team_context") == "free_hit" or decision.get("chip") == "free_hit"
    columns = st.columns(6 if is_free_hit else 5)
    columns[0].metric("Expected points", f"{float(decision.get('expected_points', 0.0)):.1f}")
    metric_offset = 0
    if is_free_hit:
        columns[1].metric(
            "Free Hit squad cost",
            f"£{float(decision.get('displayed_squad_cost_m', 0.0)):.1f}m",
        )
        metric_offset = 1
    columns[1 + metric_offset].metric(
        "Owned bank" if is_free_hit else "Bank",
        f"£{float(decision.get('bank_m', 0.0)):.1f}m",
    )
    columns[2 + metric_offset].metric(
        "Owned team value" if is_free_hit else "Team value",
        f"£{float(decision.get('team_value_m', 0.0)):.1f}m",
    )
    columns[3 + metric_offset].metric(
        "Free transfers", int(decision.get("free_transfers_after", 0))
    )
    columns[4 + metric_offset].metric(
        "Chip", str(decision.get("chip") or "None").replace("_", " ").title()
    )

    team = dict(decision.get("team", {}))
    starter_column, bench_column = st.columns([2.2, 1])
    with starter_column:
        st.subheader("Free Hit starting XI" if is_free_hit else "Starting XI")
        st.dataframe(
            _team_frame(list(team.get("squad", [])), decision, bench=False),
            width="stretch",
            hide_index=True,
        )
    with bench_column:
        st.subheader("Free Hit bench" if is_free_hit else "Bench")
        st.dataframe(
            _team_frame(list(team.get("bench", [])), decision, bench=True),
            width="stretch",
            hide_index=True,
        )

    transfers = _transfer_frame(decision)
    st.subheader("Transfers")
    if transfers.empty:
        st.info("No transfers recommended.")
    else:
        st.dataframe(transfers, width="stretch", hide_index=True)


def _render_state_commit(job: dict[str, object], result: dict[str, object]) -> None:
    recommended = dict(result.get("recommended_policy", result.get("best_run", {})))
    decisions = list(recommended.get("decisions", []))
    if not decisions or not job.get("recommended_state_path"):
        return
    first_gameweek = int(decisions[0]["gameweek"])
    request = dict(job.get("request", {}))
    destination = Path(str(request.get("manager_state_path", "")))
    if first_gameweek != 1 and not destination.exists():
        st.warning("A live state must be started from a GW1 recommendation.")
        return
    st.subheader("Advance live state")
    if job.get("state_committed_at"):
        st.success(
            f"Live manager state saved through GW{first_gameweek} at "
            f"{str(job['state_committed_at'])[:19].replace('T', ' ')} UTC."
        )
        return
    confirmed = st.checkbox(
        f"I applied the GW{first_gameweek} squad, transfers, lineup, captaincy, and chip in FPL.",
        key=f"confirm_state_{job['job_id']}",
    )
    if st.button(
        f"Save state through GW{first_gameweek}",
        disabled=not confirmed,
        key=f"commit_state_{job['job_id']}",
    ):
        commit_recommended_state(str(job["job_id"]), destination)
        st.rerun()


def _manager_state_squad_frame(
    state: ManagerState,
    predictions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    lookup = (
        predictions.drop_duplicates("player_id").set_index("player_id")
        if predictions is not None and not predictions.empty
        else pd.DataFrame()
    )
    rows = []
    for player_id in state.squad_player_ids:
        player = lookup.loc[player_id] if player_id in lookup.index else None
        rows.append(
            {
                "Player": player.get("full_name", "") if player is not None else "",
                "Club": player.get("team_name", "") if player is not None else "",
                "Pos": (
                    POSITION_LABELS.get(int(player.get("element_type", 0)), "")
                    if player is not None
                    else ""
                ),
                "Player ID": player_id,
                "Purchase price (£m)": state.purchase_price_by_player_id.get(player_id),
                "Current price (£m)": (
                    round(float(player.get("now_cost_millions", 0.0)), 1)
                    if player is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_results_view(season_name: str) -> None:
    jobs = [
        job
        for job in list_manager_jobs()
        if str(dict(job.get("request", {})).get("season_name")) == season_name
    ]
    if not jobs:
        st.info(f"No AI Manager runs are available for {season_name}.")
        return
    lookup = {str(job["job_id"]): job for job in jobs}
    job_ids = list(lookup)
    selected = st.session_state.get("ai_manager_selected_job")
    index = job_ids.index(selected) if selected in job_ids else 0
    job_id = st.selectbox(
        "Saved run",
        options=job_ids,
        index=index,
        format_func=lambda value: _status_label(lookup[value]),
    )
    st.session_state["ai_manager_selected_job"] = job_id
    job = load_manager_job(job_id)
    if job["status"] in {"queued", "running"}:
        _render_active_job(job_id)
        return
    if job["status"] == "failed":
        st.error(str(job.get("error", "The simulation worker failed.")))
        return
    if job["status"] == "canceled":
        st.warning("This run was canceled.")
        return

    result = load_manager_job_result(job_id)
    _render_run_summary(result)
    _render_policy_comparison(result)
    recommended = dict(result.get("recommended_policy", result.get("best_run", {})))
    decisions = list(recommended.get("decisions", []))
    st.subheader("Recommended season policy")
    st.dataframe(
        _decision_frame(decisions), width="stretch", hide_index=True
    )
    if decisions:
        gameweeks = [int(decision["gameweek"]) for decision in decisions]
        selected_gw = st.selectbox("Gameweek decision", options=gameweeks)
        decision = next(
            item for item in decisions if int(item["gameweek"]) == selected_gw
        )
        _render_gameweek_decision(decision)
    _render_state_commit(job, result)


def _render_manager_state_view(
    season_name: str,
    source: PredictionSeasonSource,
) -> None:
    path = _state_path(season_name)
    if not path.exists():
        st.info(f"No live manager state exists for {season_name}.")
        return
    try:
        state = load_manager_state(path)
    except (OSError, ValueError, KeyError) as exc:
        st.error(str(exc))
        return
    columns = st.columns(4)
    columns[0].metric("Processed through", f"GW{state.last_processed_gameweek}")
    columns[1].metric("Bank", f"£{state.bank_m:.1f}m")
    columns[2].metric("Free transfers", state.free_transfers)
    columns[3].metric("Squad players", len(state.squad_player_ids))

    predictions = None
    available = _prediction_gameweeks(source.predictions_dir)
    next_gameweeks = [gw for gw in available if gw > state.last_processed_gameweek]
    target_gameweek = next_gameweeks[0] if next_gameweeks else (available[-1] if available else None)
    if target_gameweek is not None:
        try:
            predictions = load_prediction_files(
                source.predictions_dir,
                start_gw=target_gameweek,
                end_gw=target_gameweek,
                expected_season_name=season_name,
            )[target_gameweek]
        except (FileNotFoundError, OSError, ValueError):
            predictions = None
    squad = _manager_state_squad_frame(state, predictions)
    st.subheader("Owned squad")
    st.dataframe(squad, width="stretch", hide_index=True)
    st.subheader("Chip usage")
    chip_rows = [
        {"Season half": half.title(), "Chip": chip.replace("_", " ").title(), "GW": gameweek}
        for half, chips in state.used_chips.items()
        for chip, gameweek in chips.items()
    ]
    if chip_rows:
        st.dataframe(pd.DataFrame(chip_rows), width="stretch", hide_index=True)
    else:
        st.info("No chips recorded.")
    if state.history:
        st.subheader("Committed decisions")
        st.dataframe(
            _decision_frame(state.history), width="stretch", hide_index=True
        )


def render_ai_manager_page() -> None:
    st.header("AI Manager")
    sources = _season_sources()
    if not sources:
        st.error(
            "No simulation seasons are available. Run the prediction pipeline to "
            "create season-tagged gameweek files."
        )
        return
    default_season = _default_season_name()
    heading = st.columns([1.3, 2])
    season_name, source = _select_season_source(
        sources,
        default_season=default_season,
        container=heading[0],
    )
    view = heading[1].segmented_control(
        "View",
        options=["run", "results", "state"],
        format_func=lambda value: {
            "run": "Run simulations",
            "results": "Review results",
            "state": "Manager state",
        }[value],
        default="run",
        key="ai_manager_view",
    )
    if view == "run":
        _render_run_view(season_name, source)
    elif view == "results":
        _render_results_view(season_name)
    else:
        _render_manager_state_view(season_name, source)


__all__ = [
    "_decision_frame",
    "_default_season_name",
    "_format_duration",
    "_parse_topups",
    "_select_season_source",
    "render_ai_manager_page",
]
