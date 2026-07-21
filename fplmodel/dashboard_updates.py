"""Dashboard-triggered refresh of live FPL data and prediction artifacts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


PipelineFunction = Callable[..., dict[str, object]]
ReplayFunction = Callable[..., list[dict[str, object]]]
ProgressCallback = Callable[[str], None]


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def refresh_dashboard_data(
    *,
    horizon: int = 4,
    run_pipeline_fn: PipelineFunction | None = None,
    replay_gameweeks_fn: ReplayFunction | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Refresh live inputs and rebuild predictions from the next GW onward.

    The live pipeline is run once with ``force_refetch=True``. Future horizon
    gameweeks are then generated without rerunning that first gameweek.
    """

    if int(horizon) < 1:
        raise ValueError("horizon must be >= 1")

    if run_pipeline_fn is None or replay_gameweeks_fn is None:
        from main import replay_gameweeks, run_pipeline

        run_pipeline_fn = run_pipeline_fn or run_pipeline
        replay_gameweeks_fn = replay_gameweeks_fn or replay_gameweeks

    _notify(
        progress_callback,
        "Refreshing official FPL data and rebuilding next-gameweek predictions...",
    )
    live_result = run_pipeline_fn(force_refetch=True)
    try:
        next_gameweek = int(live_result["next_gw"])
        season_name = str(live_result["season_name"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Live pipeline did not return valid season and gameweek data") from exc
    if not 1 <= next_gameweek <= 38:
        raise ValueError("Live pipeline next_gw must be between 1 and 38")

    end_gameweek = min(38, next_gameweek + int(horizon) - 1)
    results = [live_result]
    if end_gameweek > next_gameweek:
        _notify(
            progress_callback,
            f"Building future predictions for GW{next_gameweek + 1}-GW{end_gameweek}...",
        )
        future_results = replay_gameweeks_fn(
            start_gw=next_gameweek + 1,
            end_gw=end_gameweek,
            force_refetch=False,
            expected_season_name=season_name,
        )
        results.extend(future_results)

    gameweeks = list(range(next_gameweek, end_gameweek + 1))
    _notify(
        progress_callback,
        f"Updated {season_name} predictions for "
        f"GW{gameweeks[0]}{f'-GW{gameweeks[-1]}' if len(gameweeks) > 1 else ''}.",
    )
    return {
        "season_name": season_name,
        "start_gameweek": next_gameweek,
        "end_gameweek": end_gameweek,
        "gameweeks": gameweeks,
        "runs": results,
    }
