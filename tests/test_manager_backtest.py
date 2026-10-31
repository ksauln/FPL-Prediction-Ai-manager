from copy import deepcopy
from datetime import datetime, timezone

import pytest

from fplmodel.manager_backtest import ForecastVintage, historical_competition
from test_manager_v2 import config, frames, state


def vintage(origin, data):
    return ForecastVintage(origin_gw=origin, season_name="2026-27",
        created_at=f"2026-10-{origin:02d}T10:00:00+00:00",
        inputs_as_of=f"2026-10-{origin:02d}T09:00:00+00:00",
        trained_through_gw=origin - 1, predictions=data,
        deadline=f"2026-10-{origin:02d}T11:00:00+00:00")


def test_historical_competition_uses_deadline_vintages_and_preserves_start():
    data, initial = frames((2, 3)), state()
    saved = initial.to_dict()
    vintages = {gw: vintage(gw, {k: v for k, v in data.items() if k >= gw}) for gw in data}
    actuals = {gw: {pid: (True, 4.) for pid in range(1, 21)} for gw in data}
    result = historical_competition(vintages, actuals, initial_state=initial,
                                   upgraded_config=config(), simulations=4, seed=1)
    assert initial.to_dict() == saved
    assert set(result["managers"]) == {"legacy", "v2", "hold"}
    for manager in result["managers"].values():
        assert manager["gameweeks"] == [2, 3]
        assert manager["net_actual_points"] == pytest.approx(sum(w["net_actual_points"] for w in manager["weekly"]))
    assert result["deadline_safe"] is True


@pytest.mark.parametrize("change", [
    {"created_at": "2026-10-02T12:00:00+00:00"},
    {"inputs_as_of": "2026-10-02T12:00:00+00:00"},
    {"trained_through_gw": 2},
    {"created_at": "2026-10-02T10:00:00"},
])
def test_backtest_rejects_late_or_unverifiable_inputs(change):
    from dataclasses import replace
    data = frames((2,))
    with pytest.raises(ValueError):
        historical_competition({2: replace(vintage(2, data), **change)},
            {2: {pid: (True, 4.) for pid in range(1, 21)}},
            initial_state=state(), upgraded_config=config(), simulations=2)


def test_backtest_rejects_missing_actuals_and_future_price_leakage():
    data = frames((2, 3))
    data[3]["now_cost_millions"] = 6.
    with pytest.raises(ValueError, match="price"):
        historical_competition({2: vintage(2, data)}, {}, initial_state=state(), upgraded_config=config())
    with pytest.raises(ValueError, match="actual"):
        historical_competition({2: vintage(2, frames((2,)))}, {2: {}},
                               initial_state=state(), upgraded_config=config())


def test_reconstruction_requires_explicit_opt_in_and_is_not_claimed_as_original():
    from dataclasses import asdict
    from fplmodel.manager_backtest import ReconstructedVintage
    data = frames((2,))
    rebuilt = ReconstructedVintage(**asdict(vintage(2, data)),
        reconstructed_at="2026-10-31T10:00:00+00:00", history_through_gw=1)
    actuals = {2: {pid: (True, 4.) for pid in range(1, 21)}}
    with pytest.raises(ValueError, match="explicit"):
        historical_competition({2: rebuilt}, actuals, initial_state=state(), upgraded_config=config())
    result = historical_competition({2: rebuilt}, actuals, initial_state=state(),
        upgraded_config=config(), simulations=2, allow_reconstructed=True)
    assert result["as_of_reconstruction"] is True
    assert result["deadline_safe"] is False
