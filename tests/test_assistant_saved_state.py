from copy import deepcopy
from unittest.mock import Mock

import pytest
import pandas as pd

from test_local_llm_assistant import _prediction_frame, _user_team
from fplmodel.season_manager import ManagerState
from local_llm_fpl_assistant import analytics


def test_saved_finances_reach_owned_squad_optimizer(monkeypatch):
    state = ManagerState(squad_player_ids=list(range(1, 16)), bank_m=0.5,
                         free_transfers=1, last_processed_gameweek=4,
                         season_name="2026-27",
                         purchase_price_by_player_id={i: 4.0 for i in range(1, 16)})
    original = deepcopy(state.to_dict())
    optimizer = Mock(return_value={"transfers": [], "gain": 0.0, "hit_cost": 0.0,
                                  "projected_total_before": 50, "projected_total_after": 50})
    monkeypatch.setattr(analytics, "_transfer_candidate", optimizer, raising=False)
    context = analytics.build_analysis_context(
        user_team=_user_team(), current_predictions=_prediction_frame(),
        predictions_by_gw={5: _prediction_frame()}, loaded_gws=[5], missing_gws=[],
        free_transfers=5, max_transfers=2, manager_state=state,
    )
    passed_state, _, gw, config = optimizer.call_args.args
    assert passed_state.bank_m == 0.5
    assert passed_state.free_transfers == 1
    assert passed_state.purchase_price_by_player_id == state.purchase_price_by_player_id
    assert gw == 5
    assert context["financial_context"]["bank_m"] == 0.5
    assert state.to_dict() == original


def test_analysis_rebuilds_when_planning_controls_change():
    from local_llm_fpl_assistant.app import refresh_analysis
    session = {}
    analyse = Mock(return_value=(_user_team(), {"horizon": 3}))
    team = _user_team()
    refresh_analysis(session, team, horizon=3, free_transfers=1, max_transfers=2, analyse=analyse)
    refresh_analysis(session, team, horizon=3, free_transfers=1, max_transfers=2, analyse=analyse)
    assert analyse.call_count == 1
    refresh_analysis(session, team, horizon=5, free_transfers=1, max_transfers=2, analyse=analyse)
    assert analyse.call_count == 2
    assert analyse.call_args.kwargs["horizon"] == 5
    analyse.side_effect = ValueError("missing predictions")
    with pytest.raises(ValueError):
        refresh_analysis(session, team, horizon=5, free_transfers=2, max_transfers=2, analyse=analyse)
    assert "local_llm_analysis_context" not in session


@pytest.mark.parametrize("bank,expected_transfers", [(0.0, 0), (5.0, 1)])
def test_saved_transfer_plan_obeys_sale_values_and_bank(bank, expected_transfers):
    frame = _prediction_frame()
    candidate = dict(frame.loc[frame.player_id == 6].iloc[0])
    candidate.update(player_id=16, full_name="Upgrade", team_id=7, team_name="New",
                     now_cost_millions=4.5, expected_points=30.0)
    frame = pd.concat([frame, pd.DataFrame([candidate])], ignore_index=True)
    state = ManagerState(squad_player_ids=list(range(1, 16)), bank_m=bank,
                         free_transfers=1, last_processed_gameweek=4, season_name="2026-27",
                         purchase_price_by_player_id={i: 0.1 for i in range(1, 16)})
    result = analytics.build_analysis_context(
        user_team=_user_team(), current_predictions=frame, predictions_by_gw={5: frame},
        loaded_gws=[5], missing_gws=[], free_transfers=5, max_transfers=1, manager_state=state,
    )
    assert len(result["transfer_plan"]["recommendations"]) == expected_transfers


def test_saved_prediction_horizon_rejects_wrong_season(tmp_path, monkeypatch):
    from local_llm_fpl_assistant import data_access
    monkeypatch.setattr(data_access, "OUTPUTS_DIR", tmp_path)
    _prediction_frame().assign(season_name="2025-26", gameweek=5).to_csv(tmp_path / "predictions_gw5.csv", index=False)
    with pytest.raises(ValueError, match="metadata"):
        data_access.load_predictions_for_horizon(3, start_gameweek=5, season_name="2026-27")


def test_saved_prediction_horizon_does_not_skip_missing_next_week(tmp_path, monkeypatch):
    from local_llm_fpl_assistant import data_access
    monkeypatch.setattr(data_access, "OUTPUTS_DIR", tmp_path)
    _prediction_frame().assign(season_name="2026-27", gameweek=6).to_csv(tmp_path / "predictions_gw6.csv", index=False)
    with pytest.raises(FileNotFoundError, match="GW5"):
        data_access.load_predictions_for_horizon(3, start_gameweek=5, season_name="2026-27")
