import json

import pandas as pd
import pytest

from scripts.backtest_current_season import actual_outcomes, retarget_features


def test_retargeting_uses_origin_opponent_form_and_keeps_prices_and_player_form():
    base = pd.DataFrame([
        dict(player_id=1, team_id=1, now_cost_millions=5., goals_scored_ma5=.5,
             team_attack_ma5=2., opp_team_attack_ma5=99., status_persistent_flag=0,
             status_return_date=None, availability_next_round=.5, was_home=0., fixture_difficulty=5., official_ep_next=8.),
        dict(player_id=2, team_id=2, now_cost_millions=6., goals_scored_ma5=.2,
             team_attack_ma5=1., opp_team_attack_ma5=99., status_persistent_flag=0,
             status_return_date=None, availability_next_round=1., was_home=1., fixture_difficulty=5., official_ep_next=4.),
    ])
    saved = base.copy(deep=True)
    fixture = dict(team_h=1, team_a=2, team_h_difficulty=2, team_a_difficulty=4,
                   kickoff_time="2026-09-01T12:00:00Z")
    result = retarget_features(base, fixture, 3, 2).set_index("player_id")
    assert result.loc[1, "opp_team_attack_ma5"] == 1.
    assert result.loc[2, "opp_team_attack_ma5"] == 2.
    assert result.loc[1, "was_home"] == 1.
    assert result.loc[1, "fixture_difficulty"] == 2.
    assert result.loc[1, "goals_scored_ma5"] == .5
    assert result.loc[1, "now_cost_millions"] == 5.
    assert result.loc[1, "availability_next_round"] == 1.
    assert result.official_ep_next.isna().all()
    pd.testing.assert_frame_equal(base, saved)


def test_actuals_require_explicit_match_rows_instead_of_assuming_missing_is_dnp(tmp_path):
    predictions = pd.DataFrame({"player_id": [1]})
    path = tmp_path / "player_1.json"
    path.write_text(json.dumps({"history": []}))
    with pytest.raises(ValueError, match="Missing explicit actual"):
        actual_outcomes(predictions, 1, tmp_path)
    path.write_text(json.dumps({"history": [dict(round=1, fixture=1, minutes=0, total_points=0)]}))
    assert actual_outcomes(predictions, 1, tmp_path) == {1: (False, 0.)}
