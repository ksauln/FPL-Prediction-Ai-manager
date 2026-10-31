import json

import pandas as pd
import pytest

from fplmodel.manager_forecasts import forecast_gameweek


def test_fixture_forecasts_are_scored_separately_and_blanks_are_zero():
    players = pd.DataFrame([
        dict(player_id=1, team_id=1, expected_points=99.),
        dict(player_id=2, team_id=2, expected_points=99.),
        dict(player_id=3, team_id=3, expected_points=99.),
        dict(player_id=4, team_id=4, expected_points=99.),
    ])
    fixtures = pd.DataFrame([
        dict(id=10, event=2, team_h=1, team_a=2),
        dict(id=11, event=2, team_h=3, team_a=1),
    ])
    calls = []

    def predict(fixture_frame, gw):
        fixture = fixture_frame.iloc[0]
        calls.append(int(fixture.id))
        result = players.copy()
        result["expected_points"] = 2. if fixture.id == 10 else 7.
        result["appearance_probability"] = .8
        result["start_probability"] = .6
        return result

    result = forecast_gameweek(players, fixtures, 2, predict).set_index("player_id")
    assert calls == [10, 11]
    assert result.loc[1, "expected_points"] == 9.
    assert result.loc[4, "expected_points"] == 0.
    assert result.loc[4, "appearance_probability"] == 0.
    assert result.loc[1, "appearance_probability"] == pytest.approx(.96)
    assert [f["expected_points"] for f in json.loads(result.loc[1, "fixture_forecasts"])] == [2., 7.]
    assert [f["opponent_id"] for f in json.loads(result.loc[1, "fixture_forecasts"])] == [2, 3]


def test_missing_fixture_predictions_fail_instead_of_silently_zeroing_player():
    players = pd.DataFrame([dict(player_id=1, team_id=1), dict(player_id=2, team_id=2)])
    fixtures = pd.DataFrame([dict(id=10, event=2, team_h=1, team_a=2)])
    with pytest.raises(ValueError, match="Missing"):
        forecast_gameweek(players, fixtures, 2, lambda *_: pd.DataFrame([dict(player_id=1, team_id=1, expected_points=3)]))
