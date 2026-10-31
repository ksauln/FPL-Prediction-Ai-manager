from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fplmodel.features import _assign_pre_round_features, build_training_and_pred_frames
from fplmodel.state import ModelState


def _elements_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "code": 1001,
                "full_name": "Player One",
                "team_id": 1,
                "element_type": 3,
                "now_cost_millions": 7.5,
                "minutes": 0,
                "status": "a",
            },
            {
                "player_id": 2,
                "code": 1002,
                "full_name": "Player Two",
                "team_id": 2,
                "element_type": 4,
                "now_cost_millions": 8.0,
                "minutes": 0,
                "status": "a",
            },
        ]
    )


def _teams_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team_id": 1, "name": "Alpha", "short_name": "ALP"},
            {"team_id": 2, "name": "Beta", "short_name": "BET"},
        ]
    )


def _history_row(
    *,
    round_number: int,
    fixture: int,
    kickoff_time: str,
    total_points: float,
    team_h_score: int,
    team_a_score: int,
) -> dict[str, object]:
    return {
        "player_id": 1,
        "season_name": "2025-26",
        "round": round_number,
        "fixture": fixture,
        "kickoff_time": kickoff_time,
        "total_points": total_points,
        "minutes": 90,
        "was_home": True,
        "team": 1,
        "opponent_team": 2,
        "team_h_score": team_h_score,
        "team_a_score": team_a_score,
        "value": 75,
    }


def test_double_gameweek_training_rows_share_same_pre_round_snapshot() -> None:
    histories = pd.DataFrame(
        [
            _history_row(
                round_number=1,
                fixture=101,
                kickoff_time="2025-08-01T12:00:00Z",
                total_points=1.0,
                team_h_score=1,
                team_a_score=0,
            ),
            _history_row(
                round_number=2,
                fixture=102,
                kickoff_time="2025-08-08T12:00:00Z",
                total_points=2.0,
                team_h_score=2,
                team_a_score=0,
            ),
            _history_row(
                round_number=3,
                fixture=103,
                kickoff_time="2025-08-15T12:00:00Z",
                total_points=3.0,
                team_h_score=3,
                team_a_score=0,
            ),
            _history_row(
                round_number=4,
                fixture=104,
                kickoff_time="2025-08-22T12:00:00Z",
                total_points=4.0,
                team_h_score=1,
                team_a_score=1,
            ),
            _history_row(
                round_number=5,
                fixture=105,
                kickoff_time="2025-08-29T12:00:00Z",
                total_points=5.0,
                team_h_score=2,
                team_a_score=1,
            ),
            _history_row(
                round_number=6,
                fixture=106,
                kickoff_time="2025-09-05T12:00:00Z",
                total_points=9.0,
                team_h_score=4,
                team_a_score=0,
            ),
            _history_row(
                round_number=6,
                fixture=107,
                kickoff_time="2025-09-07T15:00:00Z",
                total_points=10.0,
                team_h_score=0,
                team_a_score=1,
            ),
        ]
    )

    with TemporaryDirectory() as tmpdir:
        X_train, y_train, _, train_metadata = build_training_and_pred_frames(
            _elements_df(),
            _teams_df(),
            histories,
            next_gw=7,
            last_finished_gw=6,
            state=ModelState(path=Path(tmpdir) / "state.json"),
            current_season_name="2025-26",
        )

    round_six = train_metadata.loc[pd.to_numeric(train_metadata["round"], errors="coerce").eq(6)].copy()
    round_six["target_points"] = y_train.loc[round_six.index].to_numpy()
    round_six_features = X_train.loc[round_six.index].copy()

    assert set(round_six["fixture"]) == {106, 107}
    assert round_six["target_points"].tolist() == [9.0, 10.0]
    assert round_six_features["total_points_lag1"].tolist() == [5.0, 5.0]
    assert round_six_features["total_points_ma5"].tolist() == [3.0, 3.0]
    assert round_six_features["team_goals_for_ma5"].tolist() == [1.8, 1.8]
    assert round_six_features["team_goals_against_ma5"].tolist() == [0.4, 0.4]
    assert round_six_features.loc[:, ["total_points_lag1", "total_points_ma5", "team_goals_for_ma5"]].nunique().eq(1).all()


def test_assign_pre_round_features_uses_prior_fixtures_only_with_missing_stats() -> None:
    frame = pd.DataFrame(
        [
            {
                "player_id": 1,
                "season_name": "2025-26",
                "round": 1,
                "fixture": 11,
                "kickoff_time": "2025-08-01T12:00:00Z",
                "total_points": 1.0,
            },
            {
                "player_id": 1,
                "season_name": "2025-26",
                "round": 2,
                "fixture": 12,
                "kickoff_time": "2025-08-08T12:00:00Z",
                "total_points": None,
            },
            {
                "player_id": 1,
                "season_name": "2025-26",
                "round": 3,
                "fixture": 13,
                "kickoff_time": "2025-08-15T12:00:00Z",
                "total_points": 3.0,
            },
            {
                "player_id": 1,
                "season_name": "2025-26",
                "round": 4,
                "fixture": 14,
                "kickoff_time": "2025-08-22T12:00:00Z",
                "total_points": 4.0,
            },
            {
                "player_id": 1,
                "season_name": "2025-26",
                "round": 4,
                "fixture": 15,
                "kickoff_time": "2025-08-24T12:00:00Z",
                "total_points": 5.0,
            },
        ]
    )

    result = _assign_pre_round_features(
        frame,
        entity_cols=("player_id", "season_name"),
        stats=("total_points",),
        windows=(2, 3),
    )

    round_four = result.loc[result["round"].eq(4)].copy()
    assert round_four["prev_matches"].tolist() == [3.0, 3.0]
    assert round_four["total_points_lag1"].tolist() == [3.0, 3.0]
    assert round_four["total_points_ma2"].tolist() == [3.0, 3.0]
    assert round_four["total_points_ma3"].tolist() == [2.0, 2.0]
