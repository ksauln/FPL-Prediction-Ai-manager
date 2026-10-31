from __future__ import annotations

import pandas as pd

from fplmodel.history_universe import build_history_universe


def _current_players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2],
            "code": [1001, 1002],
            "element_type": [3, 4],
        }
    )


def test_history_universe_retains_mapped_and_historical_only_players() -> None:
    histories = pd.DataFrame(
        {
            "player_id": [77, 88, 99],
            "player_code": [1001, 9001, 9001],
            "season_name": ["2024-25", "2024-25", "2025-26"],
            "element_type": [4, 2, 2],
            "team": [11, 12, 13],
        }
    )

    result = build_history_universe(histories, _current_players())

    assert len(result) == len(histories)
    mapped = result.loc[result["player_code"].eq(1001)].iloc[0]
    historical_only = result.loc[result["player_code"].eq(9001)]

    assert mapped["historical_player_id"] == 77
    assert mapped["player_id"] == 1
    assert mapped["current_player_id"] == 1
    assert bool(mapped["is_current_player_mapped"])
    assert bool(mapped["position_changed"])
    assert mapped["element_type"] == 4
    assert mapped["team"] == 11

    assert historical_only["player_id"].nunique() == 1
    assert historical_only["player_id"].iloc[0] < 0
    assert historical_only["training_player_key"].nunique() == 1
    assert historical_only["current_player_id"].isna().all()
    assert historical_only["current_element_type"].isna().all()
    assert historical_only["position_changed"].isna().all()
    assert not historical_only["is_current_player_mapped"].any()


def test_unmapped_surrogates_do_not_collide_with_current_or_reused_season_ids() -> None:
    histories = pd.DataFrame(
        {
            # Historical element ids are only season-local and may be reused.
            "player_id": [1, 1, 2],
            "player_code": [7001, 7002, 1002],
            "season_name": ["2024-25", "2025-26", "2025-26"],
            "element_type": [2, 3, 4],
            "team": [3, 4, 5],
        }
    )

    result = build_history_universe(histories, _current_players())

    by_code = result.set_index("player_code")
    assert by_code.loc[1002, "player_id"] == 2
    assert by_code.loc[7001, "player_id"] < 0
    assert by_code.loc[7002, "player_id"] < 0
    assert by_code.loc[7001, "player_id"] != by_code.loc[7002, "player_id"]
    assert result.groupby("player_code")["player_id"].nunique().max() == 1
    assert result.groupby("player_id")["player_code"].nunique().max() == 1
