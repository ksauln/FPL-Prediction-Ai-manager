from fplmodel.manager_backtest import score_actual_decision


def decision():
    positions = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2,
                 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 4, 14: 4, 15: 4}
    starters = [dict(player_id=p, element_type=positions[p]) for p in (1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15)]
    bench = [dict(player_id=p, element_type=positions[p], bench_order=i) for i, p in enumerate((6, 7, 12, 2), 1)]
    return dict(team=dict(squad=starters, bench=bench), captain_id=8, vice_captain_id=9,
                chip=None, transfer_hit_cost=0.)


def test_partial_autosub_counts_unreplaced_dnp_slots_in_formation():
    outcomes = {p: (True, 4.) for p in range(1, 16)}
    for p in (3, 4, 6, 12, 2):
        outcomes[p] = (False, 0.)
    outcomes[7] = (True, 6.)
    # One defender can replace a missing defender, even if another stays DNP.
    assert score_actual_decision(decision(), outcomes) == 46.


def test_missing_both_goalkeepers_does_not_block_outfield_substitution():
    outcomes = {p: (True, 4.) for p in range(1, 16)}
    for p in (1, 2, 3, 7, 12):
        outcomes[p] = (False, 0.)
    outcomes[6] = (True, 6.)
    assert score_actual_decision(decision(), outcomes) == 46.


def test_cameo_prevents_autosub_and_vice_captain_handles_dnp():
    outcomes = {p: (True, 4.) for p in range(1, 16)}
    outcomes[8] = (False, 0.)
    outcomes[3] = (True, 0.)
    assert score_actual_decision(decision(), outcomes) == 44.
