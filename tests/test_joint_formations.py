import numpy as np
import pytest

from fplmodel.config import FORMATION_OPTIONS
from fplmodel.team_picker import pick_best_xi
from test_manager_v2 import frames, state


@pytest.mark.parametrize("seed", [1, 12, 90])
def test_joint_formulation_matches_best_separate_formation_objective(seed):
    data = frames((2,))[2]
    rng = np.random.default_rng(seed)
    data["expected_points"] = rng.uniform(0, 12, size=len(data))
    data["now_cost_millions"] = rng.integers(40, 70, size=len(data)) / 10.
    regular = pick_best_xi(data, budget_m=90., formations=FORMATION_OPTIONS)
    joint = pick_best_xi(data, budget_m=90., formations=FORMATION_OPTIONS, joint_formations=True)
    assert joint["optimization_score"] == pytest.approx(regular["optimization_score"])
    assert joint["formation"] in FORMATION_OPTIONS
    assert joint["total_cost"] <= 90. + 1e-6
    assert len(joint["squad"]) == 11 and len(joint["bench"]) == 4


def test_joint_formulation_preserves_transfer_and_bank_constraints():
    data = frames((2,))[2]
    data.loc[data.player_id.isin([16, 17, 19]), "expected_points"] = 20.
    options = dict(formations=FORMATION_OPTIONS, current_player_ids=state().squad_player_ids,
                   bank_m=0., sale_value_by_player_id=state().purchase_price_by_player_id,
                   max_transfers=3, free_transfers=2)
    regular = pick_best_xi(data, **options)
    joint = pick_best_xi(data, joint_formations=True, **options)
    assert joint["optimization_score"] == pytest.approx(regular["optimization_score"])
    assert joint["transfers_made"] <= 3
