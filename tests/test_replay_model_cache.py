from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd

from fplmodel import model as model_module
from main import replay_gameweeks, train_prediction_models_for_replay


def _training_inputs() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    features = pd.DataFrame(
        {
            "minutes_lag1": [0.0, 90.0],
            "fixture_difficulty": [2.0, 4.0],
        }
    )
    targets = pd.Series([1.0, 6.0], name="total_points")
    metadata = pd.DataFrame(
        {
            "season_name": ["2025-26", "2025-26"],
            "round": [1, 2],
            "minutes": [0, 90],
            "element_type": [3, 3],
        }
    )
    return features, targets, metadata


def test_replay_reuses_models_only_for_identical_training_inputs() -> None:
    features, targets, metadata = _training_inputs()
    fitted = (object(), object(), object(), {3: 1.0}, [object()])
    train_fn = Mock(return_value=fitted)
    cache: dict[str, object] = {}

    (
        first,
        first_reused,
        first_fingerprint,
        first_source_gw,
    ) = train_prediction_models_for_replay(
        features,
        targets,
        metadata,
        model_cache=cache,
        train_fn=train_fn,
        gameweek=1,
    )
    (
        second,
        second_reused,
        second_fingerprint,
        second_source_gw,
    ) = train_prediction_models_for_replay(
        features.copy(),
        targets.copy(),
        metadata.copy(),
        model_cache=cache,
        train_fn=train_fn,
        gameweek=2,
    )

    assert first == fitted
    assert second == fitted
    assert first_reused is False
    assert second_reused is True
    assert first_fingerprint == second_fingerprint
    assert first_source_gw == 1
    assert second_source_gw == 1
    assert train_fn.call_count == 1

    changed_metadata = metadata.copy()
    changed_metadata.loc[1, "round"] = 3
    (
        third,
        third_reused,
        third_fingerprint,
        third_source_gw,
    ) = train_prediction_models_for_replay(
        features,
        targets,
        changed_metadata,
        model_cache=cache,
        train_fn=train_fn,
        gameweek=3,
    )

    assert third == fitted
    assert third_reused is False
    assert third_fingerprint != first_fingerprint
    assert third_source_gw == 3
    assert train_fn.call_count == 2


def test_replay_shares_one_process_local_model_cache() -> None:
    seen_caches: list[dict[str, object]] = []

    def fake_run_pipeline(**kwargs):
        seen_caches.append(kwargs["model_cache"])
        return {"gameweek": kwargs["override_next_gw"]}

    with patch("main.run_pipeline", side_effect=fake_run_pipeline):
        results = replay_gameweeks(start_gw=1, end_gw=3)

    assert results == [{"gameweek": 1}, {"gameweek": 2}, {"gameweek": 3}]
    assert len(seen_caches) == 3
    assert seen_caches[0] is seen_caches[1] is seen_caches[2]


def test_replay_retrains_when_effective_training_config_changes() -> None:
    features, targets, metadata = _training_inputs()
    fitted = (object(), object(), object(), {3: 1.0}, [object()])
    train_fn = Mock(return_value=fitted)
    cache: dict[str, object] = {}

    _, first_reused, first_fingerprint, _ = train_prediction_models_for_replay(
        features,
        targets,
        metadata,
        model_cache=cache,
        train_fn=train_fn,
        gameweek=1,
    )
    with patch.object(
        model_module,
        "ENABLE_HYPERPARAM_TUNING",
        not model_module.ENABLE_HYPERPARAM_TUNING,
    ):
        _, second_reused, second_fingerprint, _ = train_prediction_models_for_replay(
            features,
            targets,
            metadata,
            model_cache=cache,
            train_fn=train_fn,
            gameweek=2,
        )

    assert first_reused is False
    assert second_reused is False
    assert first_fingerprint != second_fingerprint
    assert train_fn.call_count == 2
