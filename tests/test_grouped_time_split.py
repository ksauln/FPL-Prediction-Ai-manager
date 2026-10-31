import numpy as np
import pandas as pd
import pytest

from fplmodel import model


def _build_metadata() -> pd.DataFrame:
    rows = []
    kickoff = pd.Timestamp("2024-08-01T12:00:00Z")
    row_id = 0
    group_sizes = {
        ("2024-25", 1): 6,
        ("2024-25", 2): 4,
        ("2024-25", 3): 5,
        ("2025-26", 1): 4,
        ("2025-26", 2): 7,
        ("2025-26", 3): 4,
    }
    for (season, round_no), size in group_sizes.items():
        for i in range(size):
            rows.append(
                {
                    "row_id": row_id,
                    "season_name": season,
                    "round": round_no,
                    "kickoff_time": kickoff + pd.Timedelta(days=row_id),
                    "minutes": 90 if i < max(3, size - 1) else 30,
                    "element_type": (row_id % 4) + 1,
                }
            )
            row_id += 1
    return pd.DataFrame(rows).set_index("row_id")


def test_grouped_time_series_cv_keeps_season_round_keys_intact():
    metadata = _build_metadata()
    X = pd.DataFrame({"feature": np.arange(len(metadata))}, index=metadata.index)
    y = pd.Series(np.arange(len(metadata)), index=metadata.index)

    _, _, metadata_sorted, _, cv, _ = model._prepare_training_temporal_order(X, y, metadata)

    assert isinstance(cv, model.GroupedExpandingWindowSplit)

    previous_validation_max = None
    for train_idx, test_idx in cv.split(X=np.zeros(len(metadata_sorted))):
        train_keys = set(
            metadata_sorted.loc[train_idx, ["season_name", "round"]].itertuples(index=False, name=None)
        )
        test_keys = set(
            metadata_sorted.loc[test_idx, ["season_name", "round"]].itertuples(index=False, name=None)
        )

        assert train_keys.isdisjoint(test_keys)
        assert max(train_idx) < min(test_idx)

        if previous_validation_max is not None:
            assert previous_validation_max < min(test_idx)
        previous_validation_max = max(test_idx)

        for key in test_keys:
            group_rows = metadata_sorted.index[
                (metadata_sorted["season_name"] == key[0]) & (metadata_sorted["round"] == key[1])
            ]
            assert set(group_rows) <= set(test_idx)


def test_train_models_uses_grouped_cv_for_classifier_and_regressor(monkeypatch):
    metadata = _build_metadata()
    X = pd.DataFrame(
        {
            "minutes_ma3": np.linspace(10, 90, len(metadata)),
            "minutes_lag1": np.linspace(5, 85, len(metadata)),
            "total_points_ma3": np.linspace(1, 6, len(metadata)),
        },
        index=metadata.index,
    )
    y = pd.Series(np.linspace(1, 10, len(metadata)), index=metadata.index)

    captured: dict[str, object] = {}

    class StopAfterCapture(Exception):
        pass

    def fake_build_candidates():
        return ["candidate"]

    def fake_evaluate(
        candidates,
        selection_X,
        selection_y_start,
        selection_y_appearance,
        selection_sample_weight,
        cv_strategy,
        reg_selection_X,
        reg_selection_y,
        reg_selection_weight,
        reg_selection_cv,
    ):
        captured["cv_strategy"] = cv_strategy
        captured["selection_X_len"] = len(selection_X)
        captured["reg_selection_cv"] = reg_selection_cv
        captured["reg_selection_X_len"] = len(reg_selection_X)
        raise StopAfterCapture

    monkeypatch.setattr(model, "_build_model_candidates", fake_build_candidates)
    monkeypatch.setattr(model, "_evaluate_model_candidates", fake_evaluate)
    monkeypatch.setattr(model, "MODEL_SELECTION_MAX_SAMPLES", 18)

    with pytest.raises(StopAfterCapture):
        model.train_models(X, y, metadata)

    assert isinstance(captured["cv_strategy"], model.GroupedExpandingWindowSplit)
    assert isinstance(captured["reg_selection_cv"], model.GroupedExpandingWindowSplit)
    list(
        captured["cv_strategy"].split(
            X=np.zeros(int(captured["selection_X_len"]))
        )
    )

    _, _, metadata_sorted, _, _, _ = model._prepare_training_temporal_order(X, y, metadata)
    start_mask = pd.to_numeric(metadata_sorted["minutes"], errors="coerce").fillna(0.0).ge(60.0).to_numpy()
    reg_metadata = metadata_sorted.loc[start_mask].reset_index(drop=True)

    reg_cv = captured["reg_selection_cv"]
    reg_metadata = reg_metadata.iloc[-int(captured["reg_selection_X_len"]):].reset_index(
        drop=True
    )
    assert captured["reg_selection_X_len"] == len(reg_metadata)

    for train_idx, test_idx in reg_cv.split(X=np.zeros(len(reg_metadata))):
        train_keys = set(
            reg_metadata.loc[train_idx, ["season_name", "round"]].itertuples(index=False, name=None)
        )
        test_keys = set(
            reg_metadata.loc[test_idx, ["season_name", "round"]].itertuples(index=False, name=None)
        )
        assert train_keys.isdisjoint(test_keys)
        assert max(train_idx) < min(test_idx)
