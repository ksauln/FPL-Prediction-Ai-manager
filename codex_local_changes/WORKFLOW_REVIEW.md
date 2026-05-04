# Model Workflow Review

## Scope

- Reviewed the pipeline entrypoint, feature engineering, model selection, residual update, and squad optimization flow.
- Focused changes on correctness risks in the model workflow rather than output artifact churn.

## Changes Made

- Fixed prediction frame assembly so player metadata and engineered features keep the same row alignment.
- Made player rolling features season-aware instead of ordering all seasons by gameweek number alone.
- Preserved full completed historical seasons for training while limiting only the current season to finished gameweeks.
- Reworked team-context rolling features to calculate from unique team fixtures, then merge fixture-level form back to player rows.
- Added regression tests covering the corrected workflow behavior.
- Added prediction confidence fields, including start probability, confidence score/level, and approximate 80% expected-points intervals.
- Parallelized player-history fetching with progress logs.
- Bounded model-selection tuning to the most recent training rows while preserving full-data final fits, so slow candidates do not block the whole workflow indefinitely.
- Disabled RF/MLP candidates by default after random forest tuning repeatedly monopolized the run; the default high-accuracy path now focuses on histogram gradient boosting plus XGBoost when installed.
- Fixed XGBoost 3.x and sklearn cross-validation compatibility so XGBoost can be evaluated instead of silently falling out of the candidate set.

## Remaining Notes

- There is no existing test runner configuration in the repo, so the added tests use Python's built-in `unittest`.
- A future improvement would be to add a lightweight CI workflow that runs `python -m unittest discover`.
