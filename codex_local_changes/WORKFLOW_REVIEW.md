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

## Remaining Notes

- There is no existing test runner configuration in the repo, so the added tests use Python's built-in `unittest`.
- A future improvement would be to add a lightweight CI workflow that runs `python -m unittest discover`.
