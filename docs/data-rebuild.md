# Rebuild data after cloning

Git contains the application, tests, documentation and illustrative UI
screenshots. Downloadable season/player data and generated artifacts are not
versioned: `data/raw/`, `data/processed/`, `data/external/`, `cache/`, `models/`,
`outputs/` and `logs/` stay local. Previously tracked artifacts were removed
from Git's index without deleting the owner's local files. Older commits still
contain their historical versions; this release does not rewrite Git history.

## First-time setup

Run from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data/external
git clone https://github.com/vaastav/Fantasy-Premier-League.git data/external/Fantasy-Premier-League
python main.py
python -m streamlit run streamlit_app.py
```

On Windows, activate `.venv\Scripts\activate` instead. Skip the historical clone
if that directory already contains the archive; update it intentionally with
`git -C data/external/Fantasy-Premier-League pull --no-rebase origin master`.
The pipeline fetches official bootstrap/fixture/player histories, joins available
external history, builds features, fits its enabled models and writes predictions
and squad artifacts. Its existing data/rule/season validation still applies:
incomplete required inputs produce an error rather than fabricated evidence.

The official per-player API supplies the current season, not all past seasons.
The external archive is therefore a separate setup step, especially for preseason
training. Enrichment schemas, source rights and strict GW1 requirements are in
the [preseason runbook](preseason-enrichment.md).

## What can and cannot be recreated

| Artifact | Recovery |
| --- | --- |
| Current public FPL players, fixtures, prices and match history | Fetch again through the normal pipeline |
| Public historical CSV archive | Clone/update the upstream historical repository |
| Current model bundles and forecasts | Fit/regenerate from the available inputs; exact values can differ as data/dependencies change |
| Live squad, purchase prices, free transfers, chips and committed decisions | Restore the matching `outputs/manager_state_<season>.json` backup; not reconstructed automatically |
| Completed manager runs | Restore `outputs/ai_manager_jobs/` if those decisions/results are needed |
| Private or licensed preseason evidence | Restore permitted snapshots under `data/snapshots/`; schemas are published, source data is not |
| Original deadline forecasts and frozen-model replay evidence | Restore matching season archives/vintages and model/source snapshots; today's API cannot recreate the original information set |

## Preserve an in-progress season

Before moving machines or hosting the app, back up the entire local `outputs/`,
`models/` and permitted `data/snapshots/` directories outside Git. Keep the model,
bias state, season forecast context and manager state together. Verify the copied
state before enabling live updates. Choose one active writer for the season.

Do not run the training/refresh commands merely to check out or push a code
update during the season; they intentionally refresh data and write artifacts.
The release checks use unit/AppTest fixtures instead, and verify that existing
runtime files remain unchanged.
