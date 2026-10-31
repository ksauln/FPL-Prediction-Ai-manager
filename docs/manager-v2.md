# Upgraded AI Manager: read-only season trials

The current manager remains the default. V2 can read a copy of your committed
squad, bank, purchase prices, free transfers and chips, and create recommendations
in a separate job. It cannot commit state or become the template used by the
weekly data-refresh continuation. No predictions, fitted models or bias state
are updated by a trial.

The pre-change working source, models, root predictions and live manager state
are preserved in `outputs/manager_baselines/20261031T094438Z/baseline.tar.gz`.
The accompanying `manifest.json` contains SHA-256 checksums. This backup includes
uncommitted work. Existing completed jobs and season archives remain in place.
The backup directory is ignored by Git.

## What changed

| Decision | Current manager | V2 |
| --- | --- | --- |
| Transfers | Optimize summed horizon projections | Search sequences with a different XI and captain pair each week |
| Rolling | Minimum transfer gain threshold | Explicit hold action, future transfer balances and diminishing value for unused transfers |
| Packages | Default limit of two | Configurable limit, default five in the trial UI; includes larger free packages and paid transfers |
| Chips | Gain thresholds | Compare chip/transfer sequences, including preparation, expiry and the squad afterward |
| Captaincy | EP multiplied by additional reliability penalties | Compare captain/vice pairs including nonappearance fallback |
| Strategy comparisons | Independently sampled policy blocks | All candidates face identical futures; finalists face a fresh batch |
| Future outcomes | Independent player noise | Persistent injury spells, correlated team point shocks and rotation; fixture detail where available |
| Historical assessment | Simulation outputs | Separate deadline-safe competition against the legacy manager and a hold baseline, scored on actual outcomes |

The initial squad is also selected from several candidate squads using weekly
lineups, rather than evaluating only one XI on summed player projections.

## Use in the app

1. Open **AI Manager → Run simulations**.
2. Leave **Current season manager** selected for the ongoing season workflow.
3. To compare the upgrade, choose **Upgraded manager · read-only trial**.
4. Choose **Live gameweek** to start from a copy of the committed squad, or
   **Season planning** for a new squad. Start with 128 simulations and a short
   horizon; the search is more expensive than scoring a fixed policy.
5. Choose individual-fixture forecasts if available, otherwise the existing
   pipeline projections can be used to isolate the decision-engine changes.
6. Review the decision alternatives, points gain relative to holding this week,
   probability of beating that continuation, and downside/upside percentiles.

There is no **Save state** action for v2. The API rejects v2 commits as well.
Automatic weekly continuation ignores v2 jobs and retains existing settings.

V2 ignores the old individual chip gain thresholds. It compares chip use against
waiting inside the search horizon. **Transfer gain** remains the minimum net
plan advantage required to replace the hold continuation. **Plans retained per
action** controls search breadth. **Value of a saved transfer** is a configurable
terminal estimate, not a measured claim that a transfer is worth one point.

Paid transfers now face additional checks against the best no-hit continuation.
The default extra margin is two net points per paid transfer, in addition to the
usual transfer-gain threshold, with a 75% shared-scenario win frequency required.
Fees are already deducted before this comparison; the margin is not another
charge. The trial settings expose both controls. See the
[hit-discipline fix and evaluation](manager-hit-discipline.md) for the rule,
before/after results, seed checks and limitations.

Simulation distributions are explicitly labeled with the gameweeks they cover.
The evaluation horizon may extend beyond the requested output weeks, using the
available future forecasts; returned state advances only through requested weeks.
The first recommendation is intended for the next deadline. Replan after each
deadline with fresh information.

## Individual-fixture forecasts

Use the current virtual environment. This command reads cached inputs and fitted
models, scores each match with its own opponent and venue features, and writes a
new forecast vintage. It never runs the live pipeline, fits models, fetches data,
or saves bias state. Refresh the normal pipeline first if inputs are stale.

```bash
fpl_venv/bin/python -m scripts.manager_v2 forecast \
  --start-gw 9 --end-gw 12 \
  --model-context outputs/seasons/2026-27/context_gw9.json
```

Use the upcoming gameweek and its matching archived context on subsequent runs.
The builder checks that the fitted ensemble hash matches that context, that the
training/bias cutoff is not in the future, and that input hashes remain unchanged
during the build. It retains fixture details inside each prediction CSV. Blanks
receive zero points and zero appearance probability. Double-gameweek totals are
the sum of separately scored fixtures.

Output is placed in a unique directory under
`outputs/manager_v2/forecasts/<season>/<timestamp>/`. The `vintage.json` records
generation time, input observation time, model training cutoff, source hashes
and the official origin deadline. Incomplete builds have no final manifest and
are not offered in the app. Existing forecast vintages are never overwritten.

This forecast path uses cached current-season history with the existing fitted
ensemble. It does not recreate external-history or private preseason enrichment
inputs. Its uncertainty ranges remain heuristic; fixture-level modeling and
decision-engine changes should also be evaluated separately.

For a command-line trial using the generated directory:

```bash
fpl_venv/bin/python -m scripts.manager_v2 trial \
  --predictions outputs/manager_v2/forecasts/2026-27/<timestamp> \
  --state outputs/manager_state_2026-27.json \
  --end-gw 10 --horizon 2 --beam-width 1 --simulations 128
```

The report is written to a unique file in `outputs/manager_v2/reports/`.
For a new season, replace the season and gameweek arguments appropriately.

## Historical manager competition

The competition starts all managers from the same state and reveals a new
forecast vintage each deadline. Each manager advances its own squad and finances;
actual outcomes are used only to score the chosen decision afterward.

The comparison requires a `vintage.json` plus its referenced prediction files
for every consecutive evaluation gameweek. Each vintage must:

- Be generated before the official deadline with timezone-aware timestamps.
- Use inputs observed no later than generation time and a model trained strictly
  before the evaluation gameweek.
- Keep future prices, clubs, positions and player IDs at their origin snapshot.
- Match its declared season and target gameweeks.

The manifest is a provenance record, not an independent proof of the honesty of
manually supplied forecasts. Retain the original source hashes and snapshots.
Do not relabel reconstructed or post-deadline forecasts as historical vintages.

```bash
fpl_venv/bin/python -m scripts.manager_v2 audit \
  --vintages outputs/manager_v2/forecasts

fpl_venv/bin/python -m scripts.manager_v2 backtest \
  --vintages path/to/deadline-safe-vintages \
  --actuals path/to/actual-outcomes.json \
  --state path/to/state-before-first-evaluation.json \
  --legacy-request path/to/original-job/request.json \
  --simulations 128
```

`--legacy-request` is optional and reproduces the original strategy settings;
otherwise the baseline uses the current manager defaults. Both managers must use
identical season rules. The hold baseline makes no transfers and uses no chips.

Actual outcomes use this JSON format, with explicit boolean appearance and
points for every forecast player, including DNPs:

```json
{"9": {"123": [true, 6], "456": [false, 0]}}
```

Reports contain actual net points, captain bonus, hit costs, transfers, same-week
chip gains and gains relative to holding. Same-week counterfactuals do not measure
the complete causal value of a wildcard or transfer over the rest of a season.

The existing season archive has no vintage manifests meeting this contract.
The newly generated GW9 vintage can start a prospective comparison after actual
outcomes are available; it cannot establish performance on earlier gameweeks.

A later, deeper archive audit recovered original GW1–8 next-week forecasts using
their preserved timestamps and original run logs. A separate runner also rebuilt
four-week horizons from frozen historical models. See the
[season-to-date backtest](manager-v2-season-backtest.md) for results, limitations,
reproduction commands and preservation checks. The reconstructed horizons are
explicitly distinguished from original forecast vintages.

## Search and scenario limits

V2 is a bounded beam search, not an exhaustive season optimum. Candidate packages
come from legal integer-optimizer solutions for the current week and remaining
horizon. A bounded beam is kept for every first action, including hold, so a
delayed move can survive an unattractive first week. Bank, sale-value rounding,
purchase prices, transfer carry limits/top-ups, chip halves, free-hit restoration,
and the ban on consecutive free hits carry through transitions.

The horizon is the larger of the transfer horizon and chip lookahead. Chip
opportunities beyond that horizon are not fully valued, so use a horizon that
includes relevant blanks/doubles and chip expiry when evaluating chip timing.

Each scenario has persistent player injury spells of 2–4 gameweeks. Team shocks
correlate defensive or attacking points; defensive shocks can also depend on the
opponent's attack. Shared rotation factors correlate appearance outcomes. These
are configurable assumptions, not fitted or calibrated injury probabilities or
an exact football match simulator. Signed point noise permits negative scores.
Evaluate empirical coverage and appearance calibration before promoting v2.

Candidate continuations cannot observe future sampled outcomes. They are plans
under today's information, not an adaptive policy with perfect future knowledge.
Live adaptation happens by replanning with new inputs at each deadline. Candidate
screening and finalist validation use separate seeds; each phase shares outcomes
across candidates. Reported finalist scores are comparative simulation estimates,
not an unbiased guarantee of future points. Sampling is batched to bound memory.

## Validation

Tests cover legacy default/continuation isolation, rejected state commits, input
immutability, weekly XI changes, captain fallback, rolling to fund a later package,
larger free packages, chip timing/expiry, bench-boost preparation, free-hit state
restoration and cross-half restrictions, correlated/persistent scenarios,
reproducibility, fixture totals/blanks, deadline leakage guards and actual scoring.
Streamlit AppTest covers switching between legacy and trial controls. The cached
GW9–12 forecast build and a GW9–10 trial from the real saved GW8 state were also
executed. No realized performance gain is claimed without historical evidence.

Initial v2 verification on 2026-10-31 (before subsequent backtest, hit-discipline
and UI coverage):

- `fpl_venv/bin/python -m pytest -q`: **259 passed** (one existing Darwin CPU
  discovery warning from joblib).
- `git diff --check` and Python compilation checks passed.
- Legacy deterministic and seeded fixed-policy simulation results exactly
  matched the preserved pre-change implementation on the same inputs.
- All **66** protected baseline model, prediction, live-state and job-metadata
  files retained their original SHA-256 hashes.
- A real cached GW9–12 fixture forecast vintage and GW9–10 read-only trial were
  generated under `outputs/manager_v2/`; the new GW9 vintage passed the provenance
  audit. Earlier unversioned archives were not counted as historical evidence.

The later complete suite passes **295 tests**. The original season replay is
recorded in the [pre-fix backtest](manager-v2-season-backtest.md); current results
and the remaining one-week-control deficit are in the
[hit-discipline evaluation](manager-hit-discipline.md). Baselines, archived models,
private enrichment inputs and generated evaluation reports are local artifacts
excluded from Git. Reproduction commands require those matching inputs.

## Implementation files

- `fplmodel/manager_v2.py`: weekly search, captain pairs and policy comparison.
- `fplmodel/manager_scenarios.py`: shared, batched future-outcome scenarios.
- `fplmodel/manager_forecasts.py`: individual-fixture forecasts and provenance.
- `fplmodel/manager_backtest.py`: deadline-safe historical competition.
- `fplmodel/season_manager.py`: additive version configuration and dispatch.
- `fplmodel/manager_jobs.py`: trial guards and continuation isolation.
- `fplmodel/manager_dashboard.py`: opt-in controls and comparison results.
- `scripts/manager_v2.py`: forecast, trial, audit and backtest commands.
- `tests/test_manager_v2*.py`, `tests/test_manager_forecasts.py` and
  `tests/test_manager_backtest.py`: new regression and integration coverage.
- `docs/manager-v2-plan.md`, this guide, `docs/season-manager.md`, `README.md`
  and `.gitignore`: plan, documentation and private-artifact exclusions.
