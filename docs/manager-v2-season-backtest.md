# 2026/27 AI Manager backtest: GW1–8

**Historical pre-fix report.** This page records the original comparison before
the transfer-hit correction. The revised v2 scores 412 on reconstructed four-week
inputs and 401 on the original next-week-only control. See the
[hit-discipline fix and evaluation](manager-hit-discipline.md) for the current
policy, paired seed checks and limitations. The original results below are
retained so the before/after comparison stays auditable.

The upgraded manager did not outperform the current manager in this run. On
reconstructed four-week forecasts they tied at **398 actual points after hits**.
On the original next-week-only forecast control, v2 lost by **58 points**.
The in-progress live state, fitted models and original forecast archives were
preserved.

## Results

| Inputs | Current manager | Upgraded manager | Hold baseline | V2 minus current |
| --- | ---: | ---: | ---: | ---: |
| Original archived next-week forecasts | 446 | 388 | 323 | -58 |
| Reconstructed four-week lookahead | 398 | 398 | 323 | 0 |

Both managers began with the same GW1 squad, prices, bank, zero initial free
transfers and chip inventory. Initial selection used the retained GW1 forecast
and the existing optimizer, without reading actual outcomes. No GW1 transfers
or hits were allowed. Managers subsequently maintained independent state.

The legacy parameters match the last committed live manager request: four-week
horizons, up to two regular transfers, a 1.5-point minimum transfer gain, default
chip thresholds, £100m initial budget, four-point hits and up to five banked free
transfers. V2 used its four-week planner, up to five regular transfers, beam width
four, 128 shared screening scenarios, 128 fresh finalist scenarios, and seed 90
(offset per gameweek). Parameters were not tuned against realized points.

## The main finding: transfer hits consumed the gross gain

| Reconstructed four-week test | Current | Upgraded |
| --- | ---: | ---: |
| Actual points before transfer hits | 414 | 438 |
| Hit deductions | 16 | 40 |
| Net actual points | 398 | 398 |
| Regular transfers, excluding Wildcard/Free Hit | 9 | 15 |
| Captain bonus, including scoring-chip multipliers | 33 | 30 |

V2 earned 24 more points before hits and spent 24 more points on hits. That
arithmetic explains the tie; it does not establish the causal value of any
individual transfer. Its five-transfer packages in GW5 and GW8 are useful
starting points for a later review of hit discipline and forecast uncertainty.

The current manager used Bench Boost in GW1, Wildcard in GW2, Triple Captain in
GW3 and Free Hit in GW4. V2 used Bench Boost in GW1, Free Hit in GW2, Wildcard in
GW3 and Triple Captain in GW4. Both exhausted the first-half chips by GW4. A
four-week planning window does not fully price opportunities later in that half.

## Weekly actual points: reconstructed lookahead

| GW | Current | Upgraded | Hold | V2 minus current |
| --- | ---: | ---: | ---: | ---: |
| 1 | 56 | 56 | 55 | 0 |
| 2 | 72 | 68 | 79 | -4 |
| 3 | 51 | 34 | 31 | -17 |
| 4 | 58 | 57 | 51 | -1 |
| 5 | 51 | 65 | 45 | +14 |
| 6 | 51 | 53 | 26 | +2 |
| 7 | 36 | 42 | 19 | +6 |
| 8 | 23 | 23 | 17 | 0 |
| **Total** | **398** | **398** | **323** | **0** |

## Forecast provenance and limitations

The earlier audit looked only for `vintage.json` manifests. A deeper inspection
found retained origin forecasts for every completed gameweek, generated before
their deadlines, with timestamps corroborated by the original pipeline run logs.
Their archived model, bias, feature-source and model-source files passed hash
checks. Both the feature and point-prediction implementations matched all eight
archived source snapshots. All forecast players had explicit actual match rows;
missing rows were not treated as DNPs.

The original full multi-week forecast vintages were not retained. For the main
test, each origin's next-week CSV was kept unchanged. Future fixture forecasts
were reconstructed using that origin's archived fitted ensemble, bootstrap,
fixture schedule, bias state, preseason evidence and completed prior-season
history. Current-season match history was restricted to gameweeks strictly before
the origin and kickoffs before its deadline. Player form was frozen across that
origin's lookahead, while opponent, venue and fixture difficulty were changed
per match. Future prices, clubs, positions and player universes were frozen to the
origin snapshot. Reconstructed appearance probabilities included availability
once, consistently for both managers.

These are **reconstructed historical horizons**, not forecasts actually retained
before their deadlines. Past statistics can incorporate later corrections, and
the reconstruction assumes form/priors between deadlines. The reconstructed
report explicitly sets `as_of_reconstruction=true` and `deadline_safe=false`.
Original next-week forecasts use timestamp/log evidence rather than original
vintage manifests. Eight gameweeks, one initial squad and one random seed do not
establish statistical significance or full-season performance.

The legacy score also changed from 446 with one-week inputs to 398 with
reconstructed lookahead. Horizon assumptions and long-range forecast quality
need separate validation; comparisons between those two input sets should not
be attributed solely to the decision engine.

## Issues found and verification

- A real club transfer left a v2-owned squad with four players from one club.
  V2 now permits holding that squad and still requires new transfer packages to
  restore the club limit. The legacy path was preserved. This follows the
  [official FPL help](https://fantasy.premierleague.com/help/faqs).
- Actual scoring is independently evaluated with legal partial autosubs,
  goalkeeper substitution, captain fallback, chips and hits. DNP starters that
  remain unsubstituted still occupy their formation slots. This corrected one
  hold-baseline point in GW7; it did not change the one-week manager totals.
  This actual-results evaluator does not change the live simulation scorer.
- To finish the full replay practically, v2 now selects among the same allowed
  formations in one integer solve and caches transfer packages shared by chip
  alternatives. Tests confirm equivalent optimum scores and legal transfer/bank
  constraints. Alternate tied optima are possible. The legacy solver retains its
  original default path.
- `fpl_venv/bin/python -m pytest -q`: **271 passed**, with the existing joblib
  Darwin CPU-discovery warning. Diff/compilation checks passed.
- Legacy seeded simulation results still exactly match the preserved pre-change
  implementation.
- The completed run verified **161 original model/state/archive files unchanged**;
  the earlier baseline's **66 protected files** also retained their hashes.

## Reproduce and inspect

The commands below now execute the revised implementation. Exact pre-fix
reproduction requires the locally retained frozen policy source and input
archives. Generated inputs, reports and baseline snapshots are ignored by Git
and are not supplied by a fresh clone.

```bash
fpl_venv/bin/python -m scripts.backtest_current_season \
  --end-gw 8 --horizon 4 --simulations 128 --beam-width 4 --seed 90
```

To reuse the already reconstructed forecast inputs:

```bash
fpl_venv/bin/python -m scripts.backtest_current_season \
  --end-gw 8 --horizon 4 --simulations 128 --beam-width 4 --seed 90 \
  --reuse-reconstruction outputs/manager_v2/season_backtests/20261031T102032457631Z
```

The completed comparison is in
`outputs/manager_v2/season_backtests/20261031T103345961281Z/`:

- `report.html`: interactive charts, results and weekly breakdown.
- `comparison.png` and `weekly_comparison.csv`: exportable chart and data.
- `reconstructed_four_week.json`: full decisions, configuration, engine hashes
  and reconstructed-input results.
- `archived_one_week.json`: original next-week forecast control.
- `source_audit.json` and `preservation.json`: provenance and preservation checks.

The frozen-model forecast reconstruction and origin feature files are retained
in the earlier `20261031T102032457631Z/` directory. Interrupted draft runs are
separate scratch artifacts; only the completed directory above is the final
comparison.

Code added or updated for this backtest: `scripts/backtest_current_season.py`,
`fplmodel/manager_backtest.py`, `fplmodel/manager_v2.py`,
`fplmodel/team_picker.py`, and regression tests for reconstruction provenance,
owned-squad club changes, actual autosubs, fixture retargeting and joint formations.

Keep the current manager live. V2 has not demonstrated a net points advantage in
this season-to-date test; hit discipline and the value of saving chips beyond the
short planning horizon are the next hypotheses to test prospectively.
