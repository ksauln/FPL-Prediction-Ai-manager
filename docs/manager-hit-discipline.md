# V2 transfer-hit discipline: fix and evaluation

The revised v2 scored **412 net points versus 398 before the fix** in the
GW1–8 reconstructed four-week replay. Hit deductions fell from **40 to 4**.
The legacy manager also scored 398 on those inputs. The original next-week-only
control improved from 388 to 401, but remains below the legacy manager's 446.
At evaluation time the live manager was unchanged and the revision was opt-in.
The owner enabled live v2 on 2026-11-03; see the [current engine guide](manager-v2.md).
The historical performance limitations below remain applicable.

## What was wrong

The previous policy tested paid moves against holding, not the best available
no-hit move. A good free transfer could be omitted from the fresh simulation
shortlist. Estimated free transfers at the horizon boundary could also make an
expensive package appear preferable despite a small gain in scored points.

In the original GW5 replay, the 16-point-hit package was projected only about
3.7 net points ahead of a free alternative. That is a weak margin for committing
four paid transfers on uncertain four-week forecasts.

## New decision rule

The candidate search protects a no-hit alternative, and fresh validation always
includes the generated no-hit plans. Every paid first action must beat the best
available no-hit continuation by projected **net scored points**, excluding
terminal free-transfer estimates from this comparison.

Default requirements are:

- Mean marginal net gain at least `transfer_gain_threshold + 2 × paid transfers`.
  With the standard 1.5-point threshold, one paid transfer needs 3.5 additional
  net points; four need 9.5. The underlying four-point hits are already deducted.
- At least 75% of the shared validation scenarios score more points than the
  no-hit continuation. This is a model scenario frequency, not a guarantee of
  actual performance or an empirically calibrated probability.

The two-point buffer also enters deterministic plan ranking and future beam
branches. A continuation without additional future hits is protected from beam
pruning. This buffer is a decision margin: it never changes actual hit deductions,
bank, prices, free transfers or scored points. Future recommendations are replanned
and checked again at their deadlines.

The deterministic `simulate_season` v2 path applies the expected-gain margin;
probability checks require `run_repeated_season_simulations` or the trial UI.
Clearly worthwhile paid moves and larger free packages remain available. For
example, the revised primary replay still took one four-point hit in GW6: its
fresh scenario mean was 4.73 net points above the free alternative, with an
82% scenario win frequency, clearing the 3.5-point requirement.

The UI exposes **Extra net gain per paid transfer** and **Paid transfer minimum
win chance**. Decision tables include the comparator, marginal net gain, required
margin, hit cost, scenario frequency and rejection reasons.

## Primary replay: GW1–8, seed 90

All scores include actual appearances, legal autosubs, captain fallback, chips
and hits. Before and after use the same forecasts and initial squad. Their squad,
financial and chip states also match through GW4, before the transfer-hit changes
affect realized decisions.

| Inputs | Legacy | Pre-fix v2 | Revised v2 | Revision gain |
| --- | ---: | ---: | ---: | ---: |
| Reconstructed four-week horizons | 398 | 398 | 412 | +14 |
| Original next-week-only control | 446 | 388 | 401 | +13 |

| Four-week replay | Pre-fix v2 | Revised v2 |
| --- | ---: | ---: |
| Net actual points | 398 | 412 |
| Hit deductions | 40 | 4 |
| Actual points before hits | 438 | 416 |
| Regular transfers, excluding Wildcard/Free Hit | 15 | 6 |

The policy gave up 22 gross points while saving 36 hit points, for a net gain of
14. This is a reduction in costly marginal moves, not an increase manufactured
by modifying the points or fee calculation.

## Controlled seed comparison: GW5–8

Both policies start from the **same saved pre-fix state after GW4**. The pre-fix
source was snapshotted and hash-checked against the original backtest. Both use
four-week horizons, beam width four, 128 screening scenarios and 128 fresh
validation scenarios. No actual results are exposed to either policy when it
chooses a move.

| Seed | Before points | Revised points | Gain | Before hit cost | Revised hit cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| 90 | 183 | 197 | +14 | 40 | 4 |
| 91 | 195 | 212 | +17 | 28 | 8 |
| 92 | 182 | 206 | +24 | 40 | 16 |
| **Mean** | **186.7** | **205.0** | **+18.3** | **36.0** | **9.3** |

The revised policy improved net points and reduced hits in all three runs. It
still accepted a larger paid package in seed 92 when that package cleared the
prospective checks; this is deliberately not a blanket hit ban.

## Buffer sensitivity: GW5–8, seed 90

The minimum scenario win frequency remains 75% in all cases.

| Extra net gain required per paid transfer | Points | Hit cost | Gain over pre-fix |
| --- | ---: | ---: | ---: |
| 1 point | 206 | 16 | +23 |
| **2 points: default** | **197** | **4** | **+14** |
| 3 points | 197 | 4 | +14 |

All three settings improved on the old policy here. The one-point setting scored
more in this observed period but spent more on hits. Defaults were fixed from
the decision rule before these runs and were **not changed to the highest
realized backtest score**.

## Limits and preservation

This is sensitivity analysis on eight already observed gameweeks, not a new
independent holdout. Three seeds are three simulated-policy choices against the
same actual season, not three seasons of evidence. Four-week horizons were
reconstructed from each origin's archived models, metadata and past-only history;
they are not retained original forecast vintages. The one-week control does not
exercise the full multi-week planner, but its remaining deficit is a reason to
keep the revised policy in trial mode. Prospective validation and further chip /
forecast work remain necessary before promotion.

No live state, original predictions, completed jobs, fitted models or bias state
were advanced or rewritten. The primary and controlled comparisons each checked
161 protected files. All 66 files protected by the original baseline also retain
their hashes. The legacy default and seeded simulation output still match the
preserved original implementation.

## Tests and files

`fpl_venv/bin/python -m pytest -q`: **285 passed**. The existing joblib Darwin
CPU-discovery warning remains. Diff and compilation checks passed.

Tests cover marginal comparison against a free move, shortlist preservation,
uncertain paid advantages, terminal-value loopholes, correct fee accounting,
worthwhile paid moves, larger free packages, configuration validation, UI
controls and live-state isolation.

Files changed or added:

- `fplmodel/manager_v2.py`: hit-aware search, free alternatives and paired guard.
- `fplmodel/season_manager.py`: additive v2 configuration defaults.
- `fplmodel/manager_dashboard.py`: controls and decision explanations.
- `fplmodel/manager_backtest.py`: optional frozen-policy runner for comparisons.
- `scripts/compare_hit_policies.py`: controlled seed and buffer experiments.
- `tests/test_manager_hit_discipline.py` and `tests/test_manager_v2_ui.py`:
  focused regressions and UI coverage.
- This report, `docs/manager-hit-discipline-plan.md` and `docs/manager-v2.md`:
  design, reproduction and usage.

## Artifacts and reproduction

Frozen pre-fix source:
`outputs/manager_v2/policy_baselines/20261031T123309Z/`.

Primary after-fix replay:
`outputs/manager_v2/season_backtests/20261031T124523194933Z/`.

Controlled experiments, raw decisions, protocol, summary CSV and charts:
`outputs/manager_v2/hit_policy_experiments/20261031T125156673875Z/`.

```bash
fpl_venv/bin/python -m scripts.backtest_current_season \
  --end-gw 8 --horizon 4 --simulations 128 --beam-width 4 --seed 90 \
  --reuse-reconstruction outputs/manager_v2/season_backtests/20261031T102032457631Z

fpl_venv/bin/python -m scripts.compare_hit_policies \
  --baseline outputs/manager_v2/policy_baselines/20261031T123309Z \
  --previous-report outputs/manager_v2/season_backtests/20261031T103345961281Z/reconstructed_four_week.json \
  --reconstruction outputs/manager_v2/season_backtests/20261031T102032457631Z
```

Only complete cases with matching frozen source, initial state and simulation
count were reused from the earlier experiment directory. Revised-policy cases
all ran against the current source hash; interrupted draft runs were not included.
