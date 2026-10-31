# V2 transfer-hit discipline

## Scope

Fix v2's paid-transfer decisions without changing the legacy manager, live state,
forecast models or existing backtest artifacts. Snapshot the pre-fix policy for
direct comparison. Use the repository debugging and test-first workflow.

## Evidence

The GW1–8 four-week replay tied at 398 net points, but v2 spent 40 hit points
against 16. V2 requires an advantage over holding, not the best available free
move. The free alternative can also be omitted from the simulation shortlist.
Terminal free-transfer value can justify an expensive move without a corresponding
gain in projected scored points.

## Implementation

1. Add failing tests for marginal paid-move comparisons, protected free options,
   uncertain advantages, clearly worthwhile hits and larger free packages.
2. Preserve a best no-hit root candidate during search and finalist screening.
3. Paid moves must beat the best no-hit continuation in fresh, paired scenarios.
   Net points already deduct hits. Require the usual 1.5-point minimum advantage
   plus a default two-point buffer per paid transfer, and at least 75% of paired
   scenarios beating the free continuation. Terminal transfer value cannot justify
   the hit. Apply the same per-hit buffer to deterministic planning and future
   beam branches; preserve a no-hit continuation at each search step.
4. Explain the chosen free comparator, marginal gain, buffer and probability in
   result diagnostics. Keep chip and free-transfer legality unchanged.
5. Run the full regression suite, the original seed-90 GW1–8 comparison, and
   controlled GW5–8 seed/buffer sensitivity from a common saved GW4 state.

## Evaluation contract

Defaults are fixed from the decision rule above, before observing new replay
results. Compare against the snapshotted v2 and current legacy manager on the
same forecasts and states. Report all tested settings; do not select defaults by
highest realized points. The eight observed gameweeks are not an independent
holdout. Seed/buffer sensitivity and synthetic correctness tests supplement that
limited sample; prospective validation remains necessary. No automatic promotion.

## Completion evidence

The revised primary four-week replay scored 412 against 398 before, with hit
deductions reduced from 40 to 4. Controlled seed gains were +14/+17/+24, and all
three tested buffers improved on the old seed-90 policy. Defaults stayed fixed.
285 tests passed and protected artifact hashes / legacy outputs were preserved.
See `docs/manager-hit-discipline.md` for complete results, including the remaining
one-week control deficit and the absence of an independent holdout.
