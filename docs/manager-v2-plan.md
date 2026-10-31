# AI Manager v2 implementation plan

## Goal and constraints

Improve season decisions while preserving the in-progress manager. Existing
requests and automatic continuations must keep the legacy engine. V2 runs are
explicit, read-only trials: they may read a copy of live state, but cannot commit
or become the template for automatic live continuation.

## Baseline

The working source, tests, models, root prediction files and live manager state
were preserved in `outputs/manager_baselines/20261031T094438Z/` with SHA-256
checksums. This captures uncommitted work as well as tracked files. Existing
season archives and completed jobs remain in place. Baseline manager tests:
`50 passed`.

## Implementation sequence

1. Add regression tests for version isolation, captain/vice selection, rolling,
   weekly lineups, larger packages, chip preparation and expiry.
2. Implement a bounded, rolling-horizon beam search. Generate legal transfer
   packages with the existing integer optimizer; carry bank, sale values,
   transfers and chips through every week. Include hold and preserve a hold
   continuation. Value remaining transfers explicitly at the horizon boundary.
3. Compare candidate plans on shared scenarios, then validate finalists on an
   independent batch. Add persistent injury, correlated team and rotation
   uncertainty, with reproducible seeds and per-fixture forecast support.
4. Add an offline historical competition that requires deadline-safe forecast
   vintages, uses the same initial state and actual outcomes for both managers,
   and reports net points, captain/chip contributions and transfer value.
5. Add opt-in trial controls, job guards, documentation and standalone commands.
6. Run targeted and full regression checks, smoke-test archived inputs, and
   verify live state/model/prediction hashes against the baseline.

## Limits to report

Search is bounded, not an exhaustive season optimum. Scenario assumptions must
be calibrated against real outcomes. Historical improvement can only be claimed
where genuinely pre-deadline forecasts exist; reconstructed future-aware
forecasts are rejected. No automatic promotion during this season.

## Completed verification

Implementation and regression checks are complete: 259 tests passed, a real
GW9–12 fixture vintage and GW9–10 trial were generated, and all 66 protected
baseline artifact hashes remained unchanged. Legacy deterministic and seeded
fixed-policy results exactly matched the archived implementation. Historical
performance improvement remains unmeasured because earlier deadline-safe
vintage manifests are absent. See `docs/manager-v2.md` for usage and limits.
