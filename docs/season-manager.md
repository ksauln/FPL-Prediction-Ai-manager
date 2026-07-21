# Stateful FPL Season Manager

## Design Principle

A real FPL manager needs state, transfer costs, chip timing, bench value, future fixtures, and uncertainty. This is a stateful manager, not another one-week optimizer.

## What It Does

The season manager in `fplmodel/season_manager.py` reuses the existing prediction and squad-optimization pipeline, then adds the missing season layer:

- picks an initial 15-player squad from GW1 predictions and a configurable opening horizon
- carries manager state across gameweeks: squad, bank, free transfers, used chips, captain, vice-captain, and decision history
- tracks purchase price, current team value, sale value, and bank
- evaluates transfers against projected future gain and hit costs
- solves transfers directly under sale-value, bank, club, position, and hit constraints, so multi-player packages are considered together
- only transfers when the net expected gain clears a configurable threshold
- chooses starters, bench order, captain, and vice-captain every gameweek
- evaluates chip opportunities for Wildcard, Free Hit, Bench Boost, and Triple Captain
- compares Bench Boost and Triple Captain value with later opportunities in the same chip half
- applies sampled appearances, legal autosubs, goalkeeper substitution, and vice-captain fallback
- runs repeated Monte Carlo-style simulations by sampling prediction uncertainty from each player's 80% expected-points interval
- persists season-tagged live manager state for the next gameweek

## Core Entry Points

- `simulate_season(predictions_by_gw, gameweeks=None, config=None, initial_state=None)`
  Runs one deterministic stateful simulation over the provided gameweeks.

- `run_repeated_season_simulations(predictions_by_gw, gameweeks=None, simulations=50, config=None, random_seed=None)`
  Runs many stateful simulations using sampled expected points and returns aggregate summary statistics. Periodic mode returns the policy with the best average block score as `recommended_policy`; the luckiest sampled outcome is kept separately as `best_outcome_run`.

  Use `simulation_mode="periodic_reoptimization"` for large batches such as 50,000 simulations. In that mode, the AI manager re-chooses its own squad, transfers, captains, vice-captains, and chips every `policy_refresh_interval` simulations.

  Use `simulation_mode="fixed_policy"` for the fastest large-batch run. In that mode, the AI manager chooses its own squad, transfers, captains, vice-captains, and chips once from the expected projections, then runs many stochastic point simulations against that AI-chosen plan.

  Use `simulation_mode="full_reoptimization"` only for small batches when you want the AI manager to re-decide the whole season separately inside every simulation. This is much slower because it repeatedly solves the season optimization problem.

- `load_prediction_files(output_dir=OUTPUTS_DIR, start_gw=None, end_gw=None, expected_season_name=None)`
  Loads prediction files and can reject stale, mixed-season, untagged, or incorrectly named artifacts.

- `save_manager_state(state, output_path)` / `load_manager_state(path)`
  Persists the exact squad, bank, purchase prices, transfers, chips, and history needed for the next live run.

- `save_season_simulation_artifact(result, output_path)`
  Saves a simulation result as JSON.

- `manager_jobs.create_manager_job(...)` / `start_manager_job(...)`
  Validates a GUI request, persists it, and launches the simulator in a separate process. The worker writes atomic status updates and a final result under `outputs/ai_manager_jobs/`.

- `manager_jobs.commit_recommended_state(...)`
  Advances live state to the recommended policy's first decision only after explicit user confirmation.

## Example Usage

```python
from pathlib import Path

from fplmodel.season_manager import (
    SeasonManagerConfig,
    load_prediction_files,
    run_repeated_season_simulations,
    save_season_simulation_artifact,
)

predictions = load_prediction_files(
    start_gw=1,
    end_gw=38,
    expected_season_name="2026-27",
)
config = SeasonManagerConfig(
    initial_horizon=4,
    transfer_horizon=4,
    chip_lookahead=4,
    transfer_gain_threshold=1.5,
    max_transfers_per_gw=2,
)

result = run_repeated_season_simulations(
    predictions,
    simulations=100,
    config=config,
    random_seed=42,
    simulation_mode="periodic_reoptimization",
    policy_refresh_interval=1000,
)

save_season_simulation_artifact(result, Path("outputs/season_plan_simulations.json"))
```

## Notes For Next Season

The engine is intentionally rule-configurable. FPL chip and transfer rules can change by season, so verify `SeasonRules` against the official rules before using it for a live 2026/27 team. The special 2025/26 AFCON top-up is represented by `free_transfer_topups={16: 5}` when replaying that season; it is not assumed for later seasons.

The point model keeps rotation concepts separate: one classifier estimates 60-minute probability, another estimates any appearance, and the regressor estimates points conditional on reaching 60 minutes. Historical position-level cameo points fill the gap between those probabilities. Monte Carlo autosubs use only any-appearance probability.

Transfer accounting follows the FPL sale-value rule: if a player rises in price while you own him, you receive 0.1m of sale value for every 0.2m price rise. If the player falls below your purchase price, the lower current price is used as the sale value.

The first version is a simulator. That is the right base because live weekly recommendations are just the same state engine with real current squad state and the latest prediction files.
