# AI Manager Season Runbook

This is the operating guide for preseason planning, selecting the GW1 squad, and running the same stateful manager every gameweek.

## What The Manager Controls

The manager chooses the 15-player squad, starting XI, bench order, captain, vice-captain, transfers, hits, and chips. It tracks purchase prices, FPL sale values, bank, free transfers, chip usage, and decision history.

You still apply the recommendation on the FPL website. Neither the GUI nor the notebook changes your official team.

## 1. Wait For The New Game

Do not begin a 2026/27 run until the official FPL game has launched and its API contains 2026/27 deadlines, prices, promoted clubs, transfers, new players, and fixtures.

Update the historical archive once after the prior season is complete:

```bash
git -C data/external/Fantasy-Premier-League pull --no-rebase origin master
```

The repository currently includes all 38 gameweeks of 2025/26. The pipeline checks this coverage during a preseason run and fails if the configured prior season is incomplete.

Run the launch check and forced refresh:

```bash
fpl_venv/bin/python main.py --force-refetch --expected-season 2026-27 \
  --override-next-gw 1 --override-last-finished-gw 0
```

The command fails if the API still serves 2025/26. Bootstrap and fixture caches otherwise refresh automatically every 15 minutes.

The GUI defaults to the season identified by the cached FPL event deadlines, not
the calendar alone. If prediction files were created by the older pipeline and
lack season metadata, **Validate and tag legacy files** appears only when their
season matches the cached bootstrap data. It verifies every player and club,
backs up the original CSVs under `outputs/prediction_backups/`, and then adds the
missing metadata. Never use that recovery action to relabel one season as another.

The official player-history endpoint only contains the current season. A preseason GW1 model therefore needs the configured external historical dataset. Historical rows are mapped to current players with stable player codes, not season-specific element IDs.

## 2. Confirm The Rules

Check the official [FPL Help and Rules](https://fantasy.premierleague.com/help/) and [Game Updates](https://fantasy.premierleague.com/help/new) pages after 2026/27 launches.

`SeasonRules()` currently models the confirmed 2025/26 structure of one Wildcard, Free Hit, Bench Boost, and Triple Captain in each half. The special 2025/26 AFCON transfer top-up is intentionally not a default because it may not return.

For a 2025/26 replay, configure it explicitly:

```python
rules = SeasonRules(free_transfer_topups={16: 5})
config = SeasonManagerConfig(rules=rules)
```

Update `chips_by_half`, `first_half_end_gw`, transfer limits, or top-ups in the notebook if the 2026/27 rules differ.

## 3. Build Predictions

For the initial full-season planning pass:

```bash
fpl_venv/bin/python main.py --expected-season 2026-27 \
  --replay-start-gw 1 --replay-end-gw 38
```

This writes `outputs/predictions_gw1.csv` through `outputs/predictions_gw38.csv`. Each file is tagged with its season and gameweek. The notebook rejects old or untagged artifacts.

The pipeline also copies each prediction into
`outputs/seasons/<season>/predictions_gw<N>.csv` and stores that season's
matching `bootstrap_gw<N>.json` snapshot alongside it. A latest
`bootstrap-static.json` copy supports catalog discovery. These archives drive
the GUI's **Season to simulate** selector. A season appears only after
prediction files have been generated for it; historical results in the external
dataset alone are not enough to reconstruct a manager simulation.

The future-gameweek files are planning forecasts using the information available now and the target fixtures. They are not claims that future form, injuries, starts, or price changes are already known. Regenerate them each week.

## 4. Select The GW1 Team

Launch the application:

```bash
fpl_venv/bin/streamlit run streamlit_app.py
```

Open **AI Manager** and use **Run simulations** with:

- **Season to simulate** set to `2026-27`
- **Season planning**
- GW1 through GW38
- 50,000 simulations
- **Periodic reoptimization**
- reoptimize every 1,000 simulations

Press **Run AI Manager**. The run continues in a background process and the page polls its progress, elapsed time, policy block, and ETA. Completed and failed runs remain available under **Review results** after Streamlit restarts.

Changing **Season to simulate** switches the prediction files, bootstrap player
list, saved runs, and manager-state view together. Existing season archives are
kept when the live root prediction files are replaced by a newer season.

In **Review results**, select the completed run and use the **Recommended season policy**. Do not select the single highest-scoring sampled outcome. Enter the GW1 squad on the FPL website, including the starting XI, bench order, captain, and vice-captain. Then confirm the action and press **Save state through GW1**.

The confirmation writes `outputs/manager_state_2026-27.json`. It does not submit changes to the FPL website.

### Notebook Alternative

Start Jupyter with the project environment and open `manager_test.ipynb`:

```bash
fpl_venv/bin/jupyter notebook manager_test.ipynb
```

Use:

```python
EXPECTED_SEASON = "2026-27"
RUN_MODE = "planning"
PLANNING_START_GW = 1
PLANNING_END_GW = 38
SIMULATIONS = 50000
SIMULATION_MODE = "periodic_reoptimization"
POLICY_REFRESH_INTERVAL = 1000
```

Run all cells. Use the **Recommended Policy**, not the single luckiest Monte Carlo outcome. The policy comparison ranks each reoptimized manager plan by its average score across its simulation block.

Enter the 15 players from **First Gameweek Squad**, then apply its starter roles, bench order, captain, and vice-captain. The notebook saves `outputs/season_plan_simulations.json`.

## 5. Run Each Live Gameweek

Assume the next deadline is GW8.

For the normal dashboard workflow, launch Streamlit and click **Update data & predictions** under the sidebar navigation. The button force-refreshes live FPL data, rebuilds GW8, and generates the remaining four-gameweek planning horizon through GW11. It does not change the committed live-manager state. Use the commands below when you need a custom horizon or want to run the refresh outside Streamlit.

1. Refresh current players, prices, status, history, and fixtures:

   ```bash
   fpl_venv/bin/python main.py --force-refetch --expected-season 2026-27
   ```

2. Rebuild a four-gameweek planning horizon:

   ```bash
   fpl_venv/bin/python main.py --expected-season 2026-27 \
     --replay-start-gw 8 --replay-end-gw 11
   ```

3. Launch Streamlit, open **AI Manager**, choose **Live gameweek**, and run the displayed next gameweek through the available horizon. The live state fixes the start gameweek automatically.

4. Under **Review results**, apply only the first decision to the official team: transfers, lineup, bench, captain, vice-captain, and chip.

5. After those moves are actually applied, select the confirmation checkbox and press **Save state through GW8**. This advances the live state file.

Each GUI job keeps a reviewable `recommended_state.json` under `outputs/ai_manager_jobs/<job-id>/` before changing the live state file. Do not commit state for recommendations you did not apply. The notebook workflow remains available by setting `RUN_MODE = "live"`; it writes `recommended_state_after_gw<N>.json` and uses `COMMIT_RECOMMENDED_STATE` for the equivalent confirmation.

## New Players And Transfers

New signings and transferred players are discovered from the live `bootstrap-static` player list. The 15-minute metadata cache and `--force-refetch` both ensure that current player IDs, clubs, prices, and statuses are refreshed. Removed players' old cache files are ignored.

Players with no current-season minutes remain eligible. They receive low confidence until they accumulate history rather than being forced to zero expected points.

## Simulation Modes

- `periodic_reoptimization`: recommended for 50,000 simulations. It creates a new manager policy every `POLICY_REFRESH_INTERVAL` runs, evaluates each policy over its block, and returns the best average policy.
- `fixed_policy`: fastest. It creates one manager plan and samples outcomes many times.
- `full_reoptimization`: solves a new manager plan in every simulation and is intended for small batches.

The Monte Carlo scorer applies sampled appearances, legal autosubs, goalkeeper substitution, vice-captain fallback, transfer hits, Bench Boost, and Triple Captain.

## Background Job Files

Every GUI run has its own directory under `outputs/ai_manager_jobs/` containing:

- `request.json`: validated simulation and rule settings
- `status.json`: phase, progress, elapsed time, ETA, and worker status
- `worker.log`: worker standard output and errors
- `result.json`: completed simulation result
- `recommended_state.json`: first recommended gameweek state awaiting confirmation

Canceling a job stops its worker and keeps its status files for review. Canceled and failed jobs are not resumable; start a new run with the same settings.
