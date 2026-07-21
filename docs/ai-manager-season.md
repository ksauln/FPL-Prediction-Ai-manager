# AI Manager Season Runbook

This guide explains how to run the AI manager when a new FPL season starts, how to select the first squad, and how to use the manager each gameweek.

## What The AI Manager Controls

The AI manager chooses:

- initial 15-player squad
- starting XI
- bench order
- captain and vice-captain
- weekly transfers
- when to hold transfers
- chip timing
- transfer hits
- bank, team value, and FPL sale values

You control only the run settings: data refresh, prediction range, simulation count, and simulation mode.

## Before The Season Starts

Wait until FPL has launched the new season and the official API includes:

- player prices
- promoted/relegated teams
- transfers in and out
- new players
- fixtures
- GW1 event metadata

Then run a forced data refresh:

```bash
fpl_venv/bin/python main.py --force-refetch --override-next-gw 1 --override-last-finished-gw 0
```

This pulls fresh `bootstrap-static` and fixture data from the official FPL API. New players and transferred players are included because the pipeline builds its player list from the current bootstrap payload before fetching histories.

## Generate Full-Season Prediction Files

After the forced refresh, generate prediction files for the season:

```bash
fpl_venv/bin/python main.py --replay-start-gw 1 --replay-end-gw 38
```

This creates:

- `outputs/predictions_gw1.csv` through `outputs/predictions_gw38.csv`
- best XI artifacts for each gameweek
- bench files
- residual files where available

The pipeline now only collates `data/raw/player_<id>.json` files for players in the current bootstrap list, so stale player caches from previous seasons are ignored.

## Run The AI Manager Simulations

Open the notebook:

```bash
jupyter notebook manager_test.ipynb
```

Use these settings for a serious season simulation:

```python
START_GW = 1
END_GW = 38
SIMULATIONS = 50000
SIMULATION_MODE = "periodic_reoptimization"
POLICY_REFRESH_INTERVAL = 1000
RANDOM_SEED = 90
SHOW_PROGRESS = True
```

Recommended simulation modes:

- `periodic_reoptimization`: recommended for 50,000 runs. The AI manager re-chooses the full season policy every `POLICY_REFRESH_INTERVAL` simulations.
- `fixed_policy`: fastest. The AI manager chooses one full season policy, then runs many point simulations against it.
- `full_reoptimization`: slowest. The AI manager re-chooses everything inside every simulation. Use only for small batches.

## Select The First Team

After the notebook finishes, review these sections:

- `Compare Simulation Runs`
- `Best Run: Weekly Decisions`
- `Best Run: Transfers`
- `Best Run: Chips`
- `First Gameweek Squad`

For GW1, use the `First Gameweek Squad` section:

- buy the listed 15 players
- set starters where `role = starter`
- set the bench in `bench_order`
- set captain and vice-captain from the GW1 row in `Best Run: Weekly Decisions`

The notebook saves the full artifact here:

```text
outputs/season_plan_simulations.json
```

Keep that file as the reference plan.

## Each Gameweek During The Season

Before each gameweek deadline:

1. Refresh live FPL data and player metadata:

   ```bash
   fpl_venv/bin/python main.py --force-refetch
   ```

2. Generate or refresh upcoming prediction files. For example, if the next gameweek is GW8 and you want a 4-week planning horizon:

   ```bash
   fpl_venv/bin/python main.py --replay-start-gw 8 --replay-end-gw 12
   ```

3. Reopen or rerun `manager_test.ipynb`.

4. Set:

   ```python
   START_GW = 1
   END_GW = 38
   ```

   Keep the full season range if you want to compare against the original plan. If you only want a short planning rerun, set `START_GW` to the next gameweek and `END_GW` to the end of your horizon.

5. Review the row for the upcoming gameweek in `Best Run: Weekly Decisions`.

6. Apply the recommended:

   - transfers
   - captain
   - vice-captain
   - bench order
   - chip, if one is listed

## Important Current Limitation

The current manager creates a plan from the prediction files it loads. It does not yet import your real live FPL team state from your FPL entry ID.

That means:

- if you follow the manager plan exactly, the saved season plan remains the reference
- if you manually deviate, the manager does not yet know your actual squad, bank, purchase prices, or used chips
- the next feature should be an actual-team state importer so weekly recommendations can start from your real current team

Until that importer exists, use the manager as a season planner and decision guide. If you deviate from the plan, note the difference manually before trusting future transfer rows.

## New Player And Transfer Checks

The original pipeline should be rerun whenever the player pool changes or fresh data matters:

```bash
fpl_venv/bin/python main.py --force-refetch
```

This refreshes:

- official player list
- player prices
- player teams
- player statuses
- fixtures
- histories for current bootstrap players

The pipeline ignores stale cached player history files for players no longer present in the current bootstrap list.
