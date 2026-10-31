# Preseason Enrichment Runbook

The preseason path adds point-in-time evidence without making live prediction
runs depend on restricted web pages or paid services. The baseline remains the
official FPL API plus the local historical archive. Optional evidence is stored
under `data/snapshots/<season>/<topic>/<provider>/` as an immutable CSV with a
sidecar manifest containing its capture time, source, rights class, schema,
row count, CSV SHA-256 hash, and an integrity hash binding the manifest metadata.

`data/snapshots/` is intentionally gitignored. It can contain local-use-only
data, licensed exports, and manually entered evidence. Each season archive
records the exact snapshot hashes used in `context_gw<N>.json`.

## What Changed

- Training retains the full historical player universe. Current players are
  bridged through stable player codes; historical-only players use a disjoint
  training ID and no longer disappear in an inner join.
- Prediction rolling features are one step forward: `lag1` is the latest
  completed match and `ma5` includes it. Training rows remain shifted, so their
  targets do not leak into their inputs. Both fixtures in a double Gameweek use
  the same pre-deadline feature state.
- Model selection uses expanding season/Gameweek groups, so rows from one event
  cannot be split between training and validation.
- Team form follows stable club codes across seasons. GW1 uses prior-season
  Premier League form for incumbent clubs and Championship priors for promoted
  clubs. During the first five matches, prior weight fades as current results
  accumulate.
- Sparse players can use friendly start/appearance priors. The prior fades to
  zero over ten completed historical matches.
- GW-scoped role-change evidence can nudge conditional scoring by at most 10%.
  Evidence quality and current-season reliability reduce the nudge, which fades
  to zero over ten completed matches.
- Live official `ep_next` receives 35% weight for reliable players and up to
  85% for a zero-history player. Missing official forecasts are never replaced
  with zero.
- Official set-piece orders, preseason evidence, workload, stable team-prior
  source, reliability, and every applied blend weight are retained as audit
  columns.
- Residual evaluation uses models fitted strictly before the completed
  Gameweek, then updates bias state before the next forecast. A Gameweek cannot
  be applied twice or evaluated in-sample.

## Snapshot Topics

| Topic | Required identity | Key fields | Effect |
|---|---|---|---|
| `preseason_player_usage` | `player_code` | team matches, appearances, starts, minutes, last-two minutes, source confidence | Bounded start and appearance priors |
| `preseason_role_evidence` | `player_code`, `role_target_gameweek` | attacking-role, penalty, corner, and direct-free-kick deltas; matches; confidence; observation time | Bounded, reliability-gated conditional-scoring adjustment |
| `tournament_workload` | `player_code` | direct minutes/matches/starts, last appearance, explicit readiness multiplier, confidence, valid-through GW | Workload audit; only an explicit readiness multiplier changes EP |
| `team_priors` | `team_code` | last-five goals, defence, clean sheets, two-plus conceded, match points | GW1 and early-season team/opponent context |
| `provider_expected_points` | `player_code`, `gameweek` | provider EP and optional bounded row weight | Optional 20% ensemble input by default |
| `fixture_odds` | provider event/team identity | timestamped no-vig probabilities | Archived for research; not yet a trained feature |

Unknown facts must stay blank. A missing appearance, minute value, or workload
record is not treated as zero. `readiness_multiplier` must be an explicit,
auditable value between zero and one; the code does not invent a World Cup
fatigue penalty.

## Free Team Priors

Build a private snapshot from the final 2025/26 Premier League and Championship
results:

```bash
fpl_venv/bin/python3 scripts/update_preseason_snapshots.py \
  --snapshot-root data/snapshots team-priors \
  --season 2026-27 --prior-season 2025-26
```

The updater validates Football-Data sentinel columns, maps only explicit club
aliases/stable codes, stores source hashes, and does not retain the raw CSVs.
The site describes a Sunday/Wednesday in-season update cadence and warns that
the data can be inaccurate, so provenance remains visible in every run.
Strict GW1 also verifies the prior-season manifest, the current
`team_code`-to-`team_id` mapping, and a `team_prior_as_of` date no more than 120
days before the deadline.

## Friendly Usage

The official preseason tracker is a useful discovery index, but it is an
editorial page without a lineup/minutes API. Linked club reports vary from full
XIs and substitutions to narrative-only reports. The Premier League terms also
restrict database reuse, so automatic scraping is not a default source.

Create a current-player entry template:

```bash
fpl_venv/bin/python3 scripts/update_preseason_snapshots.py player-template \
  --output data/processed/preseason_player_usage_2026-27.csv
```

Fill only observations supported by first-party reports or a licensed feed.
The final one or two friendlies should carry the most attention; the derived
start prior gives the starts share 45%, total minutes 25%, and last-two
minutes 30%. Evidence quality also requires match depth and a
`source_confidence` value.

For the 2026/27 build, the conservative importer combines a pinned
FPL-Core Insights export with explicit first-party supplements. It requires
exactly two selected evidence matches per club, validates the live
`player_code`/club mapping, and distinguishes full `absence_safe` reports from
partial reports. A partial report can prove that a listed player appeared, but
cannot turn an omitted player into a zero-minute observation:

```bash
fpl_venv/bin/python3 scripts/import_preseason_usage.py \
  --bootstrap data/raw/bootstrap-static.json \
  --core-dir "/path/to/FPL-Core-Insights/data/2026-2027/By Tournament/Friendlies/GW0" \
  --selection data/processed/preseason_usage_match_selection_2026-27.csv \
  --supplements data/processed/preseason_usage_official_supplements_2026-27.csv \
  --output data/processed/preseason_player_usage_2026-27.csv \
  --cutoff 2026-08-21T17:30:00Z
```

The importer deliberately leaves starts blank because the upstream start-time
field is not reliable. Players with no positive observation remain fully blank.

Keep every current-bootstrap player in the final file, including blank evidence
rows. Strict GW1 requires a snapshot no more than seven days old and at least
11 meaningfully evidenced selectable players for every club. Club membership
comes from the live bootstrap; the CSV's `team_code` is only a consistency
check.

Publish the completed local file before the GW1 deadline:

```bash
fpl_venv/bin/python3 scripts/update_preseason_snapshots.py publish-csv \
  --season 2026-27 --topic preseason_player_usage \
  --input data/processed/preseason_player_usage_2026-27.csv \
  --provider manual_club_reports \
  --source-url https://www.premierleague.com/en/news/4606700 \
  --rights-class local_use_only
```

## Role Changes And Set Pieces

Create a GW1 role-evidence template:

```bash
fpl_venv/bin/python3 scripts/update_preseason_snapshots.py role-template \
  --target-gameweek 1 \
  --output data/processed/preseason_role_evidence_2026-27_gw1.csv
```

Enter only changes from the player's established baseline. Each directional
delta is between `-1` and `1`: negative means a less productive role, positive
means a more productive one, blank means unknown, and `0` means confirmed
unchanged. The components are weighted 45% attacking role, 35% penalties, 12%
corners, and 8% direct free kicks. Match depth and
`role_evidence_confidence` determine evidence quality.

```bash
fpl_venv/bin/python3 scripts/update_preseason_snapshots.py publish-csv \
  --season 2026-27 --topic preseason_role_evidence \
  --input data/processed/preseason_role_evidence_2026-27_gw1.csv \
  --provider manual_club_reports \
  --source-url https://www.premierleague.com/en/news/4606700 \
  --rights-class local_use_only
```

Role evidence is selected by `role_target_gameweek`; a newer snapshot for a
different Gameweek does not hide an older matching snapshot. It adjusts
conditional scoring before official/provider EP blends and never changes start
or appearance probability.

## World Cup, Licensed Projections, And Odds

FIFA displays workload statistics but exposes no documented public export/API,
and its terms do not support a default redistributed data feed. Use a licensed
provider export or a small locally verified workload file. Preserve raw minute
strings when their encoding is unclear.

Create an audited workload template. `workload_valid_through_gameweek` prevents
a GW1 readiness judgement from silently persisting all season:

```bash
fpl_venv/bin/python3 scripts/update_preseason_snapshots.py workload-template \
  --target-gameweek 1 \
  --output data/processed/tournament_workload_2026-27_gw1.csv
```

Populate minutes, matches, starts, the last appearance timestamp,
`workload_evidence_confidence`, and an explicit `readiness_multiplier` in
`[0, 1]`, then publish it with `--topic tournament_workload`.

The same `publish-csv` command accepts `tournament_workload`,
`provider_expected_points`, and `fixture_odds`. Provider projection rows must
name their target `gameweek`; a snapshot for another Gameweek is skipped.
Credentials never belong in a source URL, CSV, manifest, log, or repository.

## Final Rerun

`auto` uses every eligible snapshot captured at or before the target deadline.
For the final GW1 build, `strict` proves that player-usage and team-prior
snapshots are cutoff-safe, that usage matches the current player roster, and
that meaningful evidence covers at least 11 selectable players for each of the
exact 20 current clubs:

```bash
fpl_venv/bin/python3 main.py --force-refetch --expected-season 2026-27 \
  --override-next-gw 1 --override-last-finished-gw 0 \
  --snapshot-mode strict
```

Use `--snapshot-mode off` for a baseline comparison. Do not publish the final
friendly-usage snapshot until the last meaningful club matches and late World
Cup returns have been checked, ideally in the final days before the deadline.
Every completed run archives the exact model ensemble, effective configuration,
source modules, dependency file, snapshots, and hashes under
`outputs/seasons/<season>/reproducibility_gw<N>/`.

For planning Gameweeks after the official next Gameweek, transient status flags
are not copied blindly through the season. `u`/`n` remain unavailable; dated
`Expected back D Mon` injuries and `Suspended until D Mon` suspensions apply
only before the player's target fixture date; undated transient flags become
neutral. Prediction artifacts retain the parsed return date, target kickoff,
and whether the future status gate was applied.

## Primary References

- [Official 2026 preseason tracker](https://www.premierleague.com/en/news/4606700)
- [Official FPL preseason scouting guide](https://www.premierleague.com/en/news/4679613)
- [Football-Data England results](https://www.football-data.co.uk/englandm.php)
- [Football-Data column notes](https://www.football-data.co.uk/notes.txt)
- [The Odds API v4 guide](https://the-odds-api.com/liveapi/guides/v4/)
- [Sportmonks fixture and lineup documentation](https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints/fixtures)
