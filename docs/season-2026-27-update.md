# FPL 2026/27 Model Update

Audit completed against the live FPL game and official Premier League
announcements on 23 July 2026.

## Confirmed Rules

- The budget remains £100.0m, squads contain 15 players, and no more than three
  players can come from one club.
- Managers receive one free transfer per Gameweek and can hold five in total.
  The 2025/26 AFCON free-transfer top-up does **not** return.
- Wildcard, Free Hit, Bench Boost, and Triple Captain are each available once
  in Gameweeks 1-19 and once in Gameweeks 20-38. Free Hit and Wildcard are not
  needed/available for the initial unlimited-transfer GW1 squad build.
- Defensive contribution points remain: two points at 10 CBIT actions for
  defenders and at 12 CBIRT actions for midfielders and forwards.
- Final scores now lock at 09:00 UK time on the day after the last match of the
  Gameweek. The pipeline requires both `finished` and `data_checked` before it
  trains on a Gameweek.

The pipeline reads `game_config`, `element_types`, and `chips` from the live
bootstrap payload. It stops before histories or models are processed if the
budget, squad shape, transfer limits, chip windows, or published scoring values
do not match the supported configuration.

## Bonus Points System Changes

For 2026/27:

- the `-1 BPS` penalty for being tackled has been removed;
- clearances, blocks, and interceptions now earn one BPS per three actions
  rather than per two;
- a goalkeeper earns two BPS for any save, one additional BPS for a save
  inside the box, and one additional BPS for a big-chance save;
- a penalty save is worth seven BPS before the new big-chance-save addition.

Historical FPL files do not expose times tackled, save location, or big-chance
save events, so the pipeline does not manufacture a false historical BPS
recalculation. It already ingests BPS, bonus, CBI, recoveries, tackles, saves,
and defensive contributions. Those rolling inputs will adapt as 2026/27 match
data arrives, while live `ep_next` supplies a bounded current-rules ensemble
input before GW1.

## Position Changes

Historical rows retain the position under which their points were actually
scored. The prediction row uses the player's current live position. Position
flags are now model features, so a reclassified player's old targets are not
silently relabelled as if they had been scored under the new position.

| Player | 2025/26 | 2026/27 |
|---|---:|---:|
| Myles Lewis-Skelly | DEF | MID |
| Lamare Bogarde | DEF | MID |
| Junior Kroupi | FWD | MID |
| Keane Lewis-Potter | DEF | MID |
| Mats Wieffer | MID | DEF |
| Georginio Rutter | MID | FWD |
| Rio Cardines | MID | DEF |
| Ryan Sessegnon | MID | DEF |
| Omar Marmoush | MID | FWD |
| Patrick Dorgu | DEF | MID |
| Eric Moreira | MID | DEF |

Eric Moreira was announced as a change but was not present in the selectable
live player pool at audit time. The pipeline maps by stable player code and
will include him automatically if he enters the live pool.

## New Official Data Used

- **Official expected points:** live `ep_next` contributes 35% of the final
  ensemble only for the official next Gameweek. A future planning replay does
  not reuse it for the wrong Gameweek. Historical `xP` was audited but rejected
  as a training input because the archive contains post-event/capture-time
  snapshots rather than consistently calibrated pre-match forecasts.
- **Fixture Difficulty Rating:** historical FDR is joined from each season's
  `fixtures.csv`; live home/away FDR comes from the official fixtures endpoint.
- **Price Change Predictor:** `price_change_percent` is retained in prediction,
  squad, and transfer records and displayed by the AI Manager. It is a team
  value/timing signal, not an FPL points component, so it does not overwrite
  expected points or force a transfer.
- **Current availability and set pieces:** the existing bootstrap ingestion
  continues to refresh availability, status, set-piece order, transfers,
  ownership, expected metrics, and defensive-contribution data.

The price signal is expected to remain zero until after the GW1 deadline while
prices are locked. FPL describes it as guidance rather than a guaranteed price
change.

The 35% live-EP blend is configurable as `OFFICIAL_EP_BLEND_WEIGHT`. The custom
model remains the majority input, while the official current-season forecast
anchors a cold start under the new positions and BPS rules.

## Sources Reviewed But Not Added As Predictors

- Real-time rank and projected-bonus UI updates improve the matchday
  experience, but do not add a pre-deadline predictive signal.
- Managerial and European-competition changes were reviewed, but the official
  article is editorial rather than a stable structured feed. Their effects will
  enter through minutes, starts, team form, fixtures, and expected statistics.
- Exact 2026/27 BPS backcasting would require Opta event fields for times
  tackled, save location, and big-chance saves. Those fields are not available
  from the FPL history endpoints, so no invented proxy is used.

## Sources

- [All 2026/27 FPL changes](https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627)
- [2026/27 position changes](https://www.premierleague.com/en/news/4679886/position-changes-for-202627-fantasy-premier-league/)
- [2026/27 BPS changes](https://www.premierleague.com/en/news/4679946/whats-new-in-202627-fantasy-changes-to-bonus-points-system)
- [2026/27 chip rules](https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627)
- [Official Price Change Predictor](https://www.premierleague.com/en/news/4680462/whats-new-in-202627-fantasy-price-change-predictor)
- [Official 2026/27 Fixture Difficulty Ratings](https://www.premierleague.com/en/news/4675493/get-the-fixture-difficulty-ratings-for-202627-fpl-season)
