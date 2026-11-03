# Changelog

## 2026-11-03

- Activate improved v2 as the default UI and weekly-continuation engine. Migrate
  historical legacy settings for future jobs without changing committed squads
  or historical results; retain explicit legacy fallback.
- Add an explicit `read_only` job flag. New live-v2 jobs write a candidate state
  and allow the existing confirmed save action. Historical v2 trials remain
  read-only and cannot configure continuation.
- Use 128 scenarios when migrating from legacy sampling counts; retain saved
  horizons, transfer limits, season rules and live-v2 strategy settings.
- Regenerate GW10–15 from saved GW9 state. The GW10 v2 action uses one free
  transfer and no hit; the paid second move fails its net-gain threshold.
- Verify 302 tests and preserve existing runtime files. State stays through GW9
  until the owner applies and saves the new recommendation.

## 2026-10-31

- Introduce Orbital's responsive dark dashboard, tactical pitch, jersey markers,
  recorded-decision summary and compact mobile metrics. Publish desktop/mobile
  screenshots and document native Streamlit accessibility limitations.
- Add the opt-in v2 manager with weekly beam search, transfer rolling, chip
  planning, captain/vice pairing and shared scenario validation. Preserve the
  legacy default and prevent v2 jobs from committing or continuing live state.
- Require paid v2 transfers to justify their marginal net advantage over the
  best no-hit alternative. Protect free alternatives from shortlist pruning and
  expose gain-buffer and scenario-frequency controls.
- Add fixture-level forecast provenance, actual-outcome scoring, historical
  replay and controlled hit-policy comparison commands. Document reconstructed
  horizon limitations, original controls, before/after results and local-only
  reproduction inputs.
- Add preseason enrichment, historical-universe, cutoff-safe replay,
  model-cache and automatic manager-continuation improvements with their tests
  and runbooks.
- Add the separate Astra assistant integration and documentation;
  API credentials remain server-side and incomplete answers are not saved.
- Document local and approved same-Wi-Fi access, future hosting options and the
  distinction between a GitHub repository and a running Python service. No
  public hosting is configured.
- Verify 295 automated tests. Preserve runtime data, model files, saved season
  state and completed jobs; these generated changes are not part of the source
  release.
- Stop tracking previously committed API caches, external-data gitlink, model
  artifacts, generated outputs and logs. Keep local copies intact and document
  fresh-clone rebuild steps and the live-state/private evidence backup boundary.
