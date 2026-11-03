# Orbital UI

The selected Orbital design is implemented in the existing Streamlit dashboard: charcoal surfaces, cyan pitch lines, lime actions, compact navigation, vector jersey markers and a recommendation column beside the squad. Performance charts use the same dark palette.

Completed results open on the saved gameweek squad before simulation analysis. Active runs still open in the run view. The run/results/state controls and page navigation remain available. Player tables and season analysis sit in expandable sections. The recommendation column displays recorded transfers, chip choice, captain/vice projections and actual hit costs; missing projections say unavailable. It does not fabricate a strategic reason, probability or performance history.

No engine, prediction, state-save, job-start or automatic-update logic was changed. That UI-only release initially retained the legacy default. The 2026-11-03
live-v2 activation now makes the improved engine the default while retaining
explicit save confirmation and historical trial isolation; see [Manager v2](manager-v2.md).

## Files

- `.streamlit/config.toml`: dark theme for native controls.
- `fplmodel/ui_theme.py`: shared presentation asset loading and brand markup.
- `fplmodel/assets/orbital.css`: global theme, sidebar, controls, focus and reduced-motion styles.
- `fplmodel/assets/orbital_pitch.css`: responsive pitch, jerseys, bench and recommendation styling.
- `streamlit_app.py`: shared theme and Orbital branding.
- `fplmodel/manager_dashboard.py`: jerseys, recommendation display, expandable details and initial view selection.
- `fplmodel/team_performance_display.py`: chart palette and backgrounds.
- `tests/test_orbital_dashboard.py`: escaped saved values, non-mutating presentation, partial decisions, initial view selection and missing-squad handling.
- `docs/orbital-ui-plan.md`: scope and verification plan.

## Verification

`fpl_venv/bin/python -m pytest -q` passed **295 tests** in 58.76 seconds with one existing joblib physical-core warning. Compilation and `git diff --check` passed. The actual app loaded its latest saved GW9 result under Streamlit AppTest without app exceptions. The existing trial and live-job tests passed.

All **1,030 protected data, model, prediction and saved-job/state files** are hash-identical to the start of this UI task. Backups of the four existing UI/config sources and the before/after preservation manifests are in `outputs/ui/orbital/20261031T134850Z`.

This ignored local evidence directory is not included in a fresh Git clone.
Published examples are [desktop squad](screenshots/orbital-desktop.png),
[desktop overview](screenshots/orbital-overview.png), and
[mobile recommendation](screenshots/orbital-mobile.png).

The same local directory contains `component-preview.html`, a standalone read-only export of the actual pitch and recommendation components with real saved policy values. Its surrounding static header/metrics are a review wrapper; the full app's controls remain in Streamlit. This artifact has not been browser-rendered or visually approved.

Desktop (1440×1000) and mobile (390×844) rendering checks are complete after the user explicitly authorized the browser connector and preview server. The approved checks found and fixed oversized pitch headings, stacked mobile metrics, a low-contrast caption, missing focus styling on expandable summaries, and missing player projections appearing as zero. Run/gameweek controls now share a desktop row; the summary uses a compact responsive grid.

The browser measured no horizontal overflow in the squad/recommendation or a synthetic 3-5-2 lineup with five long midfielder names. The mobile bench forms two columns. Enter opens/closes squad details; Tab and Shift+Tab show a cyan 2px focus ring with 3px offset. Main view controls are 44px high. Loading, empty, failed, partial and trial-permission states were inspected using an isolated, read-only verification fixture; success uses the actual saved GW9 policy. Ten screenshots and a detailed `evidence.json` are in the evidence directory. The design skill's evidence checker passes for the changed app-owned presentation.

The desktop Lighthouse snapshot scored accessibility 89 and best practices 100. Color contrast passes after the caption fix. Native Streamlit sidebar/menu/table wrappers still have unsupported ARIA attributes, and its built-in menu button lacks an accessible name. These framework limitations are recorded in `lighthouse-desktop.json`; no DOM patches or dependency changes were introduced. Physical-phone testing and exhaustive interactions on all secondary toolkit pages were not performed.

The temporary fixture server was stopped after verification. To preview locally, launch Streamlit and open `http://127.0.0.1:8501`. Restart an already-running Streamlit process to ensure the native theme and stylesheet assets reload. See [access and hosting](access-and-hosting.md) for phone access and hosting boundaries:

```sh
fpl_venv/bin/python -m streamlit run streamlit_app.py
```
