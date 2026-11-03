# Activate the improved live manager

The user wants the improved engine to power actual weekly recommendations.
The previous rollout kept v2 in trials, so automatic refreshes continued using
legacy. Change the live entry points without rewriting historical jobs or
advancing the current squad.

1. Back up current live state and relevant sources; record hashes of existing
   data, models, forecasts and completed jobs.
2. Test explicit v2 live jobs end to end: they write a separate recommended
   state, accept the existing explicit commit action, and become continuation
   templates. Existing unmarked v2 trials remain read-only.
3. Default the UI and legacy-template migration to v2. Convert legacy sampling
   counts to v2's 128-scenario default while retaining squad, finances, horizons,
   season rules and strategy limits. Retain explicit legacy selection as a
   fallback.
4. Run targeted and full tests. Start a fresh v2 job from saved GW9 using the
   current cached forecasts. Inspect the GW10 action and hit diagnostics. Do not
   save that decision into live state or edit prior recommendations.
5. Update current guides and release notes to distinguish live v2 from historical
   read-only trials, then verify all protected files remain unchanged.

The performance evidence remains limited; enabling the engine does not promise
that it will never pay for transfers. Its first action must clear the existing
marginal-gain and shared-scenario checks.
