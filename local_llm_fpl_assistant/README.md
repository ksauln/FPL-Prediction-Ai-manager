# FPL Astra Assistant

This separate Streamlit app explains the repository's FPL analysis using GPT-6 Astra.
The prediction pipeline, optimization, and AI Manager continue to produce the numbers.
The assistant cannot submit changes to the official FPL website.

## Setup

From the repository root, activate your Python environment and install dependencies:

```bash
source fpl_venv/bin/activate
python -m pip install -r requirements.txt
```

Configure `OPENAI_API_KEY` in the environment of the Streamlit server using your
secret manager or a shell prompt that does not echo the key. For zsh:

```zsh
read -rs 'OPENAI_API_KEY?OpenAI API key: '
export OPENAI_API_KEY
python -m streamlit run local_llm_fpl_assistant/app.py
```

The app does not automatically load a `.env` file. Never commit the key.
Your API project must have billing and access to `gpt-6-astra`. Missing credentials
disable the answer button; the analysis remains available. The package directory
retains its original name so existing launch commands still work.

## Prediction inputs and use

Generate prediction artifacts with the main dashboard's **Build predictions through**
control, or run the normal prediction pipeline. The assistant reads the latest root
`outputs/predictions_gw<N>.csv` files, not archived season directories.

1. Select reasoning effort (default **low**) and timeout.
2. Select a one-to-six-Gameweek horizon, free transfers, and transfer limit.
3. Choose **Saved AI Manager** and **Load saved team** to plan from the week after
   your committed decision, using saved bank, purchase prices, and free transfers.
   Alternatively load a full squad by FPL ID or CSV; these inputs lack financial
   history and their transfer affordability is explicitly unverified.
4. Review the projected points, transfer preview, risk flags, and lineup.
5. Enter a question and press **Ask Astra**.

CSV requires 15 unique `player_id` values. Optional columns are `starting`,
`bench`, `captain`, and `full_name`. Use CSV when no completed FPL picks exist.
Changing the team, planning controls, saved state, or prediction files rebuilds
the analysis and clears displayed chat history. **Refresh analysis** also reloads
news/status inputs. Saved free transfers override the sidebar estimate. Saved-team
forecasts must match the saved season and start immediately after the committed
week; missing or stale files require a prediction refresh. Chip strategy remains
in AI Manager; the assistant's transfer preview uses its regular-transfer optimizer.
Questions are independent:
previous answers are displayed locally but are not sent as conversation history.

Online lookup uses DuckDuckGo Lite for freshness-sensitive questions, or whenever
a manual search query is supplied. Search failures do not prevent an answer from
the available projections. Search snippets are untrusted evidence and the prompt
requires citations, uncertainty disclosure, and no invented news or numbers.

## Data handling and cost

Each answer sends the question, structured squad analysis, and optional search
snippets to OpenAI. Disabling online lookup disables DuckDuckGo requests; it does
not make Astra local. The entire repository is not uploaded. FPL-ID input also
contacts the public FPL API; CSV avoids that picks request.

Requests use the Responses API with `store=False`, disabling response storage
for later retrieval. This is not a promise of zero retention; OpenAI's separate
data-handling policies still apply. Keys are read server-side and are not placed
in Streamlit session state. Provider errors are sanitized before display.

API usage is billed to your API project. Higher reasoning effort can increase
latency and token usage; start with low and compare representative questions.
Check the current [Astra model pricing](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [API data controls](https://developers.openai.com/api/docs/guides/your-data).
There are no automatic retries, avoiding repeated requests after failures.

## Troubleshooting

- **Missing key:** export `OPENAI_API_KEY` in the process environment before launching Streamlit.
- **Authentication failed:** check the configured API key.
- **Model unavailable:** check your API project's Astra access.
- **Rate or quota limit:** check project billing and limits; retry later.
- **Timeout:** increase the timeout or reduce reasoning effort.
- **Incomplete/failed stream:** retry; partial text is not saved as a completed answer.
- **No prediction files:** build predictions first.
- **Missing future Gameweeks:** rebuild through the desired horizon and reload the team.
- **Online lookup failed:** retry or verify news directly using an authoritative source.

## Verification and rollback

Run `python -m pytest tests/test_astra_client.py tests/test_astra_app.py tests/test_local_llm_assistant.py -q`.
The tests mock OpenAI and do not incur API charges. Live access, answer quality,
and latency require a configured API key; use the cases in
[the Astra evaluation checklist](../docs/astra-evaluation.md).

Before deploying, commit the migration separately from other work. To roll back,
revert that migration commit and reinstall its requirements; preserve unrelated
edits and data artifacts. Ollama is no longer a runtime option in this version.

Implementation follows the official
[Astra migration guide](https://developers.openai.com/api/docs/guides/latest-model)
and [Responses API guide](https://developers.openai.com/api/docs/guides/migrate-to-responses).
