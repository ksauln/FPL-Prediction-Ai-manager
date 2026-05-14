# Local LLM FPL Assistant

This folder contains a separate Streamlit app that layers a local LLM on top of
the existing FPL prediction pipeline without modifying the original app or
pipeline flow.

## What it does

- Loads the latest prediction artifacts from `outputs/`
- Accepts a squad via FPL ID or CSV upload
- Reuses the existing deterministic analytics in `fplmodel/`
- Builds a structured team-analysis context
- Optionally searches online for current news when the question needs fresh context
- Sends that context to a local Ollama-compatible model for explanation

## Run

Start your local Ollama server first, then run:

```bash
streamlit run local_llm_fpl_assistant/app.py
```

## Expected local model setup

Default endpoint:

- `http://localhost:11434/api/chat`

Recommended starter models:

- `llama3.1:8b`
- `mistral:7b`
- `qwen2.5:7b-instruct`

## Online lookup

The app can add current search snippets to the prompt for questions involving:

- injury news
- availability
- suspensions
- press conference updates
- latest status

The lookup is intentionally narrow. The deterministic projection and transfer
numbers still come from this repository's generated artifacts; online snippets
are only supporting context for freshness-sensitive questions.

## CSV format

The CSV upload must include:

- `player_id`

Optional columns:

- `starting`
- `bench`
- `captain`
- `full_name`

If lineup columns are omitted, the app will infer a valid default split where
possible and still enrich the squad from prediction metadata.
