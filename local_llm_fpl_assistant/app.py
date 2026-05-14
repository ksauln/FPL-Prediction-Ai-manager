"""Separate Streamlit app for answering FPL questions with a local LLM."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_llm_fpl_assistant.analytics import build_analysis_context
from local_llm_fpl_assistant.data_access import (
    enrich_user_team,
    fetch_fpl_team_from_api,
    load_next_predictions,
    load_predictions_for_horizon,
    load_team_from_csv,
    validate_user_team,
)
from local_llm_fpl_assistant.llm_client import (
    LocalLLMError,
    OllamaSettings,
    stream_chat_with_local_llm,
)
from local_llm_fpl_assistant.prompts import SYSTEM_PROMPT, build_user_prompt
from local_llm_fpl_assistant.web_search import SearchSettings, build_online_research


st.set_page_config(page_title="FPL Local LLM Assistant", layout="wide")


APP_STYLE = """
<style>
    :root {
        --assistant-bg: #f6f8fb;
        --assistant-panel: #ffffff;
        --assistant-border: #d7dfeb;
        --assistant-text: #122033;
        --assistant-muted: #61708a;
        --assistant-accent: #14532d;
        --assistant-accent-soft: #dcfce7;
        --assistant-warn: #92400e;
        --assistant-warn-soft: #ffedd5;
    }

    .main .block-container {
        padding-top: 1.25rem;
        max-width: 1450px;
    }

    div[data-testid="stMetric"] {
        background: var(--assistant-panel);
        border: 1px solid var(--assistant-border);
        border-radius: 10px;
        padding: 0.85rem 0.9rem;
        box-shadow: 0 1px 2px rgba(18, 32, 51, 0.05);
    }

    .assistant-note {
        color: var(--assistant-muted);
        font-size: 0.95rem;
        margin-bottom: 0.7rem;
    }

    .assistant-box {
        background: var(--assistant-panel);
        border: 1px solid var(--assistant-border);
        border-radius: 12px;
        padding: 1rem 1.05rem;
        margin-bottom: 0.9rem;
    }
</style>
"""


SESSION_ANALYSIS_KEY = "local_llm_analysis_context"
SESSION_TEAM_KEY = "local_llm_user_team"
SESSION_CHAT_KEY = "local_llm_chat_history"


def _apply_style() -> None:
    st.markdown(APP_STYLE, unsafe_allow_html=True)


def _format_float(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _build_sidebar_settings() -> Dict[str, object]:
    st.sidebar.header("Local Model")
    base_url = st.sidebar.text_input("Base URL", value="http://localhost:11434")
    model = st.sidebar.text_input("Model", value="llama3.1:8b")
    timeout_seconds = st.sidebar.number_input(
        "Timeout (seconds)", min_value=10, max_value=600, value=120, step=10
    )
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.sidebar.header("Planning Horizon")
    horizon = st.sidebar.slider("Gameweeks", min_value=1, max_value=6, value=3, step=1)
    free_transfers = st.sidebar.slider("Free transfers", min_value=0, max_value=5, value=1, step=1)
    max_transfers = st.sidebar.slider(
        "Max suggested transfers", min_value=0, max_value=5, value=2, step=1
    )

    st.sidebar.header("Online Lookup")
    online_search_enabled = st.sidebar.checkbox(
        "Search web when useful",
        value=True,
        help="Adds current search snippets for questions about injuries, latest news, suspensions, or availability.",
    )
    search_results = st.sidebar.slider("Search results", min_value=1, max_value=8, value=5, step=1)
    search_timeout = st.sidebar.number_input(
        "Search timeout (seconds)", min_value=3, max_value=30, value=10, step=1
    )
    manual_query = st.sidebar.text_input(
        "Manual search query",
        value="",
        help="Optional. If set, the app will use this query when asking the assistant.",
    )

    return {
        "ollama": OllamaSettings(
            base_url=base_url,
            model=model,
            timeout_seconds=int(timeout_seconds),
            temperature=float(temperature),
        ),
        "horizon": int(horizon),
        "free_transfers": int(free_transfers),
        "max_transfers": int(max_transfers),
        "search": SearchSettings(
            enabled=bool(online_search_enabled),
            max_results=int(search_results),
            timeout_seconds=int(search_timeout),
            manual_query=manual_query,
        ),
    }


def _render_summary_metrics(context: Dict[str, object]) -> None:
    scope = context["analysis_scope"]
    squad = context["user_squad"]
    comparison = context["optimal_comparison"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current GW", f"GW{scope['current_gameweek']}")
    col2.metric("Projected Squad Pts", _format_float(squad["projected_points_with_captain"]))
    col3.metric("Gap To Optimal", _format_float(comparison["points_gap"]))
    col4.metric("Squad Rating", f"{_format_float(comparison['rating_pct'])}%")


def _render_team_table(team_df: pd.DataFrame) -> None:
    display_cols = [
        "full_name",
        "team_name",
        "element_type",
        "now_cost_millions",
        "expected_points",
        "confidence_level",
        "confidence_score",
        "start_probability",
        "starting",
        "bench",
        "captain",
    ]
    display_cols = [col for col in display_cols if col in team_df.columns]
    st.dataframe(
        team_df[display_cols].sort_values(["starting", "expected_points"], ascending=[False, False]),
        use_container_width=True,
        hide_index=True,
    )


def _render_transfer_preview(context: Dict[str, object]) -> None:
    plan = context["transfer_plan"]
    st.subheader("Transfer Preview")
    if not plan["recommendations"]:
        st.info("No transfers suggested for the selected constraints and horizon.")
        return

    rows: List[Dict[str, object]] = []
    for item in plan["recommendations"]:
        rows.append(
            {
                "Out": item["out"]["full_name"],
                "In": item["in"]["full_name"],
                "Delta": item["expected_points_delta"],
                "Out EP": item["out"]["expected_points"],
                "In EP": item["in"]["expected_points"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_risk_flags(context: Dict[str, object]) -> None:
    risk_flags = context["risk_flags"]
    st.subheader("Risk Flags")
    if not risk_flags:
        st.info("No material risk flags were found in the current squad context.")
        return

    for item in risk_flags:
        issues = ", ".join(item["issues"])
        news = item["news"] or "No news text available."
        st.markdown(f"- `{item['player']}`: {issues}. {news}")


def _analyse_team(
    user_team: pd.DataFrame,
    *,
    horizon: int,
    free_transfers: int,
    max_transfers: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    validate_user_team(user_team)
    next_gw, _, current_predictions = load_next_predictions()
    loaded_gws, predictions_by_gw, missing_gws = load_predictions_for_horizon(horizon)
    if loaded_gws and loaded_gws[0] != next_gw:
        next_gw = loaded_gws[0]

    enriched_team = enrich_user_team(user_team, current_predictions)
    context = build_analysis_context(
        user_team=enriched_team,
        current_predictions=current_predictions,
        predictions_by_gw=predictions_by_gw,
        loaded_gws=loaded_gws,
        missing_gws=missing_gws,
        free_transfers=free_transfers,
        max_transfers=max_transfers,
    )
    context["analysis_scope"]["current_gameweek"] = int(next_gw)
    return enriched_team, context


def _handle_team_input() -> Optional[pd.DataFrame]:
    st.subheader("1. Load Your Team")
    source = st.radio("Team source", options=["FPL ID", "CSV upload"], horizontal=True)

    if source == "FPL ID":
        fpl_id = st.number_input("FPL Team ID", min_value=1, step=1, value=1)
        event = st.number_input(
            "Gameweek to load picks from",
            min_value=1,
            step=1,
            value=1,
            help="Usually use the most recently completed or currently set team.",
        )
        if st.button("Load team from FPL ID", use_container_width=True):
            return fetch_fpl_team_from_api(int(fpl_id), int(event))
        return None

    uploaded = st.file_uploader("Upload team CSV", type=["csv"])
    if uploaded is not None and st.button("Load team from CSV", use_container_width=True):
        return load_team_from_csv(uploaded)
    return None


def _render_chat_interface(
    context: Dict[str, object],
    ollama: OllamaSettings,
    search_settings: SearchSettings,
) -> None:
    st.subheader("2. Ask The Assistant")
    st.markdown(
        '<div class="assistant-note">The model answers from the structured analysis and optional current web snippets.</div>',
        unsafe_allow_html=True,
    )

    chat_history = st.session_state.setdefault(SESSION_CHAT_KEY, [])
    quick_questions = [
        "What is the biggest weakness in my squad this week?",
        "What transfer would improve my team the most over this horizon?",
        "Who should I captain and why?",
        "Should I bench anyone differently?",
        "Is there any current injury news that changes my plan?",
    ]

    selected_quick_question = st.selectbox(
        "Quick question",
        options=[""] + quick_questions,
        index=0,
    )
    question = st.text_area(
        "Your question",
        value=selected_quick_question,
        placeholder="Ask about transfers, captaincy, benching, or where your team is losing points.",
        height=100,
    )

    if st.button("Ask local LLM", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            answer = ""
            online_research: Dict[str, object] = {}
            with st.status("Thinking...", expanded=True) as status:
                st.write("Checking your squad context.")
                st.write("Searching for current news if the question needs it.")
                online_research = build_online_research(
                    question=question,
                    context=context,
                    settings=search_settings,
                )
                if online_research.get("searched"):
                    if online_research.get("error"):
                        st.write("Online lookup failed; continuing with local projections.")
                    else:
                        result_count = len(online_research.get("results", []))
                        st.write(f"Found {result_count} current search result(s).")
                else:
                    st.write("No online lookup needed for this question.")

                prompt_context = dict(context)
                prompt_context["online_research"] = online_research
                user_prompt = build_user_prompt(question, prompt_context)
                st.write("Streaming response from the local model.")
                status.update(label="Streaming answer...", state="running", expanded=True)

                try:
                    answer = st.write_stream(
                        stream_chat_with_local_llm(
                            system_prompt=SYSTEM_PROMPT,
                            user_prompt=user_prompt,
                            settings=ollama,
                        )
                    )
                except LocalLLMError as exc:
                    status.update(label="Answer failed", state="error", expanded=True)
                    st.error(str(exc))
                else:
                    status.update(label="Answer complete", state="complete", expanded=False)
                    chat_history.append(
                        {
                            "question": question.strip(),
                            "answer": answer,
                            "online_research": online_research,
                        }
                    )
                    st.session_state[SESSION_CHAT_KEY] = chat_history

    if chat_history:
        st.subheader("Chat History")
        for item in reversed(chat_history):
            st.markdown('<div class="assistant-box">', unsafe_allow_html=True)
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(item["answer"])
            research = item.get("online_research", {})
            if research.get("searched"):
                if research.get("error"):
                    st.caption(f"Online lookup failed: {research['error']}")
                else:
                    result_count = len(research.get("results", []))
                    st.caption(f"Online lookup: {result_count} result(s) for `{research.get('query', '')}`")
            st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    _apply_style()
    st.title("FPL Local LLM Assistant")
    st.caption(
        "Separate sidecar app that reuses this repo's prediction outputs and explains them with a local LLM."
    )

    settings = _build_sidebar_settings()
    team_df = _handle_team_input()
    if team_df is not None:
        try:
            enriched_team, context = _analyse_team(
                team_df,
                horizon=settings["horizon"],
                free_transfers=settings["free_transfers"],
                max_transfers=settings["max_transfers"],
            )
        except Exception as exc:
            st.error(str(exc))
            return
        st.session_state[SESSION_TEAM_KEY] = enriched_team
        st.session_state[SESSION_ANALYSIS_KEY] = context
        st.session_state[SESSION_CHAT_KEY] = []

    stored_team = st.session_state.get(SESSION_TEAM_KEY)
    stored_context = st.session_state.get(SESSION_ANALYSIS_KEY)

    if stored_team is None or stored_context is None:
        st.info("Load a team to generate the assistant context.")
        return

    _render_summary_metrics(stored_context)

    left, right = st.columns([1.25, 1.0])
    with left:
        st.subheader("Current Team Context")
        _render_team_table(stored_team)
        _render_chat_interface(stored_context, settings["ollama"], settings["search"])

    with right:
        _render_transfer_preview(stored_context)
        _render_risk_flags(stored_context)
        st.subheader("Lineup Recommendation")
        lineup = stored_context["lineup_advice"]
        st.markdown(
            f"Suggested formation: `{lineup['suggested_formation']}`  \n"
            f"Suggested captain: `{lineup['captain']}`  \n"
            f"Projected points: `{_format_float(lineup['projected_points_with_captain'])}`"
        )
        st.subheader("Structured Context")
        st.json(stored_context, expanded=False)


if __name__ == "__main__":
    main()
