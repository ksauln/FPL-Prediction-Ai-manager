"""Behavior checks for the new decision presentation and initial view."""

from copy import deepcopy

import pytest
from streamlit.testing.v1 import AppTest

from fplmodel.manager_dashboard import _player_card_html, _recommendation_html


def test_recommendation_uses_saved_values_escapes_content_and_does_not_mutate():
    decision = {
        "gameweek": 9, "captain_id": 1, "vice_captain_id": 2,
        "transfer_hit_cost": 4,
        "transfers": [{"out_player": {"full_name": "Out <FC>"}, "in_player": {"full_name": "In <FC>"}}],
        "team": {"squad": [
            {"player_id": 1, "full_name": "Captain <script>", "expected_points": 7.2},
            {"player_id": 2, "full_name": "Vice & Co", "expected_points": 6.1},
        ]},
    }
    before = deepcopy(decision)
    html = _recommendation_html(decision)
    assert "Make 1 transfer" in html
    assert "4 pts" in html
    assert "Captain &lt;script&gt;" in html and "<script>" not in html
    assert "Out &lt;FC&gt;" in html and "In &lt;FC&gt;" in html
    assert "Vice &amp; Co" in html and "7.2 projected pts" in html
    assert decision == before


@pytest.mark.parametrize("decision,title", [
    ({"gameweek": 1}, "Set starting squad"),
    ({"gameweek": 9}, "Hold squad"),
    ({"gameweek": 9, "chip": "free_hit"}, "Play Free Hit"),
])
def test_partial_decisions_do_not_invent_captain_forecasts_or_rollover_reason(decision, title):
    html = _recommendation_html(decision)
    assert title in html
    assert "Projection unavailable" in html
    assert "Keep two" not in html
    assert "7.6" not in html


@pytest.mark.parametrize("completed,expected", [(True, "results"), (False, "run")])
def test_initial_manager_view_uses_available_results_but_keeps_all_views_accessible(monkeypatch, completed, expected):
    from pathlib import Path
    import streamlit as st
    import fplmodel.manager_dashboard as d
    from fplmodel.prediction_artifacts import PredictionSeasonSource

    source = PredictionSeasonSource("2026-27", Path("missing"), Path("missing/bootstrap.json"), False)
    monkeypatch.setattr(d, "_season_sources", lambda: {"2026-27": source})
    monkeypatch.setattr(d, "_default_season_name", lambda: "2026-27")
    monkeypatch.setattr(d, "list_manager_jobs", lambda: ([{"status": "completed", "request": {"season_name": "2026-27"}}] if completed else []))
    monkeypatch.setattr(d, "_render_run_view", lambda *args: st.caption("run view"))
    monkeypatch.setattr(d, "_render_results_view", lambda *args: st.caption("results view"))
    monkeypatch.setattr(d, "_render_manager_state_view", lambda *args: st.caption("state view"))
    app = AppTest.from_string('from fplmodel.manager_dashboard import render_ai_manager_page\nrender_ai_manager_page()').run()
    assert not app.exception
    assert app.session_state["ai_manager_view"] == expected
    app.button_group(key="ai_manager_view").set_value(["state"]).run()
    assert not app.exception
    assert app.caption[0].value == "state view"


def test_missing_squad_is_recoverable_and_does_not_offer_state_writes():
    app = AppTest.from_string('''
from fplmodel.manager_dashboard import _render_gameweek_decision
_render_gameweek_decision({"gameweek": 9})
''').run()
    assert not app.exception
    assert any("Squad details are unavailable" in item.value for item in app.info)
    assert not app.button


@pytest.mark.parametrize("points", [None, float("nan"), "unavailable"])
def test_missing_player_projection_is_not_presented_as_zero(points):
    html = _player_card_html(
        {"player_id": 1, "full_name": "Player", "expected_points": points},
        captain_id=None, vice_captain_id=None, bench=False,
    )
    assert "— pts" in html
    assert "0.0 pts" not in html
