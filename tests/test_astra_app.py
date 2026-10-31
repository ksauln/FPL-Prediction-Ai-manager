from pathlib import Path

from streamlit.testing.v1 import AppTest
import pytest


def test_sidebar_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "local_llm_fpl_assistant/app.py")).run()
    assert not app.exception
    assert app.title[0].value == "FPL Astra Assistant"
    effort = next(item for item in app.selectbox if item.label == "Reasoning effort")
    assert effort.value == "low"
    assert effort.options == ["low", "medium", "high", "xhigh", "max"]
    assert any("OPENAI_API_KEY" in warning.value for warning in app.warning)
    assert not any(item.label in ("Base URL", "Model") for item in app.text_input)
    assert not any(item.label == "Temperature" for item in app.slider)


def test_chat_without_key_does_not_search(monkeypatch):
    from local_llm_fpl_assistant import app as app_module
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    def unexpected(**kwargs):
        raise AssertionError("Search must not run without credentials")
    monkeypatch.setattr(app_module, "build_online_research", unexpected)
    app = AppTest.from_string("""
from local_llm_fpl_assistant.app import _render_chat_interface
from local_llm_fpl_assistant.llm_client import AstraSettings
from local_llm_fpl_assistant.web_search import SearchSettings
_render_chat_interface({}, AstraSettings(), SearchSettings(enabled=True))
""").run()
    assert not app.exception
    assert app.button[0].disabled


@pytest.mark.parametrize("fail", [False, True])
def test_chat_only_saves_completed_answers(monkeypatch, fail):
    from local_llm_fpl_assistant import llm_client
    from local_llm_fpl_assistant import app as app_module
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    def stream(**kwargs):
        yield "Captain Mid A."
        if fail:
            raise llm_client.LLMError("The response stream ended before completion.")
    monkeypatch.setattr(app_module, "stream_chat_with_llm", stream)
    app = AppTest.from_string("""
from local_llm_fpl_assistant.app import _render_chat_interface
from local_llm_fpl_assistant.llm_client import AstraSettings
from local_llm_fpl_assistant.web_search import SearchSettings
_render_chat_interface({}, AstraSettings(), SearchSettings(enabled=False))
""").run()
    app.text_area[0].input("Who should I captain?").run()
    app.button[0].click().run()
    assert not app.exception
    history = app.session_state["local_llm_chat_history"]
    if fail:
        assert not history
        assert app.error
    else:
        assert history[0]["answer"] == "Captain Mid A."
    assert "test-secret" not in str(history)
