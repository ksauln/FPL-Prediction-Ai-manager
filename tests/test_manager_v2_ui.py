import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_improved_engine_is_default_and_fallback_settings_stay_isolated(tmp_path):
    script = f'''
from pathlib import Path
import fplmodel.manager_dashboard as dashboard
from fplmodel.prediction_artifacts import PredictionSeasonSource
dashboard.OUTPUTS_DIR = Path({str(tmp_path)!r})
dashboard._render_data_readiness = lambda *args: list(range(1, 39))
dashboard.list_manager_jobs = lambda: []
dashboard._load_state_if_available = lambda *args: None
source = PredictionSeasonSource("2026-27", Path({str(tmp_path)!r}), Path({str(tmp_path)!r}) / "bootstrap.json", False)
dashboard._render_run_view("2026-27", source)
'''
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert app.selectbox(key="ai_manager_engine_live").value == "v2"
    assert app.checkbox(key="ai_manager_read_only").value is False
    assert app.number_input(key="ai_manager_simulations_v2").value == 128
    assert app.number_input(key="ai_manager_transfer_limit_v2").value == 5
    assert app.number_input(key="ai_manager_paid_gain_buffer").value == 2.
    assert app.number_input(key="ai_manager_paid_min_probability").value == .75
    app.selectbox(key="ai_manager_engine_live").set_value("legacy").run()
    assert not app.exception
    assert app.number_input(key="ai_manager_simulations_legacy").value == 50000
    assert app.number_input(key="ai_manager_transfer_limit_legacy").value == 2


def test_trial_results_have_no_state_commit_control():
    app = AppTest.from_string('''
import fplmodel.manager_dashboard as dashboard
dashboard._render_state_commit({"recommended_state_path": "trial.json"}, {"manager_version": "v2"})
''').run()
    assert not app.exception
    assert not app.button
    assert "live manager state is preserved" in app.caption[0].value
