from pathlib import Path

import pytest

from fplmodel.manager_jobs import (
    _continuation_template, build_live_continuation_request,
    commit_recommended_state, create_manager_job, run_manager_job,
    validate_job_request,
)
from fplmodel.season_manager import load_manager_state, save_manager_state
from test_manager_jobs import _request
from test_manager_v2 import frames, state


def test_explicit_live_v2_writes_candidate_and_commits_only_when_requested(tmp_path):
    source = tmp_path / "predictions"
    source.mkdir()
    frames((2,))[2].to_csv(source / "predictions_gw2.csv", index=False)
    destination = tmp_path / "live.json"
    save_manager_state(state(), destination)
    before = destination.read_bytes()
    job = create_manager_job(_request(
        run_mode="live", start_gw=2, end_gw=2, simulations=8,
        read_only=False, output_dir=str(source), manager_state_path=str(destination),
        manager_config={"manager_version": "v2", "enable_chips": False,
                        "max_transfers_per_gw": 0, "planning_beam_width": 1},
    ), job_root=tmp_path / "jobs")
    completed = run_manager_job(tmp_path / "jobs" / job["job_id"])
    assert completed["recommended_state_path"] is not None
    assert destination.read_bytes() == before
    assert load_manager_state(Path(completed["recommended_state_path"])).last_processed_gameweek == 2
    commit_recommended_state(job["job_id"], destination, job_root=tmp_path / "jobs")
    assert load_manager_state(destination).last_processed_gameweek == 2
    with pytest.raises(ValueError, match="expects GW3"):
        commit_recommended_state(job["job_id"], destination, job_root=tmp_path / "jobs")


def test_only_explicitly_live_v2_jobs_can_be_continuation_templates():
    trial = {"status": "completed", "request": {"season_name": "2026-27", "manager_config": {"manager_version": "v2"}}}
    live = {"status": "completed", "request": {"season_name": "2026-27", "read_only": False, "manager_config": {"manager_version": "v2"}}}
    assert _continuation_template([trial, live], season_name="2026-27") is live
    assert _continuation_template([trial], season_name="2026-27") is None


def test_legacy_history_migrates_new_week_to_v2_without_mutating_state(tmp_path):
    destination = tmp_path / "live.json"
    save_manager_state(state(), destination)
    before = destination.read_bytes()
    old = {"status": "completed", "state_committed_at": "before",
           "request": _request(manager_config={"transfer_horizon": 4, "max_transfers_per_gw": 2})}
    request = build_live_continuation_request(
        season_name="2026-27", start_gw=2, end_gw=4, output_dir=tmp_path,
        manager_state_path=destination, previous_jobs=[old],
    )
    assert request["manager_config"]["manager_version"] == "v2"
    assert request["manager_config"]["max_transfers_per_gw"] == 2
    assert request["simulations"] == 128
    assert request["read_only"] is False
    assert "manager_version" not in old["request"]["manager_config"]
    assert destination.read_bytes() == before


def test_explicit_live_v2_preserves_sampling_and_paid_transfer_settings(tmp_path):
    destination = tmp_path / "live.json"
    save_manager_state(state(), destination)
    old = {"status": "completed", "request": _request(
        simulations=256, read_only=False,
        manager_config={"manager_version": "v2", "paid_transfer_gain_buffer": 3.0,
                        "paid_transfer_min_probability": .8},
    )}
    request = build_live_continuation_request(
        season_name="2026-27", start_gw=2, end_gw=4, output_dir=tmp_path,
        manager_state_path=destination, previous_jobs=[old],
    )
    assert request["simulations"] == 256
    assert request["manager_config"]["paid_transfer_gain_buffer"] == 3.0
    assert request["manager_config"]["paid_transfer_min_probability"] == .8
    assert request["read_only"] is False


def test_read_only_flag_must_be_boolean_and_old_trials_remain_read_only():
    with pytest.raises(ValueError, match="read_only"):
        validate_job_request(_request(read_only="false"))
    validated = validate_job_request(_request(simulations=8, manager_config={"manager_version": "v2"}))
    assert validated["read_only"] is True


def test_old_explicit_legacy_default_also_migrates_to_improved_engine(tmp_path):
    destination = tmp_path / "live.json"
    save_manager_state(state(), destination)
    old = {"status": "completed", "request": _request(manager_config={"manager_version": "legacy"})}
    request = build_live_continuation_request(
        season_name="2026-27", start_gw=2, end_gw=4, output_dir=tmp_path,
        manager_state_path=destination, previous_jobs=[old],
    )
    assert request["manager_config"]["manager_version"] == "v2"
    assert request["simulations"] == 128


def test_v2_live_save_control_keeps_explicit_application_confirmation(tmp_path):
    from streamlit.testing.v1 import AppTest

    destination = tmp_path / "live.json"
    save_manager_state(state(), destination)
    before = destination.read_bytes()
    app = AppTest.from_string(f'''
from fplmodel.manager_dashboard import _render_state_commit
_render_state_commit(
    {{"job_id": "live-v2", "recommended_state_path": "candidate.json", "request": {{"manager_state_path": {str(destination)!r}}}}},
    {{"manager_version": "v2", "read_only": False, "recommended_policy": {{"decisions": [{{"gameweek": 2}}]}}}},
)
''').run()
    assert not app.exception
    assert app.button[0].disabled
    app.checkbox[0].check().run()
    assert not app.exception
    assert not app.button[0].disabled
    assert destination.read_bytes() == before
