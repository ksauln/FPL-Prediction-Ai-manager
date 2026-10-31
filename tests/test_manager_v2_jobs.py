import json

import pytest

from fplmodel.manager_jobs import (
    _continuation_template, _update_status, commit_recommended_state,
    create_manager_job, validate_job_request, run_manager_job,
)
from fplmodel.season_manager import save_manager_state
from test_manager_jobs import _request
from test_manager_v2 import frames, state


def request(**kwargs):
    return _request(simulations=8, start_gw=2, end_gw=2,
        manager_config={"manager_version": "v2", "enable_chips": False,
                        "max_transfers_per_gw": 0, "planning_beam_width": 1}, **kwargs)


def test_v2_trial_cannot_commit_even_when_a_recommended_state_exists(tmp_path):
    created = create_manager_job(request(), job_root=tmp_path / "jobs")
    job_id = created["job_id"]
    job_dir = tmp_path / "jobs" / job_id
    candidate = state()
    candidate.last_processed_gameweek = 2
    save_manager_state(candidate, job_dir / "recommended_state.json")
    _update_status(job_dir, status="completed")
    destination = tmp_path / "live.json"
    save_manager_state(state(), destination)
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="trial"):
        commit_recommended_state(job_id, destination, job_root=tmp_path / "jobs")
    assert destination.read_bytes() == before


def test_v2_is_never_an_automatic_continuation_template():
    v2 = {"status": "completed", "request": {"season_name": "2026-27", "manager_config": {"manager_version": "v2"}}}
    old = {"status": "completed", "request": {"season_name": "2026-27"}}
    assert _continuation_template([v2, old], season_name="2026-27") is old
    assert _continuation_template([v2], season_name="2026-27") is None


def test_invalid_engine_and_excessive_v2_simulations_are_rejected():
    with pytest.raises(ValueError, match="manager_version"):
        validate_job_request(_request(manager_config={"manager_version": "wrong"}))
    with pytest.raises(ValueError, match="10000"):
        validate_job_request(_request(manager_config={"manager_version": "v2"}))


def test_v2_worker_runs_a_shadow_of_live_state(tmp_path):
    data = frames((2,))
    source = tmp_path / "predictions"
    source.mkdir()
    data[2].to_csv(source / "predictions_gw2.csv", index=False)
    destination = tmp_path / "live.json"
    save_manager_state(state(), destination)
    before = destination.read_bytes()
    job = create_manager_job(request(run_mode="live", output_dir=str(source),
                                    manager_state_path=str(destination)), job_root=tmp_path / "jobs")
    result = run_manager_job(tmp_path / "jobs" / job["job_id"])
    assert result["status"] == "completed"
    assert result["recommended_state_path"] is None
    assert destination.read_bytes() == before
