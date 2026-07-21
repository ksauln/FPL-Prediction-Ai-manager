from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from fplmodel.manager_jobs import (
    _update_status,
    cancel_manager_job,
    commit_recommended_state,
    create_manager_job,
    list_manager_jobs,
    load_manager_job,
    start_manager_job,
    validate_job_request,
)


def _request(**overrides: object) -> dict[str, object]:
    request = {
        "season_name": "2026-27",
        "run_mode": "planning",
        "start_gw": 1,
        "end_gw": 38,
        "simulations": 50_000,
        "simulation_mode": "periodic_reoptimization",
        "policy_refresh_interval": 1_000,
        "random_seed": 90,
        "manager_config": {
            "initial_horizon": 4,
            "transfer_horizon": 4,
            "chip_lookahead": 4,
            "enable_chips": True,
        },
        "rules": {},
    }
    request.update(overrides)
    return request


class ManagerJobTests(unittest.TestCase):
    def test_validate_job_request_normalises_values(self) -> None:
        validated = validate_job_request(
            _request(
                start_gw="2",
                simulations="100",
                output_dir="outputs/seasons/2026-27",
            )
        )

        self.assertEqual(validated["start_gw"], 2)
        self.assertEqual(validated["simulations"], 100)
        self.assertEqual(validated["gameweeks"], list(range(2, 39)))
        self.assertEqual(validated["output_dir"], "outputs/seasons/2026-27")

    def test_validate_job_request_rejects_invalid_ranges_and_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "end_gw"):
            validate_job_request(_request(start_gw=4, end_gw=3))
        with self.assertRaisesRegex(ValueError, "simulation_mode"):
            validate_job_request(_request(simulation_mode="manual"))
        with self.assertRaisesRegex(ValueError, "season_name"):
            validate_job_request(_request(season_name="next season"))

    def test_create_and_list_jobs_persist_request_and_status(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            created = create_manager_job(_request(), job_root=root)
            loaded = load_manager_job(created["job_id"], job_root=root)
            listed = list_manager_jobs(job_root=root)

        self.assertEqual(loaded["status"], "queued")
        self.assertEqual(loaded["request"]["simulations"], 50_000)
        self.assertEqual(listed[0]["job_id"], created["job_id"])

    def test_start_job_launches_worker_and_records_pid(self) -> None:
        process = Mock(pid=4321)
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "jobs"
            created = create_manager_job(_request(), job_root=root)
            with patch("fplmodel.manager_jobs.subprocess.Popen", return_value=process) as popen:
                started = start_manager_job(
                    created["job_id"],
                    job_root=root,
                    project_root=Path(tmpdir),
                    python_executable="/test/python",
                )

            command = popen.call_args.args[0]
            log_handle = popen.call_args.kwargs["stdout"]

        self.assertEqual(started["status"], "running")
        self.assertEqual(started["pid"], 4321)
        self.assertEqual(command[:3], ["/test/python", "-m", "fplmodel.manager_jobs"])
        self.assertTrue(log_handle.closed)

    def test_start_job_records_launch_failure(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "jobs"
            created = create_manager_job(_request(), job_root=root)
            with patch(
                "fplmodel.manager_jobs.subprocess.Popen",
                side_effect=OSError("cannot launch"),
            ):
                with self.assertRaisesRegex(OSError, "cannot launch"):
                    start_manager_job(
                        created["job_id"],
                        job_root=root,
                        project_root=Path(tmpdir),
                    )
            failed = load_manager_job(created["job_id"], job_root=root)

        self.assertEqual(failed["status"], "failed")
        self.assertIn("cannot launch", failed["error"])

    def test_cancel_preserves_job_that_completes_during_signal(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "jobs"
            created = create_manager_job(_request(), job_root=root)
            job_dir = root / created["job_id"]
            status_path = job_dir / "status.json"
            running = json.loads(status_path.read_text(encoding="utf-8"))
            running.update(status="running", phase="simulating", pid=4321)
            status_path.write_text(json.dumps(running), encoding="utf-8")

            def finish_worker(_pid: int, _signal: int) -> None:
                _update_status(job_dir, status="completed", phase="complete")

            with (
                patch("fplmodel.manager_jobs._process_exists", return_value=True),
                patch("fplmodel.manager_jobs.os.kill", side_effect=finish_worker),
            ):
                status = cancel_manager_job(created["job_id"], job_root=root)

        self.assertEqual(status["status"], "completed")

    def test_commit_recommended_state_requires_completed_job(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "jobs"
            created = create_manager_job(_request(), job_root=root)
            with self.assertRaisesRegex(RuntimeError, "completed"):
                commit_recommended_state(
                    created["job_id"],
                    Path(tmpdir) / "manager_state.json",
                    job_root=root,
                )

    def test_commit_recommended_state_writes_valid_state(self) -> None:
        candidate = {
            "squad_player_ids": list(range(1, 16)),
            "bank_m": 1.2,
            "free_transfers": 1,
            "purchase_price_by_player_id": {str(i): 5.0 for i in range(1, 16)},
            "used_chips": {"first": {}, "second": {}},
            "history": [],
            "last_processed_gameweek": 1,
            "season_name": "2026-27",
        }
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "jobs"
            created = create_manager_job(_request(), job_root=root)
            job_dir = root / created["job_id"]
            candidate_path = job_dir / "recommended_state.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            status_path = job_dir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status.update(status="completed", recommended_state_path=str(candidate_path))
            status_path.write_text(json.dumps(status), encoding="utf-8")
            destination = Path(tmpdir) / "manager_state.json"

            committed = commit_recommended_state(
                created["job_id"], destination, job_root=root
            )
            saved = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(saved["last_processed_gameweek"], 1)
        self.assertEqual(committed["committed_state_path"], str(destination))
        self.assertIn("state_committed_at", committed)

    def test_commit_recommended_state_rejects_stale_gameweek(self) -> None:
        candidate = {
            "squad_player_ids": list(range(1, 16)),
            "bank_m": 1.2,
            "free_transfers": 1,
            "purchase_price_by_player_id": {str(i): 5.0 for i in range(1, 16)},
            "used_chips": {"first": {}, "second": {}},
            "history": [],
            "last_processed_gameweek": 1,
            "season_name": "2026-27",
        }
        current = {**candidate, "last_processed_gameweek": 2}
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "jobs"
            created = create_manager_job(_request(), job_root=root)
            job_dir = root / created["job_id"]
            (job_dir / "recommended_state.json").write_text(
                json.dumps(candidate), encoding="utf-8"
            )
            status_path = job_dir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["status"] = "completed"
            status_path.write_text(json.dumps(status), encoding="utf-8")
            destination = Path(tmpdir) / "manager_state.json"
            destination.write_text(json.dumps(current), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expects GW3"):
                commit_recommended_state(
                    created["job_id"], destination, job_root=root
                )


if __name__ == "__main__":
    unittest.main()
