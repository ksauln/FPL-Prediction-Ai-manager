from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from fplmodel.manager_jobs import (
    _update_status,
    build_live_continuation_request,
    cancel_manager_job,
    commit_recommended_state,
    create_manager_job,
    list_manager_jobs,
    load_manager_job,
    start_live_manager_continuation,
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
    @staticmethod
    def _write_state(path: Path, *, last_processed_gameweek: int) -> None:
        payload = {
            "squad_player_ids": list(range(1, 16)),
            "bank_m": 1.2,
            "free_transfers": 1,
            "purchase_price_by_player_id": {str(i): 5.0 for i in range(1, 16)},
            "used_chips": {"first": {}, "second": {}},
            "history": [],
            "last_processed_gameweek": last_processed_gameweek,
            "season_name": "2026-27",
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

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

    def test_live_continuation_reuses_settings_but_advances_from_saved_state(self) -> None:
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "manager_state.json"
            self._write_state(state_path, last_processed_gameweek=7)
            previous_request = validate_job_request(
                _request(
                    start_gw=1,
                    end_gw=38,
                    simulations=12_345,
                    simulation_mode="fixed_policy",
                    policy_refresh_interval=321,
                    random_seed=77,
                    output_dir="old-predictions",
                    manager_state_path=str(state_path),
                    manager_config={"transfer_horizon": 6, "enable_chips": False},
                    rules={"max_free_transfers": 4},
                )
            )
            previous_jobs = [
                {
                    "status": "completed",
                    "state_committed_at": "2026-08-20T10:00:00+00:00",
                    "request": previous_request,
                }
            ]

            request = build_live_continuation_request(
                season_name="2026-27",
                start_gw=8,
                end_gw=11,
                output_dir="fresh-predictions",
                manager_state_path=state_path,
                previous_jobs=previous_jobs,
            )

        self.assertIsNotNone(request)
        self.assertEqual(request["run_mode"], "live")
        self.assertEqual(request["gameweeks"], [8, 9, 10, 11])
        self.assertEqual(request["simulations"], 12_345)
        self.assertEqual(request["simulation_mode"], "fixed_policy")
        self.assertEqual(request["policy_refresh_interval"], 321)
        self.assertEqual(request["random_seed"], 77)
        self.assertEqual(request["manager_config"]["transfer_horizon"], 6)
        self.assertFalse(request["manager_config"]["enable_chips"])
        self.assertEqual(request["rules"]["max_free_transfers"], 4)
        self.assertEqual(request["output_dir"], "fresh-predictions")
        self.assertEqual(request["manager_state_path"], str(state_path))

    def test_live_continuation_refuses_to_skip_uncommitted_gameweek(self) -> None:
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "manager_state.json"
            self._write_state(state_path, last_processed_gameweek=6)

            with self.assertRaisesRegex(ValueError, "expects GW7"):
                build_live_continuation_request(
                    season_name="2026-27",
                    start_gw=8,
                    end_gw=11,
                    output_dir="fresh-predictions",
                    manager_state_path=state_path,
                    previous_jobs=[],
                )

    def test_live_continuation_skips_before_a_manager_state_exists(self) -> None:
        with TemporaryDirectory() as tmpdir:
            result = start_live_manager_continuation(
                season_name="2026-27",
                start_gw=1,
                end_gw=4,
                output_dir="fresh-predictions",
                manager_state_path=Path(tmpdir) / "missing-state.json",
                job_root=Path(tmpdir) / "jobs",
            )

        self.assertEqual(result["action"], "skipped")
        self.assertEqual(result["reason"], "no_saved_state")

    def test_live_continuation_waits_when_upcoming_week_already_committed(self) -> None:
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "manager_state.json"
            self._write_state(state_path, last_processed_gameweek=4)
            before = state_path.read_bytes()
            with patch("fplmodel.manager_jobs.create_manager_job") as create:
                result = start_live_manager_continuation(
                    season_name="2026-27", start_gw=4, end_gw=10,
                    output_dir=tmpdir, manager_state_path=state_path,
                    job_root=Path(tmpdir) / "jobs",
                )
            self.assertEqual(result["action"], "waiting")
            self.assertEqual(result["next_gameweek"], 5)
            self.assertEqual(state_path.read_bytes(), before)
            create.assert_not_called()

    def test_live_continuation_starts_one_background_job(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "jobs"
            state_path = Path(tmpdir) / "manager_state.json"
            self._write_state(state_path, last_processed_gameweek=7)

            with patch(
                "fplmodel.manager_jobs.start_manager_job",
                side_effect=lambda job_id, **_kwargs: {
                    "job_id": job_id,
                    "status": "running",
                },
            ) as start_job:
                result = start_live_manager_continuation(
                    season_name="2026-27",
                    start_gw=8,
                    end_gw=11,
                    output_dir="fresh-predictions",
                    manager_state_path=state_path,
                    job_root=root,
                )

            jobs = list_manager_jobs(job_root=root)

        self.assertEqual(result["action"], "started")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["request"]["run_mode"], "live")
        self.assertEqual(jobs[0]["request"]["start_gw"], 8)
        start_job.assert_called_once()

    def test_live_continuation_does_not_duplicate_an_active_job(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "jobs"
            state_path = Path(tmpdir) / "manager_state.json"
            self._write_state(state_path, last_processed_gameweek=7)
            existing = create_manager_job(
                _request(
                    run_mode="live",
                    start_gw=8,
                    end_gw=11,
                    output_dir="fresh-predictions",
                    manager_state_path=str(state_path),
                ),
                job_root=root,
            )

            with patch("fplmodel.manager_jobs.start_manager_job") as start_job:
                result = start_live_manager_continuation(
                    season_name="2026-27",
                    start_gw=8,
                    end_gw=11,
                    output_dir="fresh-predictions",
                    manager_state_path=state_path,
                    job_root=root,
                )

        self.assertEqual(result["action"], "existing")
        self.assertEqual(result["job_id"], existing["job_id"])
        start_job.assert_not_called()


if __name__ == "__main__":
    unittest.main()
