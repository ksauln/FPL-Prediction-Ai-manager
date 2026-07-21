"""Persistent background jobs for the Streamlit AI Manager page."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import traceback
from typing import Optional
from uuid import uuid4

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

from .config import OUTPUTS_DIR, PROJECT_ROOT
from .season_manager import (
    ManagerState,
    SeasonManagerConfig,
    SeasonRules,
    load_manager_state,
    load_prediction_files,
    run_repeated_season_simulations,
    save_manager_state,
    save_season_simulation_artifact,
)


MANAGER_JOB_ROOT = OUTPUTS_DIR / "ai_manager_jobs"
VALID_JOB_STATUSES = {"queued", "running", "completed", "failed", "canceled"}
VALID_SIMULATION_MODES = {
    "fixed_policy",
    "periodic_reoptimization",
    "full_reoptimization",
}
VALID_RUN_MODES = {"planning", "live"}
SEASON_PATTERN = re.compile(r"^\d{4}-\d{2}$")
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_STATUS_THREAD_LOCK = threading.RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _job_directory(job_id: str, job_root: Path) -> Path:
    if not JOB_ID_PATTERN.fullmatch(str(job_id)):
        raise ValueError("Invalid manager job id")
    return Path(job_root) / str(job_id)


def _coerce_int(payload: dict[str, object], name: str, *, minimum: int) -> int:
    try:
        value = int(payload[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _normalise_dataclass_options(
    raw: object,
    *,
    option_type: type,
    excluded: set[str],
) -> dict[str, object]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{option_type.__name__} options must be an object")
    allowed = {item.name for item in fields(option_type)} - excluded
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {option_type.__name__} options: {sorted(unknown)}"
        )
    return dict(raw)


def validate_job_request(request: dict[str, object]) -> dict[str, object]:
    """Validate and normalize a JSON-safe manager job request."""

    if not isinstance(request, dict):
        raise ValueError("Manager job request must be an object")
    season_name = str(request.get("season_name", "")).strip()
    if not SEASON_PATTERN.fullmatch(season_name):
        raise ValueError("season_name must use YYYY-YY format")

    run_mode = str(request.get("run_mode", "planning"))
    if run_mode not in VALID_RUN_MODES:
        raise ValueError(f"run_mode must be one of {sorted(VALID_RUN_MODES)}")
    simulation_mode = str(request.get("simulation_mode", "periodic_reoptimization"))
    if simulation_mode not in VALID_SIMULATION_MODES:
        raise ValueError(
            f"simulation_mode must be one of {sorted(VALID_SIMULATION_MODES)}"
        )

    start_gw = _coerce_int(request, "start_gw", minimum=1)
    end_gw = _coerce_int(request, "end_gw", minimum=1)
    if start_gw > 38:
        raise ValueError("start_gw must be <= 38")
    if end_gw > 38 or end_gw < start_gw:
        raise ValueError("end_gw must be between start_gw and 38")
    simulations = _coerce_int(request, "simulations", minimum=1)
    policy_refresh_interval = _coerce_int(
        request, "policy_refresh_interval", minimum=1
    )
    try:
        random_seed = int(request.get("random_seed", 90))
    except (TypeError, ValueError) as exc:
        raise ValueError("random_seed must be an integer") from exc

    manager_config = _normalise_dataclass_options(
        request.get("manager_config", {}),
        option_type=SeasonManagerConfig,
        excluded={"rules", "formations"},
    )
    rules = _normalise_dataclass_options(
        request.get("rules", {}),
        option_type=SeasonRules,
        excluded={"chips_by_half"},
    )
    if "free_transfer_topups" in rules:
        topups = rules["free_transfer_topups"]
        if not isinstance(topups, dict):
            raise ValueError("free_transfer_topups must be an object")
        rules["free_transfer_topups"] = {
            int(gameweek): int(count) for gameweek, count in topups.items()
        }

    output_dir = str(request.get("output_dir", OUTPUTS_DIR))
    state_path = str(
        request.get(
            "manager_state_path",
            Path(output_dir) / f"manager_state_{season_name}.json",
        )
    )
    return {
        "season_name": season_name,
        "run_mode": run_mode,
        "start_gw": start_gw,
        "end_gw": end_gw,
        "gameweeks": list(range(start_gw, end_gw + 1)),
        "simulations": simulations,
        "simulation_mode": simulation_mode,
        "policy_refresh_interval": policy_refresh_interval,
        "random_seed": random_seed,
        "manager_config": manager_config,
        "rules": rules,
        "output_dir": output_dir,
        "manager_state_path": state_path,
    }


@contextmanager
def _status_lock(job_dir: Path):
    lock_path = job_dir / ".status.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _STATUS_THREAD_LOCK, lock_path.open("a", encoding="utf-8") as lock_handle:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _update_status(
    job_dir: Path,
    *,
    allowed_statuses: Optional[set[str]] = None,
    **updates: object,
) -> dict[str, object]:
    status_path = job_dir / "status.json"
    with _status_lock(job_dir):
        status = _read_json(status_path) if status_path.exists() else {}
        if allowed_statuses is not None and status.get("status") not in allowed_statuses:
            return status
        if status.get("status") == "canceled" and updates.get("status") != "canceled":
            return status
        status.update(updates)
        status["updated_at"] = _utc_now()
        _atomic_write_json(status_path, status)
        return status


def _process_exists(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def create_manager_job(
    request: dict[str, object],
    *,
    job_root: Path = MANAGER_JOB_ROOT,
) -> dict[str, object]:
    """Create a queued manager job and persist its validated request."""

    validated = validate_job_request(request)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = f"{timestamp}-{uuid4().hex[:8]}"
    job_dir = _job_directory(job_id, Path(job_root))
    job_dir.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(job_dir / "request.json", validated)
    status = {
        "job_id": job_id,
        "status": "queued",
        "phase": "queued",
        "progress": 0.0,
        "completed_simulations": 0,
        "total_simulations": validated["simulations"],
        "created_at": _utc_now(),
    }
    _atomic_write_json(job_dir / "status.json", status)
    return status


def load_manager_job(
    job_id: str,
    *,
    job_root: Path = MANAGER_JOB_ROOT,
) -> dict[str, object]:
    job_dir = _job_directory(job_id, Path(job_root))
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Manager job {job_id} was not found")
    status = _read_json(job_dir / "status.json")
    if status.get("status") not in VALID_JOB_STATUSES:
        raise ValueError(f"Manager job {job_id} has an invalid status")
    pid = status.get("pid")
    if status.get("status") == "running" and pid is not None and not _process_exists(int(pid)):
        status = _update_status(
            job_dir,
            status="failed",
            phase="failed",
            error="The simulation worker exited before completing the job.",
            failed_at=_utc_now(),
        )
    status["request"] = _read_json(job_dir / "request.json")
    return status


def list_manager_jobs(
    *, job_root: Path = MANAGER_JOB_ROOT
) -> list[dict[str, object]]:
    root = Path(job_root)
    if not root.exists():
        return []
    jobs: list[dict[str, object]] = []
    for path in root.iterdir():
        if not path.is_dir() or not JOB_ID_PATTERN.fullmatch(path.name):
            continue
        try:
            jobs.append(load_manager_job(path.name, job_root=root))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            continue
    return sorted(jobs, key=lambda item: str(item.get("created_at", "")), reverse=True)


def load_manager_job_result(
    job_id: str,
    *,
    job_root: Path = MANAGER_JOB_ROOT,
) -> dict[str, object]:
    """Load the result artifact for a completed manager job."""

    job = load_manager_job(job_id, job_root=job_root)
    if job["status"] != "completed":
        raise RuntimeError("Manager job result is available only after completion")
    result_path = _job_directory(job_id, Path(job_root)) / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Manager job {job_id} has no result artifact")
    return _read_json(result_path)


def start_manager_job(
    job_id: str,
    *,
    job_root: Path = MANAGER_JOB_ROOT,
    project_root: Path = PROJECT_ROOT,
    python_executable: str = sys.executable,
) -> dict[str, object]:
    """Start a queued manager job in a detached worker process."""

    job = load_manager_job(job_id, job_root=job_root)
    if job["status"] != "queued":
        raise RuntimeError(f"Manager job {job_id} is already {job['status']}")
    job_dir = _job_directory(job_id, Path(job_root)).resolve()
    log_path = job_dir / "worker.log"
    command = [
        str(python_executable),
        "-m",
        "fplmodel.manager_jobs",
        "--job-dir",
        str(job_dir),
    ]
    _update_status(
        job_dir,
        status="running",
        phase="starting",
        started_at=_utc_now(),
        log_path=str(log_path),
    )
    log_handle = log_path.open("a", encoding="utf-8")
    try:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(Path(project_root).resolve()),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            _update_status(
                job_dir,
                status="failed",
                phase="failed",
                error=f"{type(exc).__name__}: {exc}",
                failed_at=_utc_now(),
            )
            raise
    finally:
        log_handle.close()
    return _update_status(job_dir, pid=int(process.pid))


def cancel_manager_job(
    job_id: str,
    *,
    job_root: Path = MANAGER_JOB_ROOT,
) -> dict[str, object]:
    """Cancel a queued or running manager job."""

    job = load_manager_job(job_id, job_root=job_root)
    if job["status"] not in {"queued", "running"}:
        raise RuntimeError(f"Manager job {job_id} cannot be canceled from {job['status']}")
    job_dir = _job_directory(job_id, Path(job_root))
    pid = job.get("pid")
    if pid is not None:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    result_path = job_dir / "result.json"
    recommended_state_path = job_dir / "recommended_state.json"
    if result_path.is_file() and recommended_state_path.is_file():
        return _update_status(
            job_dir,
            allowed_statuses={"running", "completed"},
            status="completed",
            phase="complete",
            progress=1.0,
            completed_simulations=int(job["request"]["simulations"]),
            total_simulations=int(job["request"]["simulations"]),
            eta_seconds=0.0,
            completed_at=_utc_now(),
            result_path=str(result_path),
            recommended_state_path=str(recommended_state_path),
        )
    return _update_status(
        job_dir,
        allowed_statuses={"queued", "running"},
        status="canceled",
        phase="canceled",
        canceled_at=_utc_now(),
    )


def commit_recommended_state(
    job_id: str,
    destination: Path,
    *,
    job_root: Path = MANAGER_JOB_ROOT,
) -> dict[str, object]:
    """Commit a completed job's first recommended decision as live state."""

    job = load_manager_job(job_id, job_root=job_root)
    if job["status"] != "completed":
        raise RuntimeError("Only a completed manager job can commit state")
    candidate_path = _job_directory(job_id, Path(job_root)) / "recommended_state.json"
    if not candidate_path.is_file():
        raise FileNotFoundError("The completed job has no recommended state artifact")
    state = load_manager_state(candidate_path)
    request = dict(job["request"])
    expected_season = str(request["season_name"])
    expected_job_gameweek = int(request["start_gw"])
    if state.season_name != expected_season:
        raise ValueError(
            f"Recommended state is for {state.season_name}, not {expected_season}"
        )
    if state.last_processed_gameweek != expected_job_gameweek:
        raise ValueError(
            f"Recommended state should end at GW{expected_job_gameweek}, but ends at "
            f"GW{state.last_processed_gameweek}"
        )
    destination = Path(destination)
    if destination.exists():
        current_state = load_manager_state(destination)
        if current_state.season_name not in {None, expected_season}:
            raise ValueError(
                f"Existing manager state is for {current_state.season_name}, not "
                f"{expected_season}"
            )
        expected_next = current_state.last_processed_gameweek + 1
        if state.last_processed_gameweek != expected_next:
            raise ValueError(
                f"Existing manager state expects GW{expected_next}; refusing to "
                f"commit GW{state.last_processed_gameweek}"
            )
    elif state.last_processed_gameweek != 1:
        raise ValueError("A new live manager state must start with GW1")
    destination = save_manager_state(state, Path(destination))
    return _update_status(
        _job_directory(job_id, Path(job_root)),
        committed_state_path=str(destination),
        state_committed_at=_utc_now(),
    )


def _build_manager_config(request: dict[str, object]) -> SeasonManagerConfig:
    rules_options = dict(request.get("rules", {}))
    if "free_transfer_topups" in rules_options:
        rules_options["free_transfer_topups"] = {
            int(gameweek): int(count)
            for gameweek, count in dict(
                rules_options["free_transfer_topups"]
            ).items()
        }
    rules = SeasonRules(**rules_options)
    return SeasonManagerConfig(
        rules=rules,
        **dict(request.get("manager_config", {})),
    )


def run_manager_job(job_dir: Path) -> dict[str, object]:
    """Execute one persisted job synchronously inside the worker process."""

    job_dir = Path(job_dir).resolve()
    request = validate_job_request(_read_json(job_dir / "request.json"))
    _atomic_write_json(job_dir / "request.json", request)
    started_at = datetime.now(timezone.utc)
    _update_status(
        job_dir,
        status="running",
        phase="loading_predictions",
        pid=os.getpid(),
        started_at=started_at.isoformat(),
    )
    output_dir = Path(str(request["output_dir"]))
    predictions = load_prediction_files(
        output_dir,
        start_gw=int(request["start_gw"]),
        end_gw=int(request["end_gw"]),
        expected_season_name=str(request["season_name"]),
    )
    expected_gameweeks = [int(gw) for gw in request["gameweeks"]]
    missing = [gameweek for gameweek in expected_gameweeks if gameweek not in predictions]
    if missing:
        raise FileNotFoundError(f"Missing prediction files for gameweeks: {missing}")

    state_path = Path(str(request["manager_state_path"]))
    initial_state: Optional[ManagerState] = None
    if request["run_mode"] == "live":
        if state_path.exists():
            initial_state = load_manager_state(state_path)
        elif int(request["start_gw"]) != 1:
            raise FileNotFoundError(
                f"Live mode needs {state_path.name} before starting after GW1"
            )

    def progress_callback(event: dict[str, object]) -> None:
        completed = int(event.get("completed_simulations", 0))
        total = int(event.get("total_simulations", request["simulations"]))
        _update_status(
            job_dir,
            status="running",
            phase=str(event.get("phase", "running")),
            progress=min(1.0, max(0.0, float(event.get("progress", 0.0)))),
            completed_simulations=completed,
            total_simulations=total,
            current_gameweek=event.get("current_gameweek"),
            policy_block=event.get("policy_block"),
            elapsed_seconds=event.get("elapsed_seconds"),
            eta_seconds=event.get("eta_seconds"),
        )

    result = run_repeated_season_simulations(
        predictions,
        gameweeks=expected_gameweeks,
        simulations=int(request["simulations"]),
        config=_build_manager_config(request),
        random_seed=int(request["random_seed"]),
        simulation_mode=str(request["simulation_mode"]),
        policy_refresh_interval=int(request["policy_refresh_interval"]),
        initial_state=initial_state,
        progress_callback=progress_callback,
    )
    _update_status(
        job_dir,
        status="running",
        phase="saving_results",
        progress=1.0,
        eta_seconds=None,
    )
    result_path = save_season_simulation_artifact(result, job_dir / "result.json")
    recommended_policy = result.get("recommended_policy", result.get("best_run", {}))
    decisions = list(recommended_policy.get("decisions", []))
    recommended_state_path: Optional[Path] = None
    if decisions and "manager_state_after" in decisions[0]:
        recommended_state_path = save_manager_state(
            ManagerState.from_dict(decisions[0]["manager_state_after"]),
            job_dir / "recommended_state.json",
        )

    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    return _update_status(
        job_dir,
        status="completed",
        phase="complete",
        progress=1.0,
        completed_simulations=int(request["simulations"]),
        total_simulations=int(request["simulations"]),
        elapsed_seconds=elapsed,
        eta_seconds=0.0,
        completed_at=_utc_now(),
        result_path=str(result_path),
        recommended_state_path=(
            str(recommended_state_path) if recommended_state_path else None
        ),
        summary=result.get("summary", {}),
    )


def _run_worker(job_dir: Path) -> int:
    try:
        run_manager_job(job_dir)
    except BaseException as exc:
        trace_path = Path(job_dir) / "traceback.log"
        trace_path.write_text(traceback.format_exc(), encoding="utf-8")
        try:
            current = _read_json(Path(job_dir) / "status.json")
            if current.get("status") != "canceled":
                _update_status(
                    Path(job_dir),
                    status="failed",
                    phase="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    traceback_path=str(trace_path),
                    failed_at=_utc_now(),
                )
        except Exception:
            pass
        return 1
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a persisted AI Manager job")
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return _run_worker(args.job_dir)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MANAGER_JOB_ROOT",
    "cancel_manager_job",
    "commit_recommended_state",
    "create_manager_job",
    "list_manager_jobs",
    "load_manager_job",
    "load_manager_job_result",
    "run_manager_job",
    "start_manager_job",
    "validate_job_request",
]
