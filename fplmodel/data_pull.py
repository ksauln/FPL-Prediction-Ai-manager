from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import requests

from .config import (
    FPL_BOOTSTRAP,
    FPL_FIXTURES_ALL,
    FPL_ELEMENT_SUMMARY,
    RAW_DIR,
    CACHE_TTL_DAYS,
    PLAYER_HISTORY_SEASONS_BACK,
    PLAYER_HISTORY_FETCH_WORKERS,
)
from .utils import save_json, load_json, unix_now

logger = logging.getLogger(__name__)

PLAYER_HISTORY_CACHE_SCHEMA_VERSION = 2
LIVE_METADATA_CACHE_TTL_SECONDS = 15 * 60


def _cache_file_fresh(path: Path, ttl_seconds: int) -> bool:
    if not path.exists():
        return False
    try:
        return (unix_now() - path.stat().st_mtime) <= ttl_seconds
    except OSError:
        return False

def _safe_get_json(url: str) -> Any:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_bootstrap_static(force: bool = False) -> Dict[str, Any]:
    path = RAW_DIR / "bootstrap-static.json"
    if not force and _cache_file_fresh(path, LIVE_METADATA_CACHE_TTL_SECONDS):
        return load_json(path)
    data = _safe_get_json(FPL_BOOTSTRAP)
    save_json(path, data)
    return data

def fetch_fixtures_all(force: bool = False) -> Any:
    path = RAW_DIR / "fixtures-all.json"
    if not force and _cache_file_fresh(path, LIVE_METADATA_CACHE_TTL_SECONDS):
        return load_json(path)
    data = _safe_get_json(FPL_FIXTURES_ALL)
    save_json(path, data)
    return data

def _player_cache_fresh(path: Path, min_seasons: int) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("_history_cache_schema_version", 0)) != PLAYER_HISTORY_CACHE_SCHEMA_VERSION:
            return False
        ts = meta.get("_fetched_ts", 0)
        seasons_available = int(meta.get("_history_seasons", 1))
        if seasons_available < min_seasons:
            return False
        age_days = (unix_now() - ts) / 86400.0
        return age_days <= CACHE_TTL_DAYS
    except Exception:
        return False

def _current_season_start_year(now: datetime | None = None) -> int:
    now = now or datetime.utcnow()
    return now.year if now.month >= 7 else now.year - 1

def _format_season_code(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"

def _season_codes_to_fetch(seasons_back: int, now: datetime | None = None) -> List[str]:
    if seasons_back <= 0:
        return []
    start_year = _current_season_start_year(now)
    return [_format_season_code(start_year - offset) for offset in range(1, seasons_back + 1)]

def _annotate_history(entries: List[Dict[str, Any]] | None, season_code: str) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    if not entries:
        return annotated
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        item.setdefault("season_name", season_code)
        annotated.append(item)
    return annotated

def fetch_player_history(
    player_id: int,
    force: bool = False,
    seasons_back: int = PLAYER_HISTORY_SEASONS_BACK,
) -> Dict[str, Any]:
    """
    Player element-summary includes current season 'history' and upcoming fixtures.
    """
    path = RAW_DIR / f"player_{player_id}.json"
    # The official element-summary endpoint only exposes the current season.
    # Older seasons are loaded from the external historical dataset instead.
    required_seasons = 1
    if (not force) and _player_cache_fresh(path, required_seasons):
        return load_json(path)
    url = FPL_ELEMENT_SUMMARY.format(player_id=player_id)
    now = datetime.utcnow()
    current_season_code = _format_season_code(_current_season_start_year(now))
    data = _safe_get_json(url)
    history_all = _annotate_history(data.get("history", []), current_season_code)

    included_seasons = [current_season_code]

    history_all.sort(
        key=lambda row: (
            row.get("season_name", ""),
            row.get("kickoff_time") or "",
            row.get("round") or 0,
            row.get("fixture") or 0,
        )
    )

    data["history"] = history_all
    data["_fetched_ts"] = unix_now()
    data["_history_cache_schema_version"] = PLAYER_HISTORY_CACHE_SCHEMA_VERSION
    data["_history_season_codes"] = included_seasons
    data["_history_seasons"] = len(included_seasons)
    save_json(path, data)
    return data

def bulk_fetch_player_histories(
    player_ids: List[int],
    force: bool = False,
    sleep_s: float = 0.0,
    seasons_back: int = PLAYER_HISTORY_SEASONS_BACK,
    max_workers: int = PLAYER_HISTORY_FETCH_WORKERS,
) -> None:
    if not player_ids:
        return

    if sleep_s or max_workers <= 1:
        failures = 0
        for i, pid in enumerate(player_ids, start=1):
            try:
                fetch_player_history(pid, force=force, seasons_back=seasons_back)
            except Exception as exc:
                failures += 1
                logger.warning("Failed fetch for player %s: %s", pid, exc)
            if i % 50 == 0 or i == len(player_ids):
                logger.info(
                    "Player history fetch progress: %d/%d complete (%d failure%s).",
                    i,
                    len(player_ids),
                    failures,
                    "" if failures == 1 else "s",
                )
            if sleep_s:
                time.sleep(sleep_s)
        return

    failures = 0
    workers = max(1, min(int(max_workers), len(player_ids)))
    logger.info("Fetching player histories with %d worker thread(s).", workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_pid = {
            executor.submit(
                fetch_player_history,
                pid,
                force=force,
                seasons_back=seasons_back,
            ): pid
            for pid in player_ids
        }
        for i, future in enumerate(as_completed(future_to_pid), start=1):
            pid = future_to_pid[future]
            try:
                future.result()
            except Exception as exc:
                failures += 1
                logger.warning("Failed fetch for player %s: %s", pid, exc)
            if i % 50 == 0 or i == len(player_ids):
                logger.info(
                    "Player history fetch progress: %d/%d complete (%d failure%s).",
                    i,
                    len(player_ids),
                    failures,
                    "" if failures == 1 else "s",
                )
