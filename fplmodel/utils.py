import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .config import RAW_DIR

def save_json(path: Path, obj: Any, indent: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_current_and_last_finished_gw(events_df: pd.DataFrame) -> Tuple[int, int]:
    """
    From events (bootstrap 'events'), infer current and last finished GW.
    Returns (next_gw, last_finished_gw).
    """
    id_col = "event_id" if "event_id" in events_df.columns else "id"
    if id_col not in events_df.columns:
        raise KeyError("Neither 'event_id' nor 'id' found in events dataframe.")

    # `finished` can become true while Opta review is still provisional. Since
    # 2026/27 final scoring locks at 09:00 UK time after the final match, require
    # `data_checked` as well when the API exposes it.
    next_rows = events_df[events_df["is_next"] == True]
    if len(next_rows):
        next_gw = int(next_rows.iloc[0][id_col])
    else:
        # If season completed, pick last+1 to indicate no next GW
        next_gw = int(events_df[id_col].max()) + 1

    finished_mask = events_df["finished"].fillna(False).astype(bool)
    if "data_checked" in events_df.columns:
        finished_mask &= events_df["data_checked"].fillna(False).astype(bool)
    finished = events_df[finished_mask]
    last_finished_gw = int(finished[id_col].max()) if len(finished) else 0
    return next_gw, last_finished_gw

def unix_now():
    return int(time.time())
