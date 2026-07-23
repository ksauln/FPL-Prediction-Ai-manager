"""Parse and validate the rule configuration published by the live FPL API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import (
    BUDGET_MILLIONS,
    MAX_PER_TEAM,
    SQUAD_POSITION_LIMITS,
)
from .season_manager import SeasonRules


CHIP_NAME_MAP = {
    "wildcard": "wildcard",
    "freehit": "free_hit",
    "bboost": "bench_boost",
    "3xc": "triple_captain",
}
POSITION_NAME_MAP = {"GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

# These are the scoring values published in game_config for 2026/27. The model
# consumes official total_points rather than recomputing points, but validating
# the scoring surface prevents a material rule change from silently reaching a
# model trained under different assumptions.
EXPECTED_SCORING_2026_27: dict[str, object] = {
    "long_play": 2,
    "short_play": 1,
    "goals_conceded": {"DEF": -1, "FWD": 0, "GKP": -1, "MID": 0},
    "saves": 1,
    "goals_scored": {"DEF": 6, "FWD": 4, "GKP": 10, "MID": 5},
    "assists": 3,
    "clean_sheets": {"DEF": 4, "FWD": 0, "GKP": 4, "MID": 1},
    "penalties_saved": 5,
    "penalties_missed": -2,
    "yellow_cards": -1,
    "red_cards": -3,
    "own_goals": -2,
    "bonus": 1,
    "defensive_contribution": {"DEF": 2, "FWD": 2, "GKP": 0, "MID": 2},
}


@dataclass(frozen=True)
class LiveFPLConfiguration:
    """FPL constraints derived from the current bootstrap payload."""

    budget_m: float
    squad_size: int
    starting_size: int
    max_per_team: int
    max_free_transfers: int
    transfers_cap: int
    first_half_end_gw: int
    position_limits: dict[str, int]
    position_play_limits: dict[str, tuple[int, int]]
    chips_by_half: dict[str, dict[str, int]]
    chip_windows: dict[str, tuple[tuple[int, int], ...]]
    scoring: Mapping[str, object]


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Live FPL configuration is missing {label}.")
    return value


def _as_int(value: object, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Live FPL configuration has an invalid {label}.") from exc


def parse_live_fpl_configuration(bootstrap: Mapping[str, object]) -> LiveFPLConfiguration:
    """Extract the live constraints used by the prediction and manager layers."""

    game_config = _require_mapping(bootstrap.get("game_config"), "game_config")
    rules = _require_mapping(game_config.get("rules"), "game_config.rules")
    scoring = _require_mapping(game_config.get("scoring"), "game_config.scoring")

    currency_multiplier = _as_int(
        rules.get("ui_currency_multiplier"),
        "currency multiplier",
    )
    if currency_multiplier <= 0:
        raise RuntimeError("Live FPL configuration has a non-positive currency multiplier.")
    budget_m = _as_int(rules.get("squad_total_spend"), "squad budget") / float(
        currency_multiplier
    )

    element_types = bootstrap.get("element_types")
    if not isinstance(element_types, list) or not element_types:
        raise RuntimeError("Live FPL configuration is missing element_types.")
    position_limits: dict[str, int] = {}
    position_play_limits: dict[str, tuple[int, int]] = {}
    for raw_position in element_types:
        position = _require_mapping(raw_position, "element type")
        api_name = str(position.get("singular_name_short", ""))
        label = POSITION_NAME_MAP.get(api_name)
        if label is None:
            raise RuntimeError(f"Live FPL configuration has an unknown position {api_name!r}.")
        position_limits[label] = _as_int(position.get("squad_select"), f"{label} squad limit")
        position_play_limits[label] = (
            _as_int(position.get("squad_min_play"), f"{label} minimum starters"),
            _as_int(position.get("squad_max_play"), f"{label} maximum starters"),
        )

    raw_chips = bootstrap.get("chips")
    if not isinstance(raw_chips, list) or not raw_chips:
        raise RuntimeError("Live FPL configuration is missing chips.")
    chip_windows_mutable: dict[str, list[tuple[int, int]]] = {
        chip_name: [] for chip_name in CHIP_NAME_MAP.values()
    }
    for raw_chip in raw_chips:
        chip = _require_mapping(raw_chip, "chip entry")
        api_name = str(chip.get("name", ""))
        name = CHIP_NAME_MAP.get(api_name)
        if name is None:
            continue
        start = _as_int(chip.get("start_event"), f"{name} start event")
        stop = _as_int(chip.get("stop_event"), f"{name} stop event")
        count = _as_int(chip.get("number", 1), f"{name} count")
        chip_windows_mutable[name].extend([(start, stop)] * count)

    chip_windows = {
        name: tuple(sorted(windows))
        for name, windows in chip_windows_mutable.items()
    }
    incomplete = [name for name, windows in chip_windows.items() if len(windows) != 2]
    if incomplete:
        raise RuntimeError(
            "Live FPL chip configuration is incomplete for: "
            + ", ".join(sorted(incomplete))
        )

    first_stops = {windows[0][1] for windows in chip_windows.values()}
    if len(first_stops) != 1:
        raise RuntimeError("Live FPL chips do not share one first-half end gameweek.")
    first_half_end_gw = first_stops.pop()
    chips_by_half = {
        "first": {
            name: sum(1 for start, stop in windows if stop <= first_half_end_gw)
            for name, windows in chip_windows.items()
        },
        "second": {
            name: sum(1 for start, stop in windows if start > first_half_end_gw)
            for name, windows in chip_windows.items()
        },
    }

    return LiveFPLConfiguration(
        budget_m=budget_m,
        squad_size=_as_int(rules.get("squad_squadsize"), "squad size"),
        starting_size=_as_int(rules.get("squad_squadplay"), "starting squad size"),
        max_per_team=_as_int(rules.get("squad_team_limit"), "club limit"),
        max_free_transfers=_as_int(
            rules.get("max_extra_free_transfers"),
            "maximum extra free transfers",
        )
        + 1,
        transfers_cap=_as_int(rules.get("transfers_cap"), "transfer cap"),
        first_half_end_gw=first_half_end_gw,
        position_limits=position_limits,
        position_play_limits=position_play_limits,
        chips_by_half=chips_by_half,
        chip_windows=chip_windows,
        scoring=scoring,
    )


def _validate_scoring(actual: Mapping[str, object]) -> None:
    for name, expected in EXPECTED_SCORING_2026_27.items():
        if name not in actual:
            raise RuntimeError(f"Live FPL scoring is missing {name}.")
        if actual[name] != expected:
            raise RuntimeError(
                f"Live FPL scoring changed for {name}: "
                f"expected {expected!r}, received {actual[name]!r}."
            )


def validate_live_fpl_configuration(
    bootstrap: Mapping[str, object],
    *,
    expected_rules: SeasonRules | None = None,
) -> LiveFPLConfiguration:
    """Fail before model training when live FPL constraints drift unexpectedly."""

    live = parse_live_fpl_configuration(bootstrap)
    expected = expected_rules or SeasonRules()
    errors: list[str] = []

    if live.budget_m != float(BUDGET_MILLIONS) or live.budget_m != float(expected.budget_m):
        errors.append(
            f"budget is £{live.budget_m:.1f}m, expected £{float(expected.budget_m):.1f}m"
        )
    if live.squad_size != sum(SQUAD_POSITION_LIMITS.values()):
        errors.append(
            f"squad size is {live.squad_size}, expected {sum(SQUAD_POSITION_LIMITS.values())}"
        )
    if live.starting_size != 11:
        errors.append(f"starting XI size is {live.starting_size}, expected 11")
    if live.max_per_team != MAX_PER_TEAM:
        errors.append(f"club limit is {live.max_per_team}, expected {MAX_PER_TEAM}")
    if live.max_free_transfers != int(expected.max_free_transfers):
        errors.append(
            f"max free transfers is {live.max_free_transfers}, "
            f"expected {int(expected.max_free_transfers)}"
        )
    if live.transfers_cap != 20:
        errors.append(f"per-gameweek transfer cap is {live.transfers_cap}, expected 20")
    if live.first_half_end_gw != int(expected.first_half_end_gw):
        errors.append(
            f"first-half chip deadline is GW{live.first_half_end_gw}, "
            f"expected GW{int(expected.first_half_end_gw)}"
        )
    if live.position_limits != SQUAD_POSITION_LIMITS:
        errors.append(
            f"position limits are {live.position_limits}, expected {SQUAD_POSITION_LIMITS}"
        )
    expected_play_limits = {
        "GK": (1, 1),
        "DEF": (3, 5),
        "MID": (2, 5),
        "FWD": (1, 3),
    }
    if live.position_play_limits != expected_play_limits:
        errors.append(
            f"starting position limits are {live.position_play_limits}, "
            f"expected {expected_play_limits}"
        )
    if live.chips_by_half != expected.chips_by_half:
        errors.append(
            f"chip counts are {live.chips_by_half}, expected {expected.chips_by_half}"
        )
    if live.chip_windows["free_hit"][0][0] != 2:
        errors.append("first Free Hit is available in GW1, but the manager forbids that")

    if errors:
        raise RuntimeError("Live FPL rule validation failed: " + "; ".join(errors) + ".")
    _validate_scoring(live.scoring)
    return live


def season_rules_from_bootstrap(bootstrap: Mapping[str, object]) -> SeasonRules:
    """Build manager rules from a validated live bootstrap payload."""

    live = validate_live_fpl_configuration(bootstrap)
    return SeasonRules(
        budget_m=live.budget_m,
        free_transfer_per_gameweek=1,
        max_free_transfers=live.max_free_transfers,
        transfer_hit_cost=4.0,
        first_half_end_gw=live.first_half_end_gw,
        free_transfer_topups={},
        chips_by_half=live.chips_by_half,
    )


__all__ = [
    "EXPECTED_SCORING_2026_27",
    "LiveFPLConfiguration",
    "parse_live_fpl_configuration",
    "season_rules_from_bootstrap",
    "validate_live_fpl_configuration",
]
