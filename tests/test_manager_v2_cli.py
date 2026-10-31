import json

from fplmodel.season_manager import save_manager_state
from scripts import manager_v2
from test_manager_backtest import vintage
from test_manager_v2 import frames, state


def test_backtest_command_writes_a_report_without_changing_source_state(tmp_path, monkeypatch):
    from dataclasses import asdict
    monkeypatch.setattr(manager_v2, "OUTPUTS_DIR", tmp_path / "outputs")
    initial = tmp_path / "state.json"
    save_manager_state(state(), initial)
    before = initial.read_bytes()
    forecast_root = tmp_path / "vintages"
    for gw, frame in frames((2, 3)).items():
        directory = forecast_root / str(gw)
        directory.mkdir(parents=True)
        frame.to_csv(directory / f"predictions_gw{gw}.csv", index=False)
        forecast = vintage(gw, {})
        manifest = asdict(forecast)
        manifest["predictions"] = {str(gw): f"predictions_gw{gw}.csv"}
        (directory / "vintage.json").write_text(json.dumps(manifest))
    actuals = tmp_path / "actuals.json"
    actuals.write_text(json.dumps({str(gw): {str(pid): [True, 4] for pid in range(1, 21)} for gw in (2, 3)}))
    assert manager_v2.main(["backtest", "--vintages", str(forecast_root),
        "--actuals", str(actuals), "--state", str(initial),
        "--simulations", "4", "--horizon", "1", "--beam-width", "1", "--max-transfers", "0"]) == 0
    report = next((tmp_path / "outputs" / "manager_v2" / "reports").glob("backtest_*.json"))
    payload = json.loads(report.read_text())
    assert payload["deadline_safe"]
    assert payload["managers"]["hold"]["net_actual_points"] == 96.
    assert initial.read_bytes() == before
