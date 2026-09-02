import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import pipeline
from src.conformal import Conformal
from src.scripts.selection_bounds_timeseries_vis import reconstruct_bounds_for_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("n_selected", [0, 5, 20])
def test_selection_window_stays_fixed_above_former_jump_threshold(n_selected):
    conformal = Conformal(tau_0=4, tau_1=5.0, window_width=0.5)
    conformal.s_past = np.ones(n_selected, dtype=int)

    expected_lower = 5.0 + n_selected / 4
    lower, upper = conformal.selection_bounds(n_selected)

    assert lower == expected_lower
    assert upper == expected_lower + 0.5
    assert conformal.selection_bound(n_selected) == upper
    assert np.isfinite(upper)


def test_tau_tail_is_not_part_of_the_public_selection_api():
    assert "tau_tail" not in inspect.signature(Conformal).parameters
    assert "tau_tail" not in inspect.signature(Conformal.selection_bound).parameters
    assert "tau_tail" not in inspect.signature(Conformal.selection_bounds).parameters
    assert not hasattr(Conformal(), "tau_tail")

    with pytest.raises(TypeError):
        Conformal(tau_tail=6.25)


def test_pipeline_runs_without_tau_tail(monkeypatch):
    data = pd.DataFrame(
        {
            "Label": [0.2, 0.4],
            "muhat_1": [0.0, 0.0],
            "muhat_2": [6.5, 6.5],
        }
    )
    captured = {}

    monkeypatch.setattr(
        pipeline,
        "sample_data_frame",
        lambda n, seed, data_path: data.copy(),
    )

    def capture_results(results, raw_rows, selected_rows, n_runs, **kwargs):
        captured["selected_rows"] = selected_rows
        return Path("unused")

    monkeypatch.setattr(pipeline, "dump_experiment_results", capture_results)

    config = pipeline.deep_merge(
        pipeline.DEFAULT_CONFIG,
        {
            "n_runs": 1,
            "data": {"n_off": 1, "n_on": 1, "path": "unused.csv"},
            "selection": {
                "tau_0": 1,
                "tau_1": 6.25,
                "window_width": 0.5,
            },
            "conformal": {
                "alpha": 0.1,
                "randomized_calibration": False,
            },
            "strategies": ["FULL"],
        },
    )

    assert "tau_tail" not in config["selection"]
    pipeline.run_experiment(config)

    assert len(captured["selected_rows"]) == 1
    selected = captured["selected_rows"][0]
    assert selected["selection_lower_bound"] == 6.25
    assert selected["selection_upper_bound"] == 6.75


def test_visualization_reconstruction_uses_a_fixed_width_window():
    selected_df = pd.DataFrame(
        {
            "run": np.zeros(5, dtype=int),
            "t": np.arange(5),
            "score_t": np.full(5, 6.5),
        }
    )

    reconstructed = reconstruct_bounds_for_run(
        selected_df,
        run=0,
        n_on=8,
        tau_0=4,
        tau_1=5.0,
        window_width=0.5,
    )

    assert np.all(np.isfinite(reconstructed["selection_upper_bound"]))
    assert np.allclose(
        reconstructed["selection_upper_bound"]
        - reconstructed["selection_lower_bound"],
        0.5,
    )
    former_boundary = reconstructed.loc[reconstructed["t"] == 5].iloc[0]
    assert former_boundary["selection_lower_bound"] == 6.25
    assert former_boundary["selection_upper_bound"] == 6.75


def test_active_configs_do_not_define_tau_tail():
    for config_path in sorted((PROJECT_ROOT / "configs").rglob("*.json")):
        with config_path.open() as config_file:
            config = json.load(config_file)
        assert "tau_tail" not in config.get("selection", {}), config_path
