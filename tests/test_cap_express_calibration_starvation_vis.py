import numpy as np
import pandas as pd
import pytest

from src.scripts.cap_express_calibration_starvation_vis import (
    summarize_event_bins,
    validate_summary,
)


def test_event_level_starvation_and_quantiles_are_pooled():
    events = pd.DataFrame(
        {
            "strategy": [
                "CAP",
                "CAP",
                "CAP",
                "CAP",
                "EXPRESS",
                "EXPRESS",
                "EXPRESS",
            ],
            "run": [0, 0, 1, 0, 0, 1, 1],
            "t": [1, 2, 3, 101, 1, 2, 3],
            "n_calibration": [0, 8, 10, 4, 8, 9, 10],
        }
    )

    summary, run_bins = summarize_event_bins(
        events,
        bin_width=100,
        starvation_threshold=9,
        horizon=200,
    )

    cap_first = summary.loc[
        summary["strategy"].eq("CAP") & summary["bin_start"].eq(0)
    ].iloc[0]
    assert cap_first["selected_events"] == 3
    assert cap_first["starved_events"] == 2
    assert cap_first["pooled_starvation_fraction"] == pytest.approx(2 / 3)
    assert cap_first["median_n_calibration"] == 8
    assert cap_first["q10_n_calibration"] == pytest.approx(1.6)
    assert cap_first["q90_n_calibration"] == pytest.approx(9.6)
    assert cap_first["contributing_runs"] == 2
    assert cap_first["run_starvation_q10"] == pytest.approx(0.1)
    assert cap_first["run_starvation_q90"] == pytest.approx(0.9)

    express_empty = summary.loc[
        summary["strategy"].eq("EXPRESS") & summary["bin_start"].eq(100)
    ].iloc[0]
    assert express_empty["selected_events"] == 0
    assert express_empty["starved_events"] == 0
    assert express_empty["contributing_runs"] == 0
    assert np.isnan(express_empty["pooled_starvation_fraction"])
    assert np.isnan(express_empty["run_starvation_q10"])

    assert not (
        run_bins["strategy"].eq("EXPRESS") & run_bins["bin_start"].eq(100)
    ).any()
    validate_summary(summary)


def test_validation_rejects_numerator_larger_than_denominator():
    summary = pd.DataFrame(
        {
            "starved_events": [2],
            "selected_events": [1],
            "pooled_starvation_fraction": [2.0],
            "run_starvation_q10": [0.0],
            "run_starvation_q90": [1.0],
        }
    )

    with pytest.raises(ValueError, match="numerator exceeds"):
        validate_summary(summary)
