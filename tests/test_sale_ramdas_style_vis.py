import numpy as np
import pandas as pd

from src.scripts.sale_ramdas_style_vis import (
    FINITE_RELAXED_KEY,
    finite_relaxed_events,
    summarize_events,
    summarize_run_means,
)


def test_pooled_summary_does_not_average_run_medians():
    raw_df = pd.DataFrame(
        {
            "run": [0, 0, 0, 1, 1, 1, 1],
            "strategy": ["EXPRESS"] * 7,
            "miscovered": [0] * 7,
            "n_calibration": [10] * 7,
            "interval_length": [1.0, np.inf, np.inf, 2.0, 3.0, 4.0, 5.0],
        }
    )

    pooled = summarize_events(raw_df).loc["EXPRESS"]
    run_averaged = summarize_run_means(raw_df).loc["EXPRESS"]

    assert pooled["infinite_fraction"] == 2 / 7
    assert pooled["median_interval_length"] == 4.0
    assert np.isinf(run_averaged["median_interval_length"])


def test_miscoverage_percentiles_use_all_available_runs():
    run_rates = {
        2: [0, 0, 0, 0],
        5: [0, 0, 0, 1],
        8: [0, 0, 1, 1],
        13: [0, 1, 1, 1],
        21: [1, 1, 1, 1],
    }
    rows = []
    for run, outcomes in run_rates.items():
        rows.extend(
            {
                "run": run,
                "strategy": "EXPRESS",
                "miscovered": outcome,
                "n_calibration": 10,
                "interval_length": 2.0,
            }
            for outcome in outcomes
        )

    pooled = summarize_events(pd.DataFrame(rows)).loc["EXPRESS"]
    per_run = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    assert pooled["miscoverage"] == 0.5
    assert pooled["miscoverage_q10"] == np.quantile(per_run, 0.10)
    assert pooled["miscoverage_q90"] == np.quantile(per_run, 0.90)


def test_finite_relaxed_subset_requires_an_actual_nonexact_addition():
    raw_df = pd.DataFrame(
        {
            "run": [0, 0, 0, 0],
            "strategy": [
                "FINITE-EXPRESS",
                "FINITE-EXPRESS",
                "FINITE-EXPRESS",
                "EXPRESS",
            ],
            "finite_express_added_nonexact": [0, 1, 9, np.nan],
        }
    )

    relaxed = finite_relaxed_events(raw_df)

    assert relaxed.index.to_list() == [1, 2]
    assert relaxed["strategy"].eq(FINITE_RELAXED_KEY).all()
