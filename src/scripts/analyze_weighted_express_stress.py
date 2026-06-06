from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_N_BINS = 5


def load_config(result_dir):
    result_dir = Path(result_dir)
    for name in ("resolved_config.json", "config.json"):
        path = result_dir / name
        if path.exists():
            with path.open() as f:
                return json.load(f)
    raise FileNotFoundError(f"No resolved_config.json or config.json found under {result_dir}")


def ensure_stress_column(df):
    if "weighted_express_stress" in df.columns:
        df["weighted_express_stress"] = pd.to_numeric(
            df["weighted_express_stress"],
            errors="coerce",
        )
        return df

    required = [
        "weighted_express_weighted_mean_distance",
        "weighted_express_finite_mass",
    ]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(
            "Cannot compute weighted_express_stress; missing columns: "
            + ", ".join(missing)
        )

    finite_weighted_mean = pd.to_numeric(
        df["weighted_express_weighted_mean_distance"],
        errors="coerce",
    )
    finite_mass = pd.to_numeric(df["weighted_express_finite_mass"], errors="coerce")
    df["weighted_express_stress"] = finite_weighted_mean * finite_mass
    return df


def ensure_conservative_stress_column(df):
    """Add the upper-bound-inspired conservative stress diagnostic.

    This conservative stress score is an empirical diagnostic inspired by the
    weighted swap-instability form of nonexchangeable conformal prediction. It
    is not a formal coverage-gap bound for Weighted EXPRESS.
    """
    df = ensure_stress_column(df)
    if "weighted_express_test_mass" not in df.columns:
        raise ValueError(
            "Cannot compute weighted_express_conservative_stress; missing column: "
            "weighted_express_test_mass. Re-run experiments with Weighted EXPRESS "
            "test-mass diagnostics enabled."
        )

    df["weighted_express_test_mass"] = pd.to_numeric(
        df["weighted_express_test_mass"],
        errors="coerce",
    )
    df["weighted_express_conservative_stress"] = (
        df["weighted_express_stress"] + df["weighted_express_test_mass"]
    )

    comparable = df[
        ["weighted_express_stress", "weighted_express_conservative_stress"]
    ].dropna()
    if (
        comparable["weighted_express_conservative_stress"]
        < comparable["weighted_express_stress"]
    ).any():
        raise ValueError("Expected conservative stress to be at least raw stress.")

    return df


def metric_row(df, alpha, label=None):
    finite = df["finite"]
    miscoverage = df["miscovered"].mean()
    coverage_gap = miscoverage - alpha
    positive_coverage_gap = max(0.0, coverage_gap)
    mean_stress = df["weighted_express_stress"].mean()
    mean_conservative_stress = df["weighted_express_conservative_stress"].mean()

    row = {
        "n": int(len(df)),
        "alpha": alpha,
        "miscoverage": miscoverage,
        "coverage_gap": coverage_gap,
        "positive_coverage_gap": positive_coverage_gap,
        "weighted_express_stress": mean_stress,
        "weighted_express_conservative_stress": mean_conservative_stress,
        "mean_stress": mean_stress,
        "median_stress": df["weighted_express_stress"].median(),
        "mean_conservative_stress": mean_conservative_stress,
        "median_conservative_stress": df["weighted_express_conservative_stress"].median(),
        "infinite_fraction": 1.0 - finite.mean(),
        "median_interval_length": df["interval_length_numeric"].median(),
        "median_finite_interval_length": df.loc[finite, "interval_length_numeric"].median(),
        "finite_only_miscoverage": df.loc[finite, "miscovered"].mean(),
    }
    if label is not None:
        row = {"stress_bin": label, **row}
    return row


def analyze_result_dir(result_dir, n_bins=DEFAULT_N_BINS):
    result_dir = Path(result_dir)
    raw_path = result_dir / "raw_selected_events.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw_selected_events.csv under {result_dir}")

    config = load_config(result_dir)
    alpha = float(config["conformal"]["alpha"])

    raw = pd.read_csv(raw_path)
    df = raw[raw["strategy"] == "WEIGHTED-EXPRESS"].copy()
    if df.empty:
        raise ValueError(f"No WEIGHTED-EXPRESS rows found in {raw_path}")

    df["miscovered"] = pd.to_numeric(df["miscovered"], errors="coerce")
    df["interval_length_numeric"] = pd.to_numeric(df["interval_length"], errors="coerce")
    df["finite"] = np.isfinite(df["interval_length_numeric"])
    df = ensure_conservative_stress_column(df)
    if (df["weighted_express_conservative_stress"] < df["weighted_express_stress"]).any():
        raise ValueError("Expected conservative stress to be at least raw stress.")

    # This stress score is an empirical analogue of the weighted
    # nonexchangeability term. It is not a formal coverage-gap bound for
    # Weighted EXPRESS.
    summary = pd.DataFrame([metric_row(df, alpha)])
    if (summary["positive_coverage_gap"] < 0).any():
        raise ValueError("Expected positive coverage gap to be nonnegative.")

    bin_source = df["weighted_express_stress"]
    try:
        df["stress_bin"] = pd.qcut(bin_source, q=n_bins, duplicates="drop")
    except ValueError:
        df["stress_bin"] = pd.Series(["all"] * len(df), index=df.index)

    rows = []
    for stress_bin, group in df.groupby("stress_bin", observed=True, sort=True):
        row = metric_row(group, alpha, label=str(stress_bin))
        row["bin_min_stress"] = group["weighted_express_stress"].min()
        row["bin_max_stress"] = group["weighted_express_stress"].max()
        rows.append(row)
    bins = pd.DataFrame(rows)

    return summary, bins


def parse_args():
    parser = ArgumentParser(
        description="Analyze Weighted EXPRESS stress versus empirical miscoverage.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Result directory containing raw_selected_events.csv and resolved_config.json.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_N_BINS,
        help="Number of stress quantile bins.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional output path for summary CSV.",
    )
    parser.add_argument(
        "--bins-output",
        type=Path,
        default=None,
        help="Optional output path for stress-bin CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    summary_output = args.summary_output or result_dir / "weighted_express_stress_summary.csv"
    bins_output = args.bins_output or result_dir / "weighted_express_stress_bins.csv"

    summary, bins = analyze_result_dir(result_dir, n_bins=args.bins)
    summary.to_csv(summary_output, index=False)
    bins.to_csv(bins_output, index=False)

    print(f"Wrote summary to {summary_output}")
    print(f"Wrote stress bins to {bins_output}")


if __name__ == "__main__":
    main()
