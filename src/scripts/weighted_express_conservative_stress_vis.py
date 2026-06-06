from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from analyze_weighted_express_stress import analyze_result_dir


DEFAULT_OUTPUT_NAME = "weighted_express_conservative_stress_vs_positive_gap.png"
DEFAULT_SUMMARY_NAME = "weighted_express_conservative_stress_vs_positive_gap_summary.csv"
REQUIRED_SUMMARY_COLUMNS = {
    "coverage_gap",
    "positive_coverage_gap",
    "weighted_express_stress",
    "weighted_express_conservative_stress",
    "mean_conservative_stress",
}


def result_dirs(suite_dir):
    suite_dir = Path(suite_dir)
    dirs = [
        path
        for path in suite_dir.iterdir()
        if path.is_dir() and (path / "raw_selected_events.csv").exists()
    ]
    if not dirs:
        raise FileNotFoundError(f"No result directories with raw_selected_events.csv under {suite_dir}")
    return sorted(dirs)


def load_result_config(run_dir):
    for filename in ("resolved_config.json", "config.json"):
        config_path = Path(run_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    raise FileNotFoundError(f"Missing config.json or resolved_config.json under {run_dir}")


def conservative_summary_for_run(run_dir, recompute=False):
    run_dir = Path(run_dir)
    summary_path = run_dir / "weighted_express_stress_summary.csv"
    needs_recompute = recompute or not summary_path.exists()

    if not needs_recompute:
        summary = pd.read_csv(summary_path)
        needs_recompute = not REQUIRED_SUMMARY_COLUMNS.issubset(summary.columns)

    if needs_recompute:
        summary, bins = analyze_result_dir(run_dir)
        summary.to_csv(summary_path, index=False)
        bins.to_csv(run_dir / "weighted_express_stress_bins.csv", index=False)

    if len(summary) != 1:
        raise ValueError(f"Expected exactly one summary row in {summary_path}")
    return summary.iloc[0].to_dict()


def summarize_suite(suite_dir, recompute=False):
    rows = []
    for run_dir in result_dirs(suite_dir):
        config = load_result_config(run_dir)
        row = conservative_summary_for_run(run_dir, recompute=recompute)
        row["window_width"] = float(config["selection"]["window_width"])
        row["run_dir"] = run_dir.name
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("window_width").reset_index(drop=True)

    if (summary["weighted_express_conservative_stress"] < summary["weighted_express_stress"]).any():
        raise ValueError("Expected conservative stress to be at least raw stress.")
    if (summary["positive_coverage_gap"] < 0).any():
        raise ValueError("Expected positive coverage gap to be nonnegative.")

    return summary


def plot_summary(summary, output_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = range(len(summary))

    ax.plot(
        x,
        summary["positive_coverage_gap"],
        marker="o",
        linewidth=1.8,
        label="positive coverage gap",
    )
    ax.plot(
        x,
        summary["mean_conservative_stress"],
        marker="o",
        linewidth=1.8,
        label="conservative stress",
    )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("window width")
    ax.set_ylabel("value")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{width:g}" for width in summary["window_width"]])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot positive coverage gap and conservative Weighted EXPRESS stress "
            "over a window-width sweep."
        ),
    )
    parser.add_argument(
        "--suite-dir",
        type=Path,
        required=True,
        help="Suite directory containing window-width experiment result subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults under suite-dir/vis.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV path for the plotted suite summary. Defaults under suite-dir/vis.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute per-run stress summaries from raw_selected_events.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    vis_dir = suite_dir / "vis"
    output_path = args.output or vis_dir / DEFAULT_OUTPUT_NAME
    summary_csv = args.summary_csv or vis_dir / DEFAULT_SUMMARY_NAME

    summary = summarize_suite(suite_dir, recompute=args.recompute)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    plot_summary(summary, output_path)

    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")
    print(
        "Conservative stress is an empirical upper-bound-inspired proxy, "
        "not a formal bound."
    )


if __name__ == "__main__":
    main()
