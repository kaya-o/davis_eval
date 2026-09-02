from argparse import ArgumentParser
import os
from pathlib import Path
import shlex

import numpy as np
import pandas as pd

try:
    from src.scripts.sale_ramdas_style_vis import (
        label_range_from_config,
        load_result_config,
        plot_sale_ramdas_style,
        summarize_events,
    )
except ModuleNotFoundError:
    from sale_ramdas_style_vis import (
        label_range_from_config,
        load_result_config,
        plot_sale_ramdas_style,
        summarize_events,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "suite_20260825_010303_adaptive_express"
    / "20260825_013010_adaptive_weighted_express"
)
STRATEGY = "ADAPTIVE-WEIGHTED-EXPRESS"
RAW_COLUMNS = (
    "run",
    "t",
    "strategy",
    "miscovered",
    "n_calibration",
    "interval_length",
)


def load_adaptive_events(result_dir, config):
    raw_path = Path(result_dir) / "raw_selected_events.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    raw_df = pd.read_csv(raw_path, usecols=RAW_COLUMNS)
    raw_df = raw_df[raw_df["strategy"].eq(STRATEGY)].copy()
    if raw_df.empty:
        raise ValueError(f"No {STRATEGY} events found in {raw_path}")
    if raw_df.duplicated(["run", "t"]).any():
        raise ValueError(f"Duplicate (run, t) rows found for {STRATEGY}")

    for column in ("miscovered", "n_calibration", "interval_length"):
        raw_df[column] = pd.to_numeric(raw_df[column], errors="raise")

    expected_runs = int(config.get("n_runs", 0))
    observed_runs = raw_df["run"].nunique()
    if expected_runs <= 0 or observed_runs != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} runs for {STRATEGY}, found {observed_runs}"
        )
    return raw_df


def adaptive_parameters(config):
    conformal = config.get("conformal", {})
    alpha = float(conformal["alpha"])
    lambda_min = float(conformal["adaptive_weighted_express_lambda_min"])
    lambda_max = float(conformal["adaptive_weighted_express_lambda_max"])
    max_distance = conformal.get("adaptive_weighted_express_max_distance")

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"Invalid alpha: {alpha}")
    if not 0.0 < lambda_min <= lambda_max:
        raise ValueError(
            f"Invalid adaptive lambda range: [{lambda_min}, {lambda_max}]"
        )
    return alpha, lambda_min, lambda_max, max_distance


def format_number(value):
    return f"{value:g}"


def summary_for_export(
    metrics,
    result_dir,
    n_events,
    n_runs,
    lambda_min,
    lambda_max,
    max_distance,
):
    summary = metrics.reset_index()
    summary.insert(1, "source_run_dir", Path(result_dir).name)
    summary["n_selected_events"] = n_events
    summary["n_runs"] = n_runs
    summary["adaptive_weighted_express_lambda_min"] = lambda_min
    summary["adaptive_weighted_express_lambda_max"] = lambda_max
    summary["adaptive_weighted_express_max_distance"] = max_distance
    summary["calibration_set_size_display"] = "-"
    return summary


def relative_to_project(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def write_recreation_script(output_path, result_dir, summary_csv, dpi):
    output_path = Path(output_path)
    script_path = output_path.with_suffix(".sh")
    project_from_script = os.path.relpath(
        PROJECT_ROOT.resolve(),
        start=script_path.parent.resolve(),
    )
    command = [
        "python",
        "src/scripts/sale_ramdas_adaptive_weighted_summary_vis.py",
        "--result-dir",
        str(relative_to_project(result_dir)),
        "--output",
        str(relative_to_project(output_path)),
        "--summary-csv",
        str(relative_to_project(summary_csv)),
        "--dpi",
        str(dpi),
    ]
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'cd "$(dirname "$0")/{project_from_script}"\n'
        + " ".join(shlex.quote(part) for part in command)
        + "\n"
    )
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def parse_args():
    parser = ArgumentParser(
        description=(
            "Create a one-method Sale-Ramdas-style summary for "
            "ADAPTIVE-WEIGHTED-EXPRESS."
        )
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    suite_dir = result_dir.parent
    vis_dir = suite_dir / "vis"
    output_path = args.output or (
        vis_dir / "sale_ramdas_style_summary_adaptive_weighted_express.png"
    )
    summary_csv = args.summary_csv or (
        vis_dir / "sale_ramdas_style_summary_adaptive_weighted_express.csv"
    )

    config = load_result_config(result_dir)
    if STRATEGY not in config.get("strategies", []):
        raise ValueError(f"{result_dir} is not configured for {STRATEGY}")

    raw_df = load_adaptive_events(result_dir, config)
    metrics = summarize_events(raw_df)
    if metrics.index.to_list() != [STRATEGY]:
        raise ValueError(f"Unexpected summarized strategies: {metrics.index.to_list()}")

    alpha, lambda_min, lambda_max, max_distance = adaptive_parameters(config)
    label_range = label_range_from_config(config)
    lambda_note = (
        r"(novel; $\lambda_t\in["
        + format_number(lambda_min)
        + ","
        + format_number(lambda_max)
        + "]$)"
    )
    plot_sale_ramdas_style(
        metrics,
        output_path,
        title=None,
        target_alpha=alpha,
        strategy_labels={STRATEGY: "ADAPTIVE-\nWEIGHTED-\nEXPRESS"},
        label_range=label_range,
        calibration_text_overrides={STRATEGY: "-"},
        method_note_overrides={STRATEGY: lambda_note},
        figsize=(6.4, 6.8),
        dpi=args.dpi,
    )

    exported_summary = summary_for_export(
        metrics,
        result_dir=result_dir,
        n_events=len(raw_df),
        n_runs=raw_df["run"].nunique(),
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        max_distance=max_distance,
    )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exported_summary.to_csv(summary_csv, index=False)
    script_path = write_recreation_script(
        output_path,
        result_dir=result_dir,
        summary_csv=summary_csv,
        dpi=args.dpi,
    )

    print(f"Source strategy: {STRATEGY}")
    print(f"Source result: {result_dir}")
    print(f"Wrote summary plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote recreation script to {script_path}")


if __name__ == "__main__":
    main()
