from argparse import ArgumentParser
import json
import os
from pathlib import Path
import shlex

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt


DEFAULT_RUNS = [
    (
        "easy",
        PROJECT_ROOT
        / "results"
        / "suite_20260612_033617_weighted_express_lambda_sweep_easy"
        / "20260612_035806_lambda_35",
    ),
    (
        "hard",
        PROJECT_ROOT
        / "results"
        / "suite_20260612_033624_weighted_express_lambda_sweep_hard"
        / "20260612_034854_lambda_35",
    ),
    (
        "harder",
        PROJECT_ROOT
        / "results"
        / "suite_20260612_041714_weighted_express_lambda_sweep_harder"
        / "20260612_042045_lambda_35",
    ),
]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "vis" / "weighted_express_lambda35_easy_hard_harder_metric_bars.png"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "results" / "vis" / "weighted_express_lambda35_easy_hard_harder_metric_bars.csv"
)

METRICS = [
    ("miscoverage", "miscoverage"),
    ("median_interval_length", "median interval length"),
    ("infinite_fraction", "infinite interval fraction"),
]
LABEL_MIN = 5.0
LABEL_MAX = 10.735
LABEL_RANGE = LABEL_MAX - LABEL_MIN
EXPRESS_FINITE_LABEL = "EXPRESS (finite only)"
BASE_STRATEGY_ORDER = [
    "EXPRESS",
    "RELAXED-EXPRESS",
    "WEIGHTED-EXPRESS",
    "WEIGHTED-NEIGHBORHOOD-EXPRESS",
    "ADAPTIVE-WEIGHTED-EXPRESS",
]
PANEL_STRATEGY_ORDER = {
    "miscoverage": [
        "EXPRESS",
        EXPRESS_FINITE_LABEL,
        "RELAXED-EXPRESS",
        "WEIGHTED-EXPRESS",
        "WEIGHTED-NEIGHBORHOOD-EXPRESS",
        "ADAPTIVE-WEIGHTED-EXPRESS",
    ],
    "median_interval_length": [
        "EXPRESS",
        EXPRESS_FINITE_LABEL,
        "RELAXED-EXPRESS",
        "WEIGHTED-EXPRESS",
        "WEIGHTED-NEIGHBORHOOD-EXPRESS",
        "ADAPTIVE-WEIGHTED-EXPRESS",
    ],
    "infinite_fraction": BASE_STRATEGY_ORDER,
}
STRATEGY_ORDER = [
    "EXPRESS",
    EXPRESS_FINITE_LABEL,
    "RELAXED-EXPRESS",
    "WEIGHTED-EXPRESS",
    "WEIGHTED-NEIGHBORHOOD-EXPRESS",
    "ADAPTIVE-WEIGHTED-EXPRESS",
]


def parse_run_arg(value):
    if "=" not in value:
        raise ValueError(f"Run argument must have LABEL=PATH form, got {value!r}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Run label cannot be empty in {value!r}")
    return label, Path(path)


def load_result_config(run_dir):
    for filename in ("resolved_config.json", "config.json"):
        config_path = Path(run_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    return {}


def load_runs(runs):
    rows = []
    for label, run_dir in runs:
        run_dir = Path(run_dir)
        aggregate_path = run_dir / "aggregate_results.csv"
        if not aggregate_path.exists():
            raise FileNotFoundError(f"Missing aggregate_results.csv under {run_dir}")

        config = load_result_config(run_dir)
        conformal_config = config.get("conformal", {})
        alpha = conformal_config.get("alpha")
        lambda_value = conformal_config.get("weighted_express_lambda")
        max_neighbors = conformal_config.get("weighted_neighborhood_express_max_neighbors")
        aggregate = pd.read_csv(aggregate_path)
        aggregate["setting"] = label
        aggregate["run_dir"] = run_dir.name
        aggregate["alpha"] = float(alpha) if alpha is not None else np.nan
        aggregate["lambda"] = float(lambda_value) if lambda_value is not None else np.nan
        aggregate["max_neighbors"] = int(max_neighbors) if max_neighbors is not None else np.nan
        aggregate = pd.concat(
            [aggregate, express_finite_only_row(run_dir, aggregate, label, alpha)],
            ignore_index=True,
        )
        rows.append(aggregate)

    summary = pd.concat(rows, ignore_index=True)
    summary = summary[summary["strategy"].isin(STRATEGY_ORDER)].copy()
    for metric, _ in METRICS:
        summary[metric] = pd.to_numeric(summary[metric], errors="coerce")
    summary["strategy"] = pd.Categorical(
        summary["strategy"],
        categories=STRATEGY_ORDER,
        ordered=True,
    )
    return summary.sort_values(["strategy", "setting"]).reset_index(drop=True)


def express_finite_only_row(run_dir, aggregate, label, alpha):
    raw_path = Path(run_dir) / "raw_selected_events.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw_selected_events.csv under {run_dir}")

    express_raw = pd.read_csv(
        raw_path,
        usecols=["strategy", "miscovered", "n_calibration", "interval_length"],
    )
    express_raw = express_raw[express_raw["strategy"] == "EXPRESS"].copy()
    express_raw["interval_length"] = pd.to_numeric(express_raw["interval_length"], errors="coerce")
    express_raw["miscovered"] = pd.to_numeric(express_raw["miscovered"], errors="coerce")
    express_raw["n_calibration"] = pd.to_numeric(express_raw["n_calibration"], errors="coerce")
    finite_express = express_raw[np.isfinite(express_raw["interval_length"])].copy()

    selected = len(finite_express)
    if selected == 0:
        miscovered = 0
        miscoverage = np.nan
        avg_n_calibration = np.nan
        median_interval_length = np.nan
    else:
        miscovered = int(finite_express["miscovered"].sum())
        miscoverage = miscovered / selected
        avg_n_calibration = float(finite_express["n_calibration"].mean())
        median_interval_length = float(finite_express["interval_length"].median())

    lambda_value = np.nan
    if "lambda" in aggregate.columns:
        lambda_values = pd.to_numeric(aggregate["lambda"], errors="coerce").dropna()
        if not lambda_values.empty:
            lambda_value = float(lambda_values.iloc[0])

    aggregate_columns = list(aggregate.columns)
    row = {
        "strategy": EXPRESS_FINITE_LABEL,
        "selected": selected,
        "miscovered": miscovered,
        "miscoverage": miscoverage,
        "avg_n_calibration": avg_n_calibration,
        "median_interval_length": median_interval_length,
        "infinite_fraction": np.nan,
        "setting": label,
        "run_dir": Path(run_dir).name,
        "alpha": float(alpha) if alpha is not None else np.nan,
        "lambda": lambda_value,
    }
    return pd.DataFrame([{column: row.get(column, np.nan) for column in aggregate_columns}])


def finite_axis_limit(values):
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 1.0
    max_value = float(np.max(finite_values))
    if max_value <= 0:
        return 1.0
    return max_value * 1.22


def format_bar_label(value, metric):
    if np.isinf(value):
        if metric == "median_interval_length":
            return "inf (100%)"
        return "inf"
    if metric == "median_interval_length":
        return f"{value:.2f} ({100 * value / LABEL_RANGE:.1f}%)"
    return f"{value:.3f}"


def plot_metric_bars(summary, output_path):
    settings = list(summary["setting"].drop_duplicates())
    lambda_values = summary["lambda"].dropna().unique()
    lambda_label = f"{float(lambda_values[0]):g}" if len(lambda_values) == 1 else "mixed"
    max_neighbor_values = summary["max_neighbors"].dropna().unique()
    max_neighbor_label = (
        f", max-neighbors {int(max_neighbor_values[0])}"
        if len(max_neighbor_values) == 1
        else ""
    )
    bar_height = min(0.36, 0.8 / max(len(settings), 1))
    colors = {
        "easy": "#1f77b4",
        "hard": "#d55e00",
        "harder": "#009e73",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.1))
    for ax, (metric, ylabel) in zip(axes, METRICS):
        strategies = [
            strategy
            for strategy in PANEL_STRATEGY_ORDER[metric]
            if strategy in set(summary["strategy"].astype(str))
        ]
        y = np.arange(len(strategies))
        metric_values = summary[metric].to_numpy(dtype=float)
        x_right = finite_axis_limit(metric_values)
        if metric == "median_interval_length":
            x_right *= 1.2
        inf_width = x_right * 0.92

        for setting_idx, setting in enumerate(settings):
            setting_rows = summary[summary["setting"] == setting].set_index("strategy")
            values = []
            raw_values = []
            for strategy in strategies:
                raw_value = setting_rows.loc[strategy, metric] if strategy in setting_rows.index else np.nan
                raw_value = float(raw_value)
                raw_values.append(raw_value)
                values.append(inf_width if np.isinf(raw_value) else raw_value)

            offsets = y + (setting_idx - (len(settings) - 1) / 2) * bar_height
            bars = ax.barh(
                offsets,
                values,
                height=bar_height,
                label=setting,
                color=colors.get(setting, None),
                alpha=0.9,
            )
            for bar, raw_value in zip(bars, raw_values):
                if not np.isfinite(raw_value) and not np.isinf(raw_value):
                    bar.set_visible(False)
                    continue
                if np.isinf(raw_value):
                    bar.set_hatch("//")
                    bar.set_edgecolor("black")
                label = format_bar_label(raw_value, metric)
                ax.text(
                    min(bar.get_width() + x_right * 0.015, x_right * 0.975),
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    ha="left",
                    va="center",
                    fontsize=8,
                )

        if metric == "miscoverage":
            alpha_rows = summary[["setting", "alpha"]].drop_duplicates().dropna()
            for _, row in alpha_rows.iterrows():
                ax.axvline(
                    float(row["alpha"]),
                    color=colors.get(row["setting"], "#666666"),
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                    label=f"{row['setting']} target ({float(row['alpha']):g})",
                )

        ax.set_title(ylabel)
        ax.set_xlabel(ylabel)
        ax.set_yticks(y)
        ax.set_yticklabels(strategies)
        ax.set_xlim(0, x_right)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)

    axes[0].set_ylabel("strategy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=min(len(labels), 4),
    )
    fig.suptitle(
        f"Lambda {lambda_label}{max_neighbor_label} strategy metrics: {' vs '.join(settings)}",
        y=0.99,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def relative_to_project(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def write_recreation_script(output_path, summary_csv, runs):
    output_path = Path(output_path)
    script_path = output_path.with_suffix(".sh")
    script_dir_to_project = Path("../..")
    command = [
        "python3",
        "src/scripts/two_run_metric_bars_vis.py",
        "--output",
        str(relative_to_project(output_path)),
        "--summary-csv",
        str(relative_to_project(summary_csv)),
    ]
    for label, run_dir in runs:
        command.extend(["--run", f"{label}={relative_to_project(run_dir)}"])

    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")/'
        f"{script_dir_to_project}"
        '"\n'
        + " ".join(shlex.quote(str(part)) for part in command)
        + "\n"
    )
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def parse_args():
    parser = ArgumentParser(description="Plot metric bar charts comparing two aggregate-result runs.")
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        default=None,
        help="Run to include, in LABEL=PATH form. Can be passed more than once.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main():
    args = parse_args()
    runs = [parse_run_arg(run) for run in args.runs] if args.runs else DEFAULT_RUNS
    summary = load_runs(runs)

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "setting",
        "run_dir",
        "strategy",
        "selected",
        "miscovered",
        "miscoverage",
        "median_interval_length",
        "infinite_fraction",
        "avg_n_calibration",
        "alpha",
        "lambda",
        "max_neighbors",
    ]
    summary[output_columns].to_csv(summary_csv, index=False)
    plot_metric_bars(summary, args.output)
    script_path = write_recreation_script(args.output, summary_csv, runs)

    print(f"Wrote plot to {args.output}")
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote recreation script to {script_path}")


if __name__ == "__main__":
    main()
