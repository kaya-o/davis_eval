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


DEFAULT_SUITE_DIR = PROJECT_ROOT / "results" / "suite_20260612_022224_relaxed_express_r_sweep_easy"
DEFAULT_OUTPUT_NAME = "relaxed_express_r_sweep_calibration_timeseries_3x3.png"
DEFAULT_SUMMARY_NAME = "relaxed_express_r_sweep_calibration_timeseries_3x3.csv"
STRATEGY = "RELAXED-EXPRESS"


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


def aggregate_mean_calibration_by_time(raw_path, strategy, chunksize):
    sum_by_t = pd.Series(dtype=float)
    count_by_t = pd.Series(dtype=float)

    for chunk in pd.read_csv(
        raw_path,
        usecols=["t", "strategy", "n_calibration"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["strategy"] == strategy]
        if chunk.empty:
            continue

        chunk["t"] = pd.to_numeric(chunk["t"], errors="coerce")
        chunk["n_calibration"] = pd.to_numeric(chunk["n_calibration"], errors="coerce")
        chunk = chunk.dropna(subset=["t", "n_calibration"])
        grouped = chunk.groupby("t")["n_calibration"].agg(["sum", "count"])
        sum_by_t = sum_by_t.add(grouped["sum"], fill_value=0.0)
        count_by_t = count_by_t.add(grouped["count"], fill_value=0.0)

    if count_by_t.empty:
        raise ValueError(f"No rows for strategy {strategy!r} in {raw_path}")

    mean_by_t = (sum_by_t / count_by_t).sort_index()
    return pd.DataFrame({
        "t": mean_by_t.index.astype(int),
        "mean_n_calibration": mean_by_t.to_numpy(dtype=float),
        "selected_events": count_by_t.loc[mean_by_t.index].to_numpy(dtype=int),
    })


def summarize_suite(suite_dir, strategy, chunksize):
    rows = []
    for run_dir in result_dirs(suite_dir):
        config = load_result_config(run_dir)
        r_value = config.get("conformal", {}).get("relaxed_express_max_distance")
        if r_value is None:
            continue

        time_df = aggregate_mean_calibration_by_time(
            run_dir / "raw_selected_events.csv",
            strategy,
            chunksize,
        )
        time_df["r"] = float(r_value)
        time_df["run_dir"] = run_dir.name
        rows.append(time_df)

    if not rows:
        raise FileNotFoundError(
            f"No {strategy} runs with conformal.relaxed_express_max_distance under {suite_dir}"
        )

    return pd.concat(rows, ignore_index=True).sort_values(["r", "t"]).reset_index(drop=True)


def plot_summary(summary_df, output_path, strategy):
    r_values = sorted(summary_df["r"].unique())
    n_panels = len(r_values)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4.2 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)

    y_max = summary_df["mean_n_calibration"].max()
    y_top = y_max * 1.08 if np.isfinite(y_max) and y_max > 0 else 1.0

    for ax, r_value in zip(axes, r_values):
        panel = summary_df[summary_df["r"] == r_value].sort_values("t")
        ax.scatter(
            panel["t"],
            panel["mean_n_calibration"],
            s=3,
            alpha=0.75,
            color="#1f77b4",
            linewidths=0,
        )
        ax.set_title(f"r={r_value:g}", fontsize=10)
        ax.set_ylim(bottom=0, top=y_top)
        ax.grid(alpha=0.25)

    for ax in axes[n_panels:]:
        ax.axis("off")

    axes_matrix = axes.reshape(n_rows, n_cols)
    for ax in axes_matrix[:, 0]:
        if ax.has_data():
            ax.set_ylabel("calibration set size")
    for ax in axes_matrix[-1, :]:
        if ax.has_data():
            ax.set_xlabel("t")

    fig.suptitle(f"{strategy} calibration set size over time", y=0.995, fontsize=13)
    fig.tight_layout()
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


def write_recreation_script(output_path, suite_dir, summary_csv, strategy, chunksize):
    output_path = Path(output_path)
    script_path = output_path.with_suffix(".sh")
    command = [
        "python3",
        "src/scripts/relaxed_express_r_sweep_calibration_timeseries_vis.py",
        "--suite-dir",
        str(relative_to_project(suite_dir)),
        "--strategy",
        strategy,
        "--output",
        str(relative_to_project(output_path)),
        "--summary-csv",
        str(relative_to_project(summary_csv)),
        "--chunksize",
        str(chunksize),
    ]
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")/../../.."\n'
        + " ".join(shlex.quote(part) for part in command)
        + "\n"
    )
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def parse_args():
    parser = ArgumentParser(
        description="Plot RELAXED-EXPRESS calibration set size over time for each swept radius."
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--strategy", default=STRATEGY)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    vis_dir = suite_dir / "vis"
    output_path = args.output or vis_dir / DEFAULT_OUTPUT_NAME
    summary_csv = args.summary_csv or vis_dir / DEFAULT_SUMMARY_NAME

    summary_df = summarize_suite(
        suite_dir=suite_dir,
        strategy=args.strategy,
        chunksize=args.chunksize,
    )
    summary_df = summary_df[["r", "run_dir", "t", "mean_n_calibration", "selected_events"]]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    plot_summary(summary_df, output_path, args.strategy)
    script_path = write_recreation_script(
        output_path,
        suite_dir,
        summary_csv,
        args.strategy,
        args.chunksize,
    )

    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote recreation script to {script_path}")
    print(f"r panels={summary_df['r'].nunique()}")


if __name__ == "__main__":
    main()
