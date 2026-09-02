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


DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "suite_20260612_041714_weighted_express_lambda_sweep_harder"
    / "20260612_042045_lambda_35"
)
DEFAULT_OUTPUT_NAME = "selection_bounds_timeseries_3x3.png"
DEFAULT_SUMMARY_NAME = "selection_bounds_timeseries_3x3.csv"


def load_result_config(result_dir):
    for filename in ("resolved_config.json", "config.json"):
        config_path = Path(result_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    raise FileNotFoundError(f"Missing config.json or resolved_config.json under {result_dir}")


def reconstruct_bounds_for_run(selected_df, run, n_on, tau_0, tau_1, window_width):
    run_selected = selected_df[selected_df["run"] == run].sort_values("t").copy()
    selected_by_t = np.zeros(n_on, dtype=np.int64)
    valid_selected_times = run_selected["t"].to_numpy(dtype=int)
    valid_selected_times = valid_selected_times[(0 <= valid_selected_times) & (valid_selected_times < n_on)]
    selected_by_t[valid_selected_times] = 1

    past_selected = np.cumsum(selected_by_t) - selected_by_t
    t = np.arange(n_on, dtype=np.int64)
    lower = tau_1 + past_selected / tau_0
    upper = lower + window_width

    score_by_t = np.full(n_on, np.nan)
    if not run_selected.empty:
        score_by_t[valid_selected_times] = run_selected.loc[
            run_selected["t"].isin(valid_selected_times),
            "score_t",
        ].to_numpy(dtype=float)

    return pd.DataFrame({
        "run": run,
        "t": t,
        "selection_lower_bound": lower,
        "selection_upper_bound": upper,
        "selected": selected_by_t,
        "score_t": score_by_t,
        "past_selected": past_selected,
    })


def reconstruct_bounds(result_dir, n_runs):
    result_dir = Path(result_dir)
    config = load_result_config(result_dir)
    data_config = config["data"]
    selection_config = config["selection"]
    selected_path = result_dir / "selected_datapoints.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected_datapoints.csv under {result_dir}")

    selected_df = pd.read_csv(selected_path)
    runs = sorted(selected_df["run"].unique())[:n_runs]
    if len(runs) < n_runs:
        raise ValueError(f"Requested {n_runs} runs, but only found {len(runs)} in {selected_path}")

    n_on = int(data_config["n_on"])
    tau_0 = float(selection_config["tau_0"])
    tau_1 = float(selection_config["tau_1"])
    window_width = float(selection_config["window_width"])

    rows = [
        reconstruct_bounds_for_run(
            selected_df=selected_df,
            run=run,
            n_on=n_on,
            tau_0=tau_0,
            tau_1=tau_1,
            window_width=window_width,
        )
        for run in runs
    ]
    return pd.concat(rows, ignore_index=True), config


def finite_upper_for_plot(summary_df):
    finite_upper = summary_df.loc[
        np.isfinite(summary_df["selection_upper_bound"]),
        "selection_upper_bound",
    ]
    score = summary_df["score_t"].dropna()
    ymax = max(finite_upper.max(), score.max() if not score.empty else finite_upper.max())
    return float(ymax)


def plot_selection_bounds(summary_df, output_path):
    runs = sorted(summary_df["run"].unique())
    if len(runs) != 9:
        raise ValueError(f"Expected exactly 9 runs for a 3x3 panel, got {len(runs)}")

    y_max = finite_upper_for_plot(summary_df)
    y_min = float(summary_df["selection_lower_bound"].min())
    y_pad = max((y_max - y_min) * 0.08, 0.05)
    y_bottom = y_min - y_pad
    y_top = y_max + y_pad

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 10.5), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, run in zip(axes, runs):
        run_df = summary_df[summary_df["run"] == run].sort_values("t")
        t = run_df["t"].to_numpy()
        lower = run_df["selection_lower_bound"].to_numpy()
        upper = run_df["selection_upper_bound"].replace([np.inf, -np.inf], np.nan).to_numpy()
        selected = run_df[run_df["selected"] == 1]

        ax.fill_between(t, lower, upper, step="post", color="#f59e0b", alpha=0.22, linewidth=0)
        ax.step(t, lower, where="post", color="#d97706", linewidth=1.1)
        ax.step(t, upper, where="post", color="#d97706", linewidth=1.1)
        ax.scatter(
            selected["t"],
            selected["score_t"],
            s=4,
            alpha=0.35,
            edgecolors="none",
            color="#1f77b4",
        )
        ax.set_title(f"Run {run}", fontsize=10)
        ax.grid(alpha=0.22)
        ax.set_xlim(0, int(t.max()))
        ax.set_ylim(y_bottom, y_top)

    for ax in axes[6:9]:
        ax.set_xlabel("t")
    for ax in axes[::3]:
        ax.set_ylabel("selection score")

    handles = [
        plt.Line2D([0], [0], color="#d97706", lw=1.6),
        plt.Rectangle((0, 0), 1, 1, color="#f59e0b", alpha=0.22),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markersize=5),
    ]
    fig.legend(
        handles,
        ["selection bounds", "selection window", "selected score"],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Selection Window Over Time", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))

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


def write_recreation_script(output_path, result_dir, summary_csv, n_runs):
    output_path = Path(output_path)
    script_path = output_path.with_suffix(".sh")
    script_dir_to_project = Path("../../..")
    command = [
        "python3",
        "src/scripts/selection_bounds_timeseries_vis.py",
        "--result-dir",
        str(relative_to_project(result_dir)),
        "--output",
        str(relative_to_project(output_path)),
        "--summary-csv",
        str(relative_to_project(summary_csv)),
        "--n-runs",
        str(n_runs),
    ]
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")/'
        f"{script_dir_to_project}"
        '"\n'
        + " ".join(shlex.quote(part) for part in command)
        + "\n"
    )
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def parse_args():
    parser = ArgumentParser(description="Plot reconstructed selection bounds over online time.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--n-runs", type=int, default=9)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_path = args.output or result_dir / DEFAULT_OUTPUT_NAME
    summary_csv = args.summary_csv or result_dir / DEFAULT_SUMMARY_NAME

    summary_df, _ = reconstruct_bounds(result_dir, n_runs=args.n_runs)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    plot_selection_bounds(summary_df, output_path)
    script_path = write_recreation_script(output_path, result_dir, summary_csv, args.n_runs)

    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote recreation script to {script_path}")


if __name__ == "__main__":
    main()
