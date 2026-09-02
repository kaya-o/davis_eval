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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator, StrMethodFormatter


DEFAULT_SUITE_DIR = (
    PROJECT_ROOT
    / "results"
    / "suite_20260825_010303_adaptive_express"
)
ADAPTIVE_STRATEGY = "ADAPTIVE-WEIGHTED-EXPRESS"
AVAILABILITY_SOURCE_COLUMN = (
    "adaptive_weighted_express_express_n_calibration_for_stress"
)
STRESS_SOURCE_COLUMN = "adaptive_weighted_express_stress"
LAMBDA_SOURCE_COLUMN = "adaptive_weighted_express_lambda_t"
STRESS_COUNT_COLUMN = "adaptive_weighted_express_stress_count"
STRESS_COUNT_SOURCE_COLUMN = "adaptive_weighted_express_stress_count_source"
POINT_COLOR = "#0072B2"
REFERENCE_COLOR = "#4D4D4D"


def load_result_config(result_dir):
    for filename in ("resolved_config.json", "config.json"):
        path = Path(result_dir) / filename
        if path.exists():
            with path.open() as handle:
                return json.load(handle)
    raise FileNotFoundError(
        f"Missing resolved_config.json or config.json under {result_dir}"
    )


def find_adaptive_result_dir(suite_dir):
    suite_dir = Path(suite_dir)
    matches = []
    for result_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        aggregate_path = result_dir / "aggregate_results.csv"
        raw_path = result_dir / "raw_selected_events.csv"
        if not aggregate_path.exists() or not raw_path.exists():
            continue
        strategies = pd.read_csv(aggregate_path, usecols=["strategy"])["strategy"]
        if strategies.astype(str).eq(ADAPTIVE_STRATEGY).any():
            matches.append(result_dir)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one result directory containing {ADAPTIVE_STRATEGY} "
            f"under {suite_dir}, found {len(matches)}: "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


def load_selected_events(result_dir, run, n_on):
    raw_path = Path(result_dir) / "raw_selected_events.csv"
    required_columns = [
        "run",
        "t",
        "strategy",
        AVAILABILITY_SOURCE_COLUMN,
        STRESS_SOURCE_COLUMN,
        LAMBDA_SOURCE_COLUMN,
        STRESS_COUNT_COLUMN,
        STRESS_COUNT_SOURCE_COLUMN,
    ]
    available_columns = set(pd.read_csv(raw_path, nrows=0).columns)
    missing_columns = sorted(set(required_columns) - available_columns)
    if missing_columns:
        raise ValueError(f"Missing required raw-event columns: {missing_columns}")

    raw = pd.read_csv(raw_path, usecols=required_columns)
    selected = raw[
        raw["strategy"].astype(str).eq(ADAPTIVE_STRATEGY)
        & pd.to_numeric(raw["run"], errors="coerce").eq(run)
    ].copy()
    if selected.empty:
        raise ValueError(f"No {ADAPTIVE_STRATEGY} events found for run={run}")

    numeric_columns = [
        "run",
        "t",
        AVAILABILITY_SOURCE_COLUMN,
        STRESS_SOURCE_COLUMN,
        LAMBDA_SOURCE_COLUMN,
        STRESS_COUNT_COLUMN,
    ]
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if selected[numeric_columns].isna().any().any():
        missing = selected[numeric_columns].isna().sum()
        raise ValueError(
            "Missing or nonnumeric selected-event diagnostics: "
            f"{missing[missing > 0].to_dict()}"
        )
    if not np.isfinite(selected[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Selected-event diagnostics contain nonfinite values")
    if selected.duplicated(["run", "t"]).any():
        raise ValueError(f"Duplicate adaptive (run, t) rows found for run={run}")

    timesteps = selected["t"].to_numpy(dtype=float)
    if not np.equal(timesteps, np.floor(timesteps)).all():
        raise ValueError("Timesteps must be integer-valued")
    if np.min(timesteps) < 0 or np.max(timesteps) >= n_on:
        raise ValueError(
            f"Timesteps must lie in [0, {n_on - 1}], found "
            f"[{np.min(timesteps)}, {np.max(timesteps)}]"
        )

    count_sources = set(
        selected[STRESS_COUNT_SOURCE_COLUMN].dropna().astype(str)
    )
    if count_sources != {"express_calibration"}:
        raise ValueError(
            "Expected adaptive stress source express_calibration, found "
            f"{sorted(count_sources)}"
        )
    if not np.array_equal(
        selected[STRESS_COUNT_COLUMN].to_numpy(dtype=float),
        selected[AVAILABILITY_SOURCE_COLUMN].to_numpy(dtype=float),
    ):
        raise ValueError(
            "Adaptive stress input count does not equal exact EXPRESS availability"
        )

    selected = selected.sort_values("t").rename(
        columns={
            AVAILABILITY_SOURCE_COLUMN: "calibration_availability",
            STRESS_SOURCE_COLUMN: "stress",
            LAMBDA_SOURCE_COLUMN: "lambda_t",
        }
    )
    selected["run"] = selected["run"].astype(int)
    selected["t"] = selected["t"].astype(int)
    return selected[["run", "t", "calibration_availability", "stress", "lambda_t"]]


def validate_controller_values(
    selected,
    stress_mode,
    midpoint,
    slope,
    lambda_min,
    lambda_max,
):
    if stress_mode != "sigmoid":
        raise ValueError(f"Expected sigmoid stress mode, found {stress_mode!r}")
    if not np.isfinite(midpoint) or not np.isfinite(slope) or slope <= 0:
        raise ValueError(
            f"Invalid sigmoid midpoint/slope: midpoint={midpoint}, slope={slope}"
        )
    if lambda_min <= 0 or lambda_max < lambda_min:
        raise ValueError(f"Invalid adaptive lambda range [{lambda_min}, {lambda_max}]")

    counts = selected["calibration_availability"].to_numpy(dtype=float)
    expected_stress = 1.0 / (1.0 + np.exp(slope * (counts - midpoint)))
    if not np.allclose(
        selected["stress"].to_numpy(dtype=float),
        expected_stress,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Stored adaptive stress does not match the configured sigmoid")

    stress = selected["stress"].to_numpy(dtype=float)
    expected_lambda = np.power(lambda_max, 1.0 - stress) * np.power(
        lambda_min,
        stress,
    )
    if not np.allclose(
        selected["lambda_t"].to_numpy(dtype=float),
        expected_lambda,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Stored lambda_t does not match the configured controller")


def availability_axis(selected, midpoint):
    maximum = max(
        float(selected["calibration_availability"].max()),
        float(midpoint),
    )
    ticks = MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]).tick_values(
        0.0,
        maximum * 1.04,
    )
    ticks = ticks[ticks >= 0]
    return ticks, float(ticks[-1])


def plot_selected_events(
    selected,
    output_path,
    n_on,
    midpoint,
    lambda_min,
    lambda_max,
    dpi,
):
    x = selected["t"].to_numpy(dtype=float)
    panel_specs = [
        {
            "column": "calibration_availability",
            "ylabel": r"Exact calibration count, $N_t$",
            "reference": midpoint,
        },
        {
            "column": "stress",
            "ylabel": r"Stress, $s_t$",
            "reference": 0.5,
        },
        {
            "column": "lambda_t",
            "ylabel": r"Weight-decay rate, $\lambda_t$",
            "reference": float(np.sqrt(lambda_min * lambda_max)),
        },
    ]

    with plt.rc_context(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.0,
            "axes.linewidth": 0.9,
        }
    ):
        fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.7), sharex=True)

        for panel_index, (ax, spec) in enumerate(zip(axes, panel_specs)):
            ax.axhline(
                spec["reference"],
                color=REFERENCE_COLOR,
                linestyle=(0, (4, 2)),
                linewidth=1.35,
                zorder=1,
            )
            ax.scatter(
                x,
                selected[spec["column"]].to_numpy(dtype=float),
                s=6.0,
                alpha=0.72,
                linewidths=0,
                color=POINT_COLOR,
                zorder=3,
            )
            ax.set_ylabel(spec["ylabel"])
            ax.grid(axis="y", color="#B0B0B0", linewidth=0.7, alpha=0.28)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(
                0.985,
                0.95,
                f"({chr(ord('a') + panel_index)})",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                fontweight="semibold",
            )

        availability_ticks, availability_top = availability_axis(selected, midpoint)
        axes[0].set_yticks(availability_ticks)
        axes[0].set_ylim(0.0, availability_top)
        axes[1].set_yticks(np.linspace(0.0, 1.0, 6))
        axes[1].set_ylim(0.0, 1.03)
        lambda_ticks = MaxNLocator(
            nbins=6,
            steps=[1, 2, 2.5, 5, 10],
        ).tick_values(0.0, lambda_max)
        lambda_ticks = lambda_ticks[
            (lambda_ticks >= 0.0) & (lambda_ticks <= lambda_max)
        ]
        axes[2].set_yticks(lambda_ticks)
        axes[2].set_ylim(0.0, lambda_max * 1.03)
        axes[2].set_xlim(0.0, float(n_on))
        axes[2].xaxis.set_major_locator(MultipleLocator(2500))
        axes[2].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axes[2].set_xlabel(r"Online timestep, $t$")

        legend_handles = [
            Line2D(
                [],
                [],
                color=POINT_COLOR,
                marker="o",
                linestyle="none",
                markersize=4.5,
                markeredgewidth=0,
                label="Selected timestep",
            ),
            Line2D(
                [],
                [],
                color=REFERENCE_COLOR,
                linestyle=(0, (4, 2)),
                linewidth=1.35,
                label="Controller midpoint",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=2,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.5,
        )
        fig.subplots_adjust(
            left=0.115,
            right=0.985,
            bottom=0.105,
            top=0.91,
            hspace=0.20,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
        plt.close(fig)


def relative_to_project(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def write_recreation_script(output_path, suite_dir, csv_path, run, dpi):
    output_path = Path(output_path)
    script_path = output_path.with_suffix(".sh")
    project_from_script = os.path.relpath(
        PROJECT_ROOT.resolve(),
        start=script_path.parent.resolve(),
    )
    command = [
        "python",
        "src/scripts/adaptive_weighted_express_single_run_selected_events_vis.py",
        "--suite-dir",
        str(relative_to_project(suite_dir)),
        "--output",
        str(relative_to_project(output_path)),
        "--csv",
        str(relative_to_project(csv_path)),
        "--run",
        str(run),
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
            "Plot unbinned selected-event adaptive diagnostics for one run of "
            "ADAPTIVE-WEIGHTED-EXPRESS."
        )
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    result_dir = find_adaptive_result_dir(suite_dir)
    config = load_result_config(result_dir)
    n_on = int(config.get("data", {}).get("n_on", 0))
    if n_on <= 0:
        raise ValueError(f"Invalid data.n_on={n_on}")

    conformal = config.get("conformal", {})
    stress_mode = conformal.get("adaptive_weighted_express_stress_mode")
    midpoint = float(
        conformal.get("adaptive_weighted_express_stress_midpoint_count")
    )
    slope = float(conformal.get("adaptive_weighted_express_stress_slope"))
    lambda_min = float(conformal.get("adaptive_weighted_express_lambda_min"))
    lambda_max = float(conformal.get("adaptive_weighted_express_lambda_max"))

    selected = load_selected_events(result_dir, run=args.run, n_on=n_on)
    validate_controller_values(
        selected,
        stress_mode=stress_mode,
        midpoint=midpoint,
        slope=slope,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )

    vis_dir = suite_dir / "vis"
    stem = (
        "adaptive_weighted_express_calibration_availability_stress_lambda_"
        f"run_{args.run:02d}_selected_events_3panel"
    )
    output_path = args.output or vis_dir / f"{stem}.png"
    csv_path = args.csv or vis_dir / f"{stem}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(csv_path, index=False)
    plot_selected_events(
        selected,
        output_path=output_path,
        n_on=n_on,
        midpoint=midpoint,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        dpi=args.dpi,
    )
    recreation_script = write_recreation_script(
        output_path,
        suite_dir=suite_dir,
        csv_path=csv_path,
        run=args.run,
        dpi=args.dpi,
    )

    print(f"Adaptive result: {result_dir}")
    print(f"Run: {args.run}")
    print(f"Selected events: {len(selected):,}")
    print(f"Unselected timestep gaps: {n_on - len(selected):,}")
    print(f"Wrote plot to {output_path}")
    print(f"Wrote selected-event CSV to {csv_path}")
    print(f"Wrote recreation script to {recreation_script}")


if __name__ == "__main__":
    main()
