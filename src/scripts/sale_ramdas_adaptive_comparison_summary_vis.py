from argparse import ArgumentParser
import json
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
DEFAULT_SUITE_DIR = (
    PROJECT_ROOT / "results" / "suite_20260825_010303_adaptive_express"
)
STRATEGIES = (
    "EXPRESS",
    "FINITE-EXPRESS",
    "WEIGHTED-EXPRESS",
    "ADAPTIVE-WEIGHTED-EXPRESS",
)
RAW_COLUMNS = (
    "run",
    "t",
    "strategy",
    "miscovered",
    "n_calibration",
    "interval_length",
)


def result_dirs(suite_dir):
    directories = sorted(
        path
        for path in Path(suite_dir).iterdir()
        if path.is_dir()
        and (path / "aggregate_results.csv").exists()
        and (path / "raw_selected_events.csv").exists()
    )
    if not directories:
        raise FileNotFoundError(f"No complete result directories under {suite_dir}")
    return directories


def weighted_lambda(config):
    value = config.get("conformal", {}).get("weighted_express_lambda")
    return None if value is None else float(value)


def resolve_strategy_sources(suite_dir, benchmark_lambda):
    candidates = {strategy: [] for strategy in STRATEGIES}
    configs_by_dir = {}

    for run_dir in result_dirs(suite_dir):
        config = load_result_config(run_dir)
        configured = config.get("strategies", [])
        aggregate = pd.read_csv(run_dir / "aggregate_results.csv", usecols=["strategy"])
        present = set(aggregate["strategy"].astype(str))
        configs_by_dir[run_dir] = config

        if configured == ["EXPRESS"] and "EXPRESS" in present:
            candidates["EXPRESS"].append(run_dir)
        if configured == ["FINITE-EXPRESS"] and "FINITE-EXPRESS" in present:
            candidates["FINITE-EXPRESS"].append(run_dir)
        if (
            configured == ["WEIGHTED-EXPRESS"]
            and "WEIGHTED-EXPRESS" in present
            and weighted_lambda(config) is not None
            and np.isclose(
                weighted_lambda(config),
                benchmark_lambda,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            candidates["WEIGHTED-EXPRESS"].append(run_dir)
        if (
            "ADAPTIVE-WEIGHTED-EXPRESS" in configured
            and "ADAPTIVE-WEIGHTED-EXPRESS" in present
        ):
            candidates["ADAPTIVE-WEIGHTED-EXPRESS"].append(run_dir)

    sources = {}
    for strategy, run_candidates in candidates.items():
        if len(run_candidates) != 1:
            names = [path.name for path in run_candidates]
            raise ValueError(
                f"Expected exactly one source for {strategy}, "
                f"found {len(run_candidates)}: {names}"
            )
        sources[strategy] = run_candidates[0]
    configs = {
        strategy: configs_by_dir[run_dir]
        for strategy, run_dir in sources.items()
    }
    return sources, configs


def comparison_signature(config):
    return {
        "n_runs": config.get("n_runs"),
        "seed": config.get("seed"),
        "data": config.get("data"),
        "selection": config.get("selection"),
        "prediction": config.get("prediction"),
        "alpha": config.get("conformal", {}).get("alpha"),
        "randomized_calibration": config.get("conformal", {}).get(
            "randomized_calibration"
        ),
    }


def validate_comparison_configs(configs):
    signatures = {
        json.dumps(comparison_signature(config), sort_keys=True)
        for config in configs.values()
    }
    if len(signatures) != 1:
        details = {
            strategy: comparison_signature(config)
            for strategy, config in configs.items()
        }
        raise ValueError(f"Comparison configs are not aligned: {details}")


def load_comparison_events(sources):
    frames = []
    reference_keys = None
    reference_strategy = None

    for strategy in STRATEGIES:
        run_dir = sources[strategy]
        frame = pd.read_csv(run_dir / "raw_selected_events.csv", usecols=RAW_COLUMNS)
        frame = frame[frame["strategy"].eq(strategy)].copy()
        if frame.empty:
            raise ValueError(f"No {strategy} events found in {run_dir}")
        if frame.duplicated(["run", "t"]).any():
            raise ValueError(f"Duplicate (run, t) rows found for {strategy}")

        for column in ("miscovered", "n_calibration", "interval_length"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        frame = frame.sort_values(["run", "t"]).reset_index(drop=True)
        keys = pd.MultiIndex.from_frame(frame[["run", "t"]])
        if reference_keys is None:
            reference_keys = keys
            reference_strategy = strategy
        elif not keys.equals(reference_keys):
            raise ValueError(
                f"Selected (run, t) events for {strategy} do not match "
                f"those for {reference_strategy}"
            )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def target_alpha(configs):
    values = {
        float(config.get("conformal", {}).get("alpha"))
        for config in configs.values()
    }
    if len(values) != 1:
        raise ValueError(f"Expected one target alpha, found {sorted(values)}")
    alpha = values.pop()
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"Invalid target alpha: {alpha}")
    return alpha


def adaptive_lambda_range(configs):
    conformal = configs["ADAPTIVE-WEIGHTED-EXPRESS"].get("conformal", {})
    lambda_min = float(conformal["adaptive_weighted_express_lambda_min"])
    lambda_max = float(conformal["adaptive_weighted_express_lambda_max"])
    if not 0.0 < lambda_min <= lambda_max:
        raise ValueError(f"Invalid adaptive lambda range: [{lambda_min}, {lambda_max}]")
    return lambda_min, lambda_max


def format_number(value):
    return f"{value:g}"


def summary_for_export(
    metrics,
    sources,
    benchmark_lambda,
    lambda_min,
    lambda_max,
):
    summary = metrics.reset_index()
    summary.insert(
        1,
        "source_run_dir",
        summary["strategy"].map(
            {strategy: run_dir.name for strategy, run_dir in sources.items()}
        ),
    )
    summary["weighted_express_lambda"] = np.where(
        summary["strategy"].eq("WEIGHTED-EXPRESS"),
        benchmark_lambda,
        np.nan,
    )
    summary["adaptive_weighted_express_lambda_min"] = np.where(
        summary["strategy"].eq("ADAPTIVE-WEIGHTED-EXPRESS"),
        lambda_min,
        np.nan,
    )
    summary["adaptive_weighted_express_lambda_max"] = np.where(
        summary["strategy"].eq("ADAPTIVE-WEIGHTED-EXPRESS"),
        lambda_max,
        np.nan,
    )
    summary["calibration_set_size_display"] = summary[
        "avg_n_calibration"
    ].map(lambda value: f"{value:.3f}")
    summary.loc[
        summary["strategy"].isin(
            ["WEIGHTED-EXPRESS", "ADAPTIVE-WEIGHTED-EXPRESS"]
        ),
        "calibration_set_size_display",
    ] = "-"
    return summary


def relative_to_project(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def write_recreation_script(
    output_path,
    suite_dir,
    summary_csv,
    benchmark_lambda,
    dpi,
):
    output_path = Path(output_path)
    script_path = output_path.with_suffix(".sh")
    project_from_script = os.path.relpath(
        PROJECT_ROOT.resolve(),
        start=script_path.parent.resolve(),
    )
    command = [
        "python",
        "src/scripts/sale_ramdas_adaptive_comparison_summary_vis.py",
        "--suite-dir",
        str(relative_to_project(suite_dir)),
        "--benchmark-lambda",
        format_number(benchmark_lambda),
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
            "Create a Sale-Ramdas-style summary for EXPRESS, FINITE-EXPRESS, "
            "a fixed WEIGHTED-EXPRESS benchmark, and ADAPTIVE-WEIGHTED-EXPRESS."
        )
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--benchmark-lambda", type=float, default=25.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    vis_dir = suite_dir / "vis"
    output_path = args.output or (
        vis_dir / "sale_ramdas_style_summary_adaptive_express_comparison.png"
    )
    summary_csv = args.summary_csv or (
        vis_dir / "sale_ramdas_style_summary_adaptive_express_comparison.csv"
    )

    sources, configs = resolve_strategy_sources(
        suite_dir,
        benchmark_lambda=args.benchmark_lambda,
    )
    validate_comparison_configs(configs)
    raw_events = load_comparison_events(sources)
    metrics = summarize_events(raw_events)
    alpha = target_alpha(configs)
    lambda_min, lambda_max = adaptive_lambda_range(configs)
    label_range = label_range_from_config(configs["EXPRESS"])

    lambda_min_text = format_number(lambda_min)
    lambda_max_text = format_number(lambda_max)
    plot_sale_ramdas_style(
        metrics,
        output_path,
        title=None,
        target_alpha=alpha,
        strategy_labels={
            "EXPRESS": "EXPRESS",
            "FINITE-EXPRESS": "FINITE\nEXPRESS",
            "WEIGHTED-EXPRESS": "WEIGHTED\nEXPRESS",
            "ADAPTIVE-WEIGHTED-EXPRESS": "ADAPTIVE-WEIGHTED\nEXPRESS",
        },
        label_range=label_range,
        calibration_text_overrides={
            "WEIGHTED-EXPRESS": "-",
            "ADAPTIVE-WEIGHTED-EXPRESS": "-",
        },
        method_note_overrides={
            "WEIGHTED-EXPRESS": (
                f"(novel; $\\lambda={format_number(args.benchmark_lambda)}$)"
            ),
            "ADAPTIVE-WEIGHTED-EXPRESS": (
                r"(novel; $\lambda_t\in["
                + lambda_min_text
                + ","
                + lambda_max_text
                + "]$)"
            ),
        },
        dpi=args.dpi,
    )

    exported_summary = summary_for_export(
        metrics,
        sources=sources,
        benchmark_lambda=args.benchmark_lambda,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exported_summary.to_csv(summary_csv, index=False)
    script_path = write_recreation_script(
        output_path,
        suite_dir=suite_dir,
        summary_csv=summary_csv,
        benchmark_lambda=args.benchmark_lambda,
        dpi=args.dpi,
    )

    print("Strategies: " + ", ".join(STRATEGIES))
    for strategy in STRATEGIES:
        print(f"{strategy}: {sources[strategy]}")
    print(f"Wrote summary plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote recreation script to {script_path}")


if __name__ == "__main__":
    main()
