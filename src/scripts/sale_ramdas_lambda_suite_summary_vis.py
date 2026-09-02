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
    PROJECT_ROOT / "results" / "suite_20260819_130943_lambda_sweep"
)
DEFAULT_LAMBDA = 25.0
STRATEGIES = ("FULL", "EXPRESS", "RELAXED-EXPRESS", "WEIGHTED-EXPRESS")
RAW_COLUMNS = (
    "run",
    "t",
    "strategy",
    "miscovered",
    "n_calibration",
    "interval_length",
)


def result_dirs(suite_dir):
    dirs = sorted(
        path
        for path in Path(suite_dir).iterdir()
        if path.is_dir()
        and (path / "aggregate_results.csv").exists()
        and (path / "raw_selected_events.csv").exists()
    )
    if not dirs:
        raise FileNotFoundError(f"No complete result directories under {suite_dir}")
    return dirs


def weighted_lambda(config):
    value = config.get("conformal", {}).get("weighted_express_lambda")
    return None if value is None else float(value)


def resolve_strategy_sources(suite_dir, lambda_value):
    candidates = {strategy: [] for strategy in STRATEGIES}
    configs = {}

    for run_dir in result_dirs(suite_dir):
        aggregate = pd.read_csv(run_dir / "aggregate_results.csv", usecols=["strategy"])
        present = set(aggregate["strategy"].astype(str))
        config = load_result_config(run_dir)

        for strategy in STRATEGIES:
            if strategy not in present:
                continue
            if strategy == "WEIGHTED-EXPRESS":
                observed_lambda = weighted_lambda(config)
                if observed_lambda is None or not np.isclose(
                    observed_lambda,
                    lambda_value,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    continue
            candidates[strategy].append(run_dir)
            configs[run_dir] = config

    sources = {}
    for strategy, run_dirs in candidates.items():
        if len(run_dirs) != 1:
            names = [run_dir.name for run_dir in run_dirs]
            raise ValueError(
                f"Expected exactly one source for {strategy}, found {len(run_dirs)}: {names}"
            )
        sources[strategy] = run_dirs[0]

    return sources, {strategy: configs[run_dir] for strategy, run_dir in sources.items()}


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
        frame = frame[frame["strategy"] == strategy].copy()
        if frame.empty:
            raise ValueError(f"No {strategy} events found in {run_dir}")

        frame["miscovered"] = pd.to_numeric(frame["miscovered"], errors="raise")
        frame["n_calibration"] = pd.to_numeric(
            frame["n_calibration"], errors="raise"
        )
        frame["interval_length"] = pd.to_numeric(
            frame["interval_length"], errors="raise"
        )
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
    return values.pop()


def relaxed_radius(configs):
    value = configs["RELAXED-EXPRESS"].get("conformal", {}).get(
        "relaxed_express_max_distance"
    )
    if value is None:
        raise KeyError("RELAXED-EXPRESS config has no maximum distance")
    return float(value)


def summary_for_export(metrics, sources, lambda_value, radius):
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
        lambda_value,
        np.nan,
    )
    summary["relaxed_express_max_distance"] = np.where(
        summary["strategy"].eq("RELAXED-EXPRESS"),
        radius,
        np.nan,
    )
    summary["calibration_set_size_display"] = summary["avg_n_calibration"].map(
        lambda value: f"{value:.3f}"
    )
    summary.loc[
        summary["strategy"].eq("WEIGHTED-EXPRESS"),
        "calibration_set_size_display",
    ] = "-"
    return summary


def relative_to_project(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def write_recreation_script(output_path, suite_dir, summary_csv, lambda_value):
    output_path = Path(output_path)
    script_path = output_path.with_suffix(".sh")
    project_from_script = os.path.relpath(
        PROJECT_ROOT.resolve(),
        start=script_path.parent.resolve(),
    )
    command = [
        "python",
        "src/scripts/sale_ramdas_lambda_suite_summary_vis.py",
        "--suite-dir",
        str(relative_to_project(suite_dir)),
        "--lambda-value",
        f"{lambda_value:g}",
        "--output",
        str(relative_to_project(output_path)),
        "--summary-csv",
        str(relative_to_project(summary_csv)),
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
            "Create a Sale-Ramdas-style summary for one WEIGHTED-EXPRESS "
            "lambda and the suite baselines."
        )
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--lambda-value", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    lambda_slug = f"{args.lambda_value:g}".replace(".", "_")
    vis_dir = suite_dir / "vis"
    output_path = args.output or (
        vis_dir / f"sale_ramdas_style_summary_lambda_{lambda_slug}.png"
    )
    summary_csv = args.summary_csv or (
        vis_dir / f"sale_ramdas_style_summary_lambda_{lambda_slug}.csv"
    )

    sources, configs = resolve_strategy_sources(suite_dir, args.lambda_value)
    validate_comparison_configs(configs)
    raw_events = load_comparison_events(sources)
    metrics = summarize_events(raw_events)
    alpha = target_alpha(configs)
    radius = relaxed_radius(configs)
    label_range = label_range_from_config(configs["WEIGHTED-EXPRESS"])

    strategy_labels = {
        "FULL": "FULL",
        "EXPRESS": "EXPRESS",
        "RELAXED-EXPRESS": "RELAXED\nEXPRESS",
        "WEIGHTED-EXPRESS": "WEIGHTED\nEXPRESS",
    }
    plot_sale_ramdas_style(
        metrics,
        output_path,
        title=None,
        target_alpha=alpha,
        strategy_labels=strategy_labels,
        label_range=label_range,
        calibration_text_overrides={"WEIGHTED-EXPRESS": "-"},
        method_note_overrides={
            "RELAXED-EXPRESS": f"(novel; $r={radius:g}$)",
            "WEIGHTED-EXPRESS": (
                f"(novel; $\\lambda={args.lambda_value:g}$)"
            ),
        },
    )

    exported_summary = summary_for_export(
        metrics,
        sources=sources,
        lambda_value=args.lambda_value,
        radius=radius,
    )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exported_summary.to_csv(summary_csv, index=False)
    script_path = write_recreation_script(
        output_path,
        suite_dir=suite_dir,
        summary_csv=summary_csv,
        lambda_value=args.lambda_value,
    )

    print(f"Wrote summary plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote recreation script to {script_path}")


if __name__ == "__main__":
    main()
