from argparse import ArgumentParser
import os
from pathlib import Path

import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt


STRATEGIES = (
    ("CAP", "#1f77b4"),
    ("EXPRESS", "#d62728"),
)
DEFAULT_OUTPUT_NAME = "cap_express_calibration_size_all_selection_events.png"


def load_events(raw_path, chunksize=500_000):
    strategy_names = {strategy for strategy, _ in STRATEGIES}
    frames = []
    for chunk in pd.read_csv(
        raw_path,
        usecols=["run", "t", "strategy", "n_calibration"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["strategy"].isin(strategy_names)].copy()
        if chunk.empty:
            continue
        chunk["run"] = pd.to_numeric(chunk["run"], errors="coerce")
        chunk["t"] = pd.to_numeric(chunk["t"], errors="coerce")
        chunk["n_calibration"] = pd.to_numeric(
            chunk["n_calibration"], errors="coerce"
        )
        frames.append(chunk.dropna(subset=["run", "t", "n_calibration"]))

    if not frames:
        raise ValueError(f"No CAP or EXPRESS rows found in {raw_path}")

    events = pd.concat(frames, ignore_index=True)
    missing = strategy_names - set(events["strategy"].unique())
    if missing:
        raise ValueError(f"Missing strategies in {raw_path}: {sorted(missing)}")
    return events


def plot_events(events, output_path, title=None):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        sharex=True,
        sharey=True,
    )

    for ax, (strategy, color) in zip(axes, STRATEGIES):
        panel = events[events["strategy"].eq(strategy)]
        ax.scatter(
            panel["t"],
            panel["n_calibration"],
            s=1,
            alpha=0.06,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_title(strategy, fontsize=15)
        ax.set_xlabel("Online time $t$", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.2)
        ax.text(
            0.98,
            0.97,
            f"{len(panel):,} events",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )

    axes[0].set_ylabel("Calibration set size", fontsize=13)
    if title:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(
        description=(
            "Plot one point per selected event for CAP and EXPRESS calibration sizes."
        )
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Result directory containing raw_selected_events.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults inside the result directory.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    raw_path = args.result_dir / "raw_selected_events.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    output_path = args.output or args.result_dir / DEFAULT_OUTPUT_NAME
    events = load_events(raw_path, chunksize=args.chunksize)
    plot_events(events, output_path, title=args.title)

    print(f"Wrote plot to {output_path}")
    for strategy, _ in STRATEGIES:
        count = int(events["strategy"].eq(strategy).sum())
        print(f"{strategy}: {count:,} events")


if __name__ == "__main__":
    main()
