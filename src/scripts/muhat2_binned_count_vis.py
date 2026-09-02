import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "davis_other_data_models.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "vis" / "all_davis_muhat2_binned_counts.png"

FONT_CACHE_DIR = Path(tempfile.gettempdir()) / "davis_eval_matplotlib_cache"
MPL_CACHE_DIR = FONT_CACHE_DIR / "matplotlib"
XDG_CACHE_DIR = FONT_CACHE_DIR / "xdg"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


BIN_EDGES = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.inf])
BIN_LABELS = [
    r"$[5,6)$",
    r"$[6,7)$",
    r"$[7,8)$",
    r"$[8,9)$",
    r"$[9,10)$",
    r"$\geq 10$",
]


def count_muhat2_bins(data_path=DATA_PATH):
    data = pd.read_csv(data_path)
    if "muhat_2" not in data.columns:
        raise ValueError(f"Expected column 'muhat_2' in {data_path}.")

    values = pd.to_numeric(data["muhat_2"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Column 'muhat_2' contains non-finite values.")

    counts, _ = np.histogram(values, bins=BIN_EDGES)
    below_five = int(np.count_nonzero(values < BIN_EDGES[0]))
    return counts, below_five, values.size


def plot_muhat2_binned_counts(data_path=DATA_PATH, output_path=OUTPUT_PATH):
    counts, below_five, total = count_muhat2_bins(data_path)
    x = np.arange(len(BIN_LABELS))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(
        x,
        counts,
        width=0.72,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    ax.set_xticks(x, BIN_LABELS)
    ax.set_xlabel(r"Predicted affinity score, $\hat{\mu}_2$", fontsize=12)
    ax.set_ylabel("Number of drug–target pairs", fontsize=12)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count in zip(bars, counts):
        ax.annotate(
            f"{int(count):,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    ax.set_ylim(0, float(counts.max()) * 1.10)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path, counts, below_five, total


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot counts of DAVIS muhat_2 predictions in fixed score bins."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path, counts, below_five, total = plot_muhat2_binned_counts(
        data_path=args.data_path,
        output_path=args.output,
    )
    print(f"Wrote binned muhat_2 count plot to {output_path}")
    print(f"Bin counts: {dict(zip(BIN_LABELS, counts.tolist()))}")
    print(f"Scores below 5 (not plotted): {below_five:,} of {total:,}")


if __name__ == "__main__":
    main()
