import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "davis_other_data_models.csv"
VIS_DIR = PROJECT_ROOT / "data" / "vis"
DEFAULT_PLOT_PATH = VIS_DIR / "davis_affinity_scores_scatter.png"
DEFAULT_SORTED_PLOT_PATH = VIS_DIR / "davis_affinity_scores_sorted_scatter.png"

FONT_CACHE_DIR = Path(tempfile.gettempdir()) / "davis_eval_matplotlib_cache"
MPL_CACHE_DIR = FONT_CACHE_DIR / "matplotlib"
XDG_CACHE_DIR = FONT_CACHE_DIR / "xdg"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def infer_score_column(data_df):
    for column in ("Label", "Affinity"):
        if column in data_df.columns:
            return column
    raise ValueError("Expected an affinity score column named 'Label' or 'Affinity'.")


def plot_affinity_scores(
    data_path=DATA_PATH,
    save_path=DEFAULT_PLOT_PATH,
    score_column=None,
    sort_scores=False,
):
    data_df = pd.read_csv(data_path)
    score_column = score_column or infer_score_column(data_df)
    scores = data_df[score_column].to_numpy(dtype=float)
    if sort_scores:
        scores = np.sort(scores)
        x = np.arange(1, scores.size + 1)
    else:
        x = np.arange(scores.size)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    marker_size = 6 if sort_scores else 10
    score_mean = float(np.mean(scores))
    score_median = float(np.median(scores))
    ax.scatter(
        x,
        scores,
        s=marker_size,
        alpha=0.35,
        edgecolors="none",
        rasterized=True,
    )
    ax.axhline(
        score_mean,
        color="tab:red",
        linewidth=1.5,
        label=rf"Mean $= {score_mean:.3f}$",
    )
    ax.axhline(
        score_median,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=rf"Median $= {score_median:.3f}$",
    )

    xlabel = (
        "Drug–target pairs (ascending affinity score)"
        if sort_scores
        else "Datapoint index"
    )
    if not sort_scores:
        ax.set_title("DAVIS Affinity Scores")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(r"Affinity score, $Y$", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    if sort_scores:
        rank_ticks = list(range(5_000, scores.size, 5_000)) + [scores.size]
        right_padding = max(500, 0.025 * scores.size)
        ax.set_xlim(0.5, scores.size + right_padding)
        ax.set_xticks(rank_ticks)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DAVIS affinity scores as a scatterplot.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument(
        "--score-column",
        default=None,
        help="Affinity score column. Defaults to Label if present, otherwise Affinity.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort affinity scores before plotting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = plot_affinity_scores(
        data_path=args.data_path,
        save_path=args.save_path,
        score_column=args.score_column,
        sort_scores=args.sort,
    )
    print(f"Wrote affinity score scatterplot to {path}")
