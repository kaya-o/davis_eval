import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.pipeline import N_OFF, N_ON, generate_data

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt


VIS_DIR = PROJECT_ROOT / "data" / "vis"
DATA_PATH = PROJECT_ROOT / "data" / "davis_other_data_models.csv"
DEFAULT_OUTPUT_PATH = VIS_DIR / "all_davis_muhat2_sorted_prediction_and_residual.png"
DEFAULT_SEED = 42


def full_davis_data(data_path=DATA_PATH):
    data_df = pd.read_csv(data_path)
    return {
        "y": data_df["Label"].to_numpy(dtype=float),
        "muhat_1": data_df["muhat_1"].to_numpy(dtype=float),
        "muhat_2": data_df["muhat_2"].to_numpy(dtype=float),
        "muhat_3": data_df["muhat_3"].to_numpy(dtype=float),
    }


def online_split(seed=DEFAULT_SEED, n_off=N_OFF, n_on=N_ON):
    y, muhat_1, muhat_2, muhat_3 = generate_data(n_off + n_on, seed)
    return {
        "y": y[n_off:],
        "muhat_1": muhat_1[n_off:],
        "muhat_2": muhat_2[n_off:],
        "muhat_3": muhat_3[n_off:],
    }


def plot_sorted_muhat2_and_residual(data, output_path=DEFAULT_OUTPUT_PATH):
    y = np.asarray(data["y"], dtype=float)
    muhat_1 = np.asarray(data["muhat_1"], dtype=float)
    muhat_2 = np.asarray(data["muhat_2"], dtype=float)

    sort_idx = np.argsort(muhat_2)
    sorted_muhat_2 = muhat_2[sort_idx]
    sorted_residual = np.abs(y[sort_idx] - muhat_1[sort_idx])
    x = np.arange(sorted_muhat_2.size)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)

    axes[0].plot(x, sorted_muhat_2, color="tab:blue", linewidth=1.8)
    axes[0].scatter(x, sorted_muhat_2, s=14, alpha=0.55, edgecolors="none", color="tab:blue")
    reference_lines = [
        ("Mean", np.mean(muhat_2), "tab:red", "-"),
        ("Median", np.median(muhat_2), "black", "--"),
        ("1%", np.quantile(muhat_2, 0.01), "tab:green", ":"),
        ("5%", np.quantile(muhat_2, 0.05), "tab:purple", ":"),
        ("10%", np.quantile(muhat_2, 0.10), "tab:orange", ":"),
    ]
    for label, value, color, linestyle in reference_lines:
        axes[0].axhline(
            value,
            color=color,
            linestyle=linestyle,
            linewidth=1.3,
            label=f"{label} = {value:.3f}",
        )
    axes[0].set_ylabel(r"$\hat{\mu}_2$")
    axes[0].grid(alpha=0.24)
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)

    axes[1].scatter(
        x,
        sorted_residual,
        s=16,
        alpha=0.65,
        edgecolors="none",
        color="tab:orange",
    )
    axes[1].set_xlabel(r"Datapoints sorted by $\hat{\mu}_2$")
    axes[1].set_ylabel(r"$|Y - \hat{\mu}_1|$")
    axes[1].grid(alpha=0.24)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot muhat_2 predictions and signed residuals in the same sorted order."
    )
    parser.add_argument("--online-split", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-off", type=int, default=N_OFF)
    parser.add_argument("--n-on", type=int, default=N_ON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    data = (
        online_split(seed=args.seed, n_off=args.n_off, n_on=args.n_on)
        if args.online_split
        else full_davis_data()
    )
    output_path = plot_sorted_muhat2_and_residual(data, output_path=args.output)
    print(f"Wrote muhat_2 residual plot to {output_path}")


if __name__ == "__main__":
    main()
