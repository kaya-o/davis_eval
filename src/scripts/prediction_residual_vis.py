import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt


DATA_PATH = PROJECT_ROOT / "data" / "davis_other_data_models.csv"
VIS_DIR = PROJECT_ROOT / "data" / "vis"
DEFAULT_OUTPUT_PATH = VIS_DIR / "prediction_residual_correlations.png"
DEFAULT_DISTANCE_OUTPUT_PATH = VIS_DIR / "prediction_distance_from_mean_residual_correlations.png"
PREDICTION_COLUMNS = ("muhat_1", "muhat_2", "muhat_3")


def can_correlate(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (
        x.size >= 2
        and y.size >= 2
        and np.isfinite(x).all()
        and np.isfinite(y).all()
        and np.std(x) > 0
        and np.std(y) > 0
    )


def pearson_r(x, y):
    if not can_correlate(x, y):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_rho(x, y):
    if not can_correlate(x, y):
        return np.nan
    ranked_x = pd.Series(x).rank(method="average").to_numpy()
    ranked_y = pd.Series(y).rank(method="average").to_numpy()
    return pearson_r(ranked_x, ranked_y)


def load_prediction_residuals(data_path=DATA_PATH):
    data_df = pd.read_csv(data_path)
    labels = data_df["Label"].to_numpy(dtype=float)
    label_mean = float(np.mean(labels))

    rows = {}
    for column in PREDICTION_COLUMNS:
        predictions = data_df[column].to_numpy(dtype=float)
        rows[column] = {
            "prediction": predictions,
            "distance_from_label_mean": np.abs(predictions - label_mean),
            "residual": np.abs(labels - predictions),
            "label_mean": label_mean,
        }

    return rows


def plot_prediction_residuals(data, output_path=DEFAULT_OUTPUT_PATH):
    all_predictions = np.concatenate([values["prediction"] for values in data.values()])
    all_residuals = np.concatenate([values["residual"] for values in data.values()])

    x_min, x_max = np.nanmin(all_predictions), np.nanmax(all_predictions)
    y_min, y_max = np.nanmin(all_residuals), np.nanmax(all_residuals)
    x_pad = 0.04 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.04 * (y_max - y_min) if y_max > y_min else 1.0

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 11.2), sharex=True, sharey=True)

    for ax, (model_name, values) in zip(axes, data.items()):
        predictions = values["prediction"]
        residuals = values["residual"]
        pearson = pearson_r(predictions, residuals)
        spearman = spearman_rho(predictions, residuals)

        ax.scatter(
            predictions,
            residuals,
            s=9,
            alpha=0.28,
            edgecolors="none",
            color="tab:blue",
        )
        ax.set_title(model_name)
        ax.set_ylabel(r"Absolute residual $|Y - \hat{\mu}_i|$")
        ax.grid(alpha=0.24)
        ax.text(
            0.98,
            0.94,
            f"Pearson r = {pearson:.3f}\nSpearman rho = {spearman:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 2.5,
            },
        )

    axes[0].set_xlim(x_min - x_pad, x_max + x_pad)
    axes[0].set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
    axes[-1].set_xlabel("Model prediction")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_prediction_distance_residuals(data, output_path=DEFAULT_DISTANCE_OUTPUT_PATH):
    all_distances = np.concatenate([
        values["distance_from_label_mean"] for values in data.values()
    ])
    all_residuals = np.concatenate([values["residual"] for values in data.values()])
    label_mean = next(iter(data.values()))["label_mean"]

    x_min, x_max = np.nanmin(all_distances), np.nanmax(all_distances)
    y_min, y_max = np.nanmin(all_residuals), np.nanmax(all_residuals)
    x_pad = 0.04 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.04 * (y_max - y_min) if y_max > y_min else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.9), sharex=True, sharey=True)

    for ax, (model_name, values) in zip(axes, data.items()):
        distances = values["distance_from_label_mean"]
        residuals = values["residual"]
        pearson = pearson_r(distances, residuals)
        spearman = spearman_rho(distances, residuals)

        ax.scatter(
            distances,
            residuals,
            s=9,
            alpha=0.28,
            edgecolors="none",
            color="tab:blue",
        )
        ax.set_title(model_name)
        ax.set_xlabel(r"$|\hat{\mu}_i - \bar{Y}|$")
        ax.grid(alpha=0.24)
        ax.text(
            0.5,
            -0.23,
            f"Pearson r = {pearson:.3f}\nSpearman rho = {spearman:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

    axes[0].set_ylabel(r"Absolute residual $|Y - \hat{\mu}_i|$")
    axes[0].set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
    axes[0].set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
    fig.text(
        0.5,
        0.035,
        rf"True label mean: $\bar{{Y}} = {label_mean:.3f}$",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.15, 1, 1))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    data = load_prediction_residuals()
    output_path = plot_prediction_residuals(data)
    distance_output_path = plot_prediction_distance_residuals(data)
    print(f"Wrote prediction residual plot to {output_path}")
    print(f"Wrote prediction distance residual plot to {distance_output_path}")


if __name__ == "__main__":
    main()
