import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "davis_other_data_models.csv"
VIS_DIR = PROJECT_ROOT / "data" / "vis"
DEFAULT_PLOT_PATH = VIS_DIR / "muhat_vs_y.png"


def generate_data(n: int, seed: int):
    """
    FROM ZHENG AND JIN 2025
    Generate online data of size n from pre-generated DAVIS CSV:
    - Load the pre-generated CSV 'davis_other_data.csv'.
    - Randomly select n samples using the given seed for reproducibility.
    """
    # Load the pre-generated CSV
    data_df = pd.read_csv(DATA_PATH)
    
    # Check if n exceeds available samples
    print(f"Available samples in CSV: {len(data_df)}")
    if n > len(data_df):
        raise ValueError(f"Requested n={n} exceeds available samples in CSV ({len(data_df)})")
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Randomly select n indices
    select_indices = np.random.permutation(len(data_df))[:n]
    
    # Select the rows
    selected_df = data_df.iloc[select_indices]
    
    Y = np.array(selected_df['Label'])
    muhat_1 = np.array(selected_df['muhat_1'])
    muhat_2 = np.array(selected_df['muhat_2'])
    muhat_3 = np.array(selected_df['muhat_3'])
    
    return Y, muhat_1, muhat_2, muhat_3


def plot_muhat_vs_y(
    Y: np.ndarray,
    muhat_1: np.ndarray,
    muhat_2: np.ndarray,
    muhat_3: np.ndarray,
    save_path: Optional[Union[str, Path]] = DEFAULT_PLOT_PATH,
    show: bool = True,
    point_alpha: float = 0.35,
):
    """
    Plot true labels against each model's predictions.

    Each subplot shows:
    - scatter points for one muhat_i vs. Y
    - a fitted linear trend line
    - the ideal y=x line for reference
    - r, R2, RMSE, and MAE metrics
    """
    predictions = {
        "muhat_1": np.asarray(muhat_1),
        "muhat_2": np.asarray(muhat_2),
        "muhat_3": np.asarray(muhat_3),
    }
    Y = np.asarray(Y)

    if Y.size < 2:
        raise ValueError("At least two samples are required to fit a linear trend line.")
    for model_name, muhat in predictions.items():
        if muhat.shape != Y.shape:
            raise ValueError(f"{model_name} must have the same shape as Y.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharex=True, sharey=True)

    y_min = min(Y.min(), *(pred.min() for pred in predictions.values()))
    y_max = max(Y.max(), *(pred.max() for pred in predictions.values()))
    padding = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    axis_min = y_min - padding
    axis_max = y_max + padding
    reference_x = np.linspace(axis_min, axis_max, 100)

    for ax, (model_name, muhat) in zip(axes, predictions.items()):
        ax.scatter(Y, muhat, alpha=point_alpha, s=20, edgecolors="none")

        slope, intercept = np.polyfit(Y, muhat, deg=1)
        trend_y = slope * reference_x + intercept
        correlation = np.corrcoef(Y, muhat)[0, 1]
        residuals = Y - muhat
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        ax.plot(reference_x, trend_y, color="tab:red", linewidth=2, label="Linear fit")
        ax.plot(reference_x, reference_x, color="black", linestyle="--", linewidth=1, label="Ideal y=x")

        metrics_text = (
            f"r = {correlation:.3f}\n"
            f"R2 = {r_squared:.3f}\n"
            f"RMSE = {rmse:.3f}\n"
            f"MAE = {mae:.3f}"
        )

        ax.set_title(f"{model_name} vs Y")
        ax.set_xlabel("True label Y")
        ax.set_ylabel(f"Prediction {model_name}")
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.grid(alpha=0.25)
        ax.legend()
        ax.text(
            0.98,
            0.02,
            metrics_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    Y, muhat_1, muhat_2, muhat_3 = generate_data(24042, 42)
    plot_muhat_vs_y(Y, muhat_1, muhat_2, muhat_3)
