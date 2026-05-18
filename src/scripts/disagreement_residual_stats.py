import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency fallback
    stats = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "davis_other_data_models.csv"
EXP_DIR = PROJECT_ROOT / "data" / "exp"
PREDICTION_COLUMNS = ("muhat_1", "muhat_2", "muhat_3")
DEFAULT_THRESHOLDS = (0.1, 0.15, 0.3)
DEFAULT_N_OFF = 200
DEFAULT_N_ON = 22500
DEFAULT_SEED = 42


def disagreement_metrics(predictions):
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] < 2:
        raise ValueError("predictions must be a 2d array with at least two model columns.")

    return {
        "variance": np.var(predictions, axis=1),
        "std": np.std(predictions, axis=1),
        "range": np.max(predictions, axis=1) - np.min(predictions, axis=1),
        "mean_pairwise_abs": mean_pairwise_abs_difference(predictions),
    }


def mean_pairwise_abs_difference(predictions):
    n_models = predictions.shape[1]
    pairwise_diffs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pairwise_diffs.append(np.abs(predictions[:, i] - predictions[:, j]))
    return np.mean(np.stack(pairwise_diffs, axis=0), axis=0)


def residual_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    primary_prediction = predictions[:, 0]
    ensemble_mean = np.mean(predictions, axis=1)

    return {
        "abs_resid_muhat_1": np.abs(labels - primary_prediction),
        "abs_resid_ensemble_mean": np.abs(labels - ensemble_mean),
        "signed_resid_muhat_1": labels - primary_prediction,
        "squared_resid_muhat_1": (labels - primary_prediction) ** 2,
    }


def correlation_rows(disagreements, residuals):
    rows = []
    for disagreement_name, disagreement_values in disagreements.items():
        for residual_name, residual_values in residuals.items():
            pearson_r, pearson_p = pearson(disagreement_values, residual_values)
            spearman_rho, spearman_p = spearman(disagreement_values, residual_values)
            kendall_tau, kendall_p = kendall(disagreement_values, residual_values)
            rows.append(
                {
                    "disagreement": disagreement_name,
                    "residual": residual_name,
                    "n": len(disagreement_values),
                    "pearson_r": pearson_r,
                    "pearson_r_squared": pearson_r**2 if np.isfinite(pearson_r) else np.nan,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "kendall_tau": kendall_tau,
                    "kendall_p": kendall_p,
                }
            )
    return pd.DataFrame(rows)


def pearson(x, y):
    if not can_correlate(x, y):
        return np.nan, np.nan
    if stats is not None:
        result = stats.pearsonr(x, y)
        return result.statistic, result.pvalue
    return np.corrcoef(x, y)[0, 1], np.nan


def spearman(x, y):
    if not can_correlate(x, y):
        return np.nan, np.nan
    if stats is not None:
        result = stats.spearmanr(x, y)
        return result.statistic, result.pvalue
    ranked_x = pd.Series(x).rank(method="average").to_numpy()
    ranked_y = pd.Series(y).rank(method="average").to_numpy()
    return np.corrcoef(ranked_x, ranked_y)[0, 1], np.nan


def kendall(x, y):
    if not can_correlate(x, y):
        return np.nan, np.nan
    if stats is not None:
        result = stats.kendalltau(x, y)
        return result.statistic, result.pvalue
    return np.nan, np.nan


def can_correlate(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return len(x) >= 2 and len(y) >= 2 and np.std(x) > 0 and np.std(y) > 0


def decile_rows(variance_score, std_score, residuals, n_bins=10):
    ranked_score = pd.Series(variance_score).rank(method="first")
    decile_df = pd.DataFrame(
        {
            "variance": variance_score,
            "std": std_score,
            "abs_resid_muhat_1": residuals["abs_resid_muhat_1"],
            "abs_resid_ensemble_mean": residuals["abs_resid_ensemble_mean"],
        }
    )
    decile_df["decile"] = pd.qcut(ranked_score, n_bins, labels=False) + 1

    return (
        decile_df.groupby("decile")
        .agg(
            n=("variance", "size"),
            variance_min=("variance", "min"),
            variance_max=("variance", "max"),
            variance_mean=("variance", "mean"),
            std_mean=("std", "mean"),
            abs_resid_muhat_1_mean=("abs_resid_muhat_1", "mean"),
            abs_resid_muhat_1_median=("abs_resid_muhat_1", "median"),
            abs_resid_ensemble_mean_mean=("abs_resid_ensemble_mean", "mean"),
            abs_resid_ensemble_mean_median=("abs_resid_ensemble_mean", "median"),
        )
        .reset_index()
    )


def threshold_rows(variance_score, residuals, thresholds):
    rows = []
    for threshold in thresholds:
        selected = variance_score > threshold
        unselected = ~selected
        row = {
            "threshold": threshold,
            "comparison": "variance_gt_threshold",
            "n": len(variance_score),
            "selected": int(np.sum(selected)),
            "selected_fraction": float(np.mean(selected)),
        }

        for residual_name in ("abs_resid_muhat_1", "abs_resid_ensemble_mean"):
            selected_values = residuals[residual_name][selected]
            unselected_values = residuals[residual_name][unselected]
            row.update(residual_threshold_stats(residual_name, selected_values, unselected_values))
        rows.append(row)
    return pd.DataFrame(rows)


def residual_threshold_stats(residual_name, selected_values, unselected_values):
    selected_mean = mean_or_nan(selected_values)
    unselected_mean = mean_or_nan(unselected_values)
    selected_median = median_or_nan(selected_values)
    unselected_median = median_or_nan(unselected_values)
    mann_whitney_p = np.nan

    if stats is not None and len(selected_values) > 0 and len(unselected_values) > 0:
        result = stats.mannwhitneyu(selected_values, unselected_values, alternative="greater")
        mann_whitney_p = result.pvalue

    return {
        f"{residual_name}_selected_mean": selected_mean,
        f"{residual_name}_unselected_mean": unselected_mean,
        f"{residual_name}_selected_minus_unselected_mean": selected_mean - unselected_mean,
        f"{residual_name}_selected_to_unselected_mean_ratio": safe_ratio(
            selected_mean, unselected_mean
        ),
        f"{residual_name}_selected_median": selected_median,
        f"{residual_name}_unselected_median": unselected_median,
        f"{residual_name}_mann_whitney_greater_p": mann_whitney_p,
    }


def pipeline_segment_rows(labels, predictions, seed, n_off, n_on, threshold):
    n_samples = n_off + n_on
    if n_samples > len(labels):
        raise ValueError(f"Requested n_off + n_on = {n_samples}, but only {len(labels)} rows exist.")

    rng = np.random.RandomState(seed)
    selected_indices = rng.permutation(len(labels))[:n_samples]
    selected_predictions = predictions[selected_indices]
    selected_labels = labels[selected_indices]
    variance_score = disagreement_metrics(selected_predictions)["variance"]
    abs_resid = np.abs(selected_labels - selected_predictions[:, 0])

    segments = {
        "all_sampled": np.ones(n_samples, dtype=bool),
        "offline": np.arange(n_samples) < n_off,
        "online": np.arange(n_samples) >= n_off,
        "online_selected_initial_threshold": np.concatenate(
            [np.zeros(n_off, dtype=bool), variance_score[n_off:] > threshold]
        ),
    }

    rows = []
    for segment_name, mask in segments.items():
        segment_variance = variance_score[mask]
        segment_residual = abs_resid[mask]
        pearson_r, pearson_p = pearson(segment_variance, segment_residual)
        spearman_rho, spearman_p = spearman(segment_variance, segment_residual)
        rows.append(
            {
                "segment": segment_name,
                "seed": seed,
                "n_off": n_off,
                "n_on": n_on,
                "threshold": threshold,
                "n": int(np.sum(mask)),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "abs_resid_muhat_1_mean": mean_or_nan(segment_residual),
                "abs_resid_muhat_1_median": median_or_nan(segment_residual),
                "variance_mean": mean_or_nan(segment_variance),
                "variance_median": median_or_nan(segment_variance),
            }
        )
    return pd.DataFrame(rows)


def build_summary(data_df, correlations_df, deciles_df, thresholds_df, pipeline_segments_df):
    variance_abs_corr = select_single(
        correlations_df,
        disagreement="variance",
        residual="abs_resid_muhat_1",
    )
    ensemble_abs_corr = select_single(
        correlations_df,
        disagreement="variance",
        residual="abs_resid_ensemble_mean",
    )
    signed_corr = select_single(
        correlations_df,
        disagreement="variance",
        residual="signed_resid_muhat_1",
    )

    bottom_decile = deciles_df.loc[deciles_df["decile"].idxmin()]
    top_decile = deciles_df.loc[deciles_df["decile"].idxmax()]
    pipeline_threshold = thresholds_df.loc[
        np.isclose(thresholds_df["threshold"], 0.15)
    ]
    if pipeline_threshold.empty:
        pipeline_threshold = thresholds_df.iloc[[0]]

    return {
        "n_rows": int(len(data_df)),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "primary_prediction_column": "muhat_1",
        "pipeline_disagreement_score": "variance",
        "pipeline_residual": "abs_resid_muhat_1",
        "variance_vs_abs_resid_muhat_1": row_to_plain_dict(variance_abs_corr),
        "variance_vs_abs_resid_ensemble_mean": row_to_plain_dict(ensemble_abs_corr),
        "variance_vs_signed_resid_muhat_1": row_to_plain_dict(signed_corr),
        "top_vs_bottom_decile_abs_resid_muhat_1_mean_ratio": safe_ratio(
            top_decile["abs_resid_muhat_1_mean"],
            bottom_decile["abs_resid_muhat_1_mean"],
        ),
        "top_vs_bottom_decile_abs_resid_ensemble_mean_mean_ratio": safe_ratio(
            top_decile["abs_resid_ensemble_mean_mean"],
            bottom_decile["abs_resid_ensemble_mean_mean"],
        ),
        "threshold_summary": row_to_plain_dict(pipeline_threshold.iloc[0]),
        "pipeline_seed_segments": [
            row_to_plain_dict(row) for _, row in pipeline_segments_df.iterrows()
        ],
    }


def select_single(df, **conditions):
    mask = np.ones(len(df), dtype=bool)
    for column, value in conditions.items():
        mask &= df[column].eq(value).to_numpy()
    matches = df.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"Expected one match for {conditions}, found {len(matches)}.")
    return matches.iloc[0]


def row_to_plain_dict(row):
    return {key: plain_value(value) for key, value in row.to_dict().items()}


def plain_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def mean_or_nan(values):
    return float(np.mean(values)) if len(values) else np.nan


def median_or_nan(values):
    return float(np.median(values)) if len(values) else np.nan


def safe_ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def write_outputs(
    data_path=DATA_PATH,
    output_dir=EXP_DIR,
    thresholds=DEFAULT_THRESHOLDS,
    seed=DEFAULT_SEED,
    n_off=DEFAULT_N_OFF,
    n_on=DEFAULT_N_ON,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_df = pd.read_csv(data_path)
    labels = data_df["Label"].to_numpy(dtype=float)
    predictions = data_df.loc[:, PREDICTION_COLUMNS].to_numpy(dtype=float)

    disagreements = disagreement_metrics(predictions)
    residuals = residual_metrics(labels, predictions)

    correlations_df = correlation_rows(disagreements, residuals)
    deciles_df = decile_rows(disagreements["variance"], disagreements["std"], residuals)
    thresholds_df = threshold_rows(disagreements["variance"], residuals, thresholds)
    pipeline_segments_df = pipeline_segment_rows(
        labels,
        predictions,
        seed=seed,
        n_off=n_off,
        n_on=n_on,
        threshold=0.15,
    )
    summary = build_summary(
        data_df,
        correlations_df,
        deciles_df,
        thresholds_df,
        pipeline_segments_df,
    )

    output_paths = {
        "correlations": output_dir / "disagreement_residual_correlations.csv",
        "deciles": output_dir / "disagreement_residual_deciles.csv",
        "thresholds": output_dir / "disagreement_residual_thresholds.csv",
        "pipeline_segments": output_dir / "disagreement_residual_pipeline_segments.csv",
        "summary": output_dir / "disagreement_residual_summary.json",
    }

    correlations_df.to_csv(output_paths["correlations"], index=False)
    deciles_df.to_csv(output_paths["deciles"], index=False)
    thresholds_df.to_csv(output_paths["thresholds"], index=False)
    pipeline_segments_df.to_csv(output_paths["pipeline_segments"], index=False)
    output_paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")

    return output_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute DAVIS ensemble-disagreement vs residual statistics."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR)
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        dest="thresholds",
        help="Variance threshold to compare. Repeat for multiple thresholds.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-off", type=int, default=DEFAULT_N_OFF)
    parser.add_argument("--n-on", type=int, default=DEFAULT_N_ON)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    thresholds = args.thresholds if args.thresholds is not None else DEFAULT_THRESHOLDS
    paths = write_outputs(
        data_path=args.data_path,
        output_dir=args.output_dir,
        thresholds=thresholds,
        seed=args.seed,
        n_off=args.n_off,
        n_on=args.n_on,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
