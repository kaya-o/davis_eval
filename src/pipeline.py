import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.conformal import Conformal, dump_experiment_results
except ModuleNotFoundError:
    from conformal import Conformal, dump_experiment_results

N_OFF = 500
N_ON = 20000
DEFAULT_DATA_PATH = "data/davis_other_data_models.csv"
DEFAULT_CONFIG_PATH = Path("configs/davis_experiment.json")
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_STRATEGIES = ["FULL", "S-FULL", "S-FIX", "ADA", "EXPRESS", "K-EXPRESS", "EXPRESS-M"]
DEFAULT_CONFIG = {
    "experiment_name": "davis_muhat2_moving_window",
    "n_runs": 10,
    "seed": 42,
    "data": {
        "path": DEFAULT_DATA_PATH,
        "n_off": N_OFF,
        "n_on": N_ON,
        "label_column": "Label",
    },
    "selection": {
        "score_column": "muhat_2",
        "tau_1": 5.0,
        "tau_0": 4000,
        "window_width": 0.5,
        "tau_tail": 6.25,
    },
    "prediction": {
        "point_prediction_column": "muhat_1",
    },
    "conformal": {
        "alpha": 0.4,
        "randomized_calibration": True,
        "k_express": 7500,
        "relaxed_express_target_size": None,
        "relaxed_express_rank_delta": None,
        "distance_express_max_distance": 0.02,
        "distance_express_debug": False,
        "express_distance": "endpoint",
        "weighted_express_lambda": 1.0,
        "weighted_express_distance_normalization": "rank",
        "weighted_express_max_distance": None,
        "weighted_express_max_rank_pct": 0.05,
        "weighted_express_debug": False,
        "adaptive_weighted_express_low_distance_threshold": 0.01,
        "adaptive_weighted_express_target_low_distance_count": 6,
        "adaptive_weighted_express_lambda_min": 35.0,
        "adaptive_weighted_express_lambda_max": 300.0,
        "adaptive_weighted_express_max_distance": 1.0,
        "adaptive_weighted_express_debug": False,
        "weighted_neighborhood_express_lambda": 1.0,
        "weighted_neighborhood_express_distance_normalization": "rank",
        "weighted_neighborhood_express_max_distance": None,
        "weighted_neighborhood_express_max_rank_pct": 0.05,
        "weighted_neighborhood_express_max_neighbors": 200,
        "weighted_neighborhood_express_debug": False,
    },
    "strategies": DEFAULT_STRATEGIES,
}


def deep_merge(base, overrides):
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path):
    config_path = Path(config_path)
    with config_path.open() as f:
        user_config = json.load(f)
    return deep_merge(DEFAULT_CONFIG, user_config)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DAVIS conformal selection experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a JSON experiment config.",
    )
    parser.add_argument(
        "--suite-name",
        default=None,
        help="Optional suite directory name under results/.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory suffix. Requires --suite-name and is prefixed with a timestamp.",
    )
    return parser.parse_args()


def validate_output_name(name, label):
    if name is None:
        return
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{label} must be a simple directory name, got {name!r}")


def variance_score(muhat_1, muhat_2, muhat_3):
    muhat_1 = np.asarray(muhat_1)
    muhat_2 = np.asarray(muhat_2)
    muhat_3 = np.asarray(muhat_3)

    if not (muhat_1.shape == muhat_2.shape == muhat_3.shape):
        raise ValueError("muhat_1, muhat_2, and muhat_3 must have the same shape.")

    return np.var(np.stack([muhat_1, muhat_2, muhat_3], axis=0), axis=0)


def sample_data_frame(n: int, seed: int, data_path=DEFAULT_DATA_PATH):
    """
    FROM ZHENG AND JIN 2025
    Generate online data of size n from pre-generated DAVIS CSV:
    - Load the pre-generated CSV 'davis_other_data.csv'.
    - Randomly select n samples using the given seed for reproducibility.
    """
    data_df = pd.read_csv(data_path)
    
    # Check if n exceeds available samples
    #print(f"Available samples in CSV: {len(data_df)}")
    if n > len(data_df):
        raise ValueError(f"Requested n={n} exceeds available samples in CSV ({len(data_df)})")
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Randomly select n indices
    select_indices = np.random.permutation(len(data_df))[:n]
    
    # Select the rows
    return data_df.iloc[select_indices].reset_index(drop=True)


def generate_data(n: int, seed: int, data_path=DEFAULT_DATA_PATH):
    selected_df = sample_data_frame(n, seed, data_path=data_path)
    
    Y = np.array(selected_df['Label'])
    muhat_1 = np.array(selected_df['muhat_1'])
    muhat_2 = np.array(selected_df['muhat_2'])
    muhat_3 = np.array(selected_df['muhat_3'])
    
    return Y, muhat_1, muhat_2, muhat_3


def values_from_column(data_df, column_name):
    if column_name in data_df.columns:
        return np.asarray(data_df[column_name])

    normalized = column_name.lower()
    if normalized in {"ensemble_variance", "variance", "variance_score"}:
        return variance_score(data_df["muhat_1"], data_df["muhat_2"], data_df["muhat_3"])

    raise ValueError(f"Unknown column or score expression: {column_name}")


def run_experiment(config, config_path=None, output_root=DEFAULT_RESULTS_ROOT, run_name=None):
    data_config = config["data"]
    selection_config = config["selection"]
    prediction_config = config["prediction"]
    conformal_config = config["conformal"]

    n_off = int(data_config["n_off"])
    n_on = int(data_config["n_on"])
    n_samples = n_off + n_on
    seed = int(config["seed"])
    n_runs = int(config["n_runs"])
    strategies = list(config["strategies"])
    k_express = int(conformal_config["k_express"])
    relaxed_express_target_size = conformal_config.get("relaxed_express_target_size")
    if relaxed_express_target_size is not None:
        relaxed_express_target_size = int(relaxed_express_target_size)
    relaxed_express_rank_delta = conformal_config.get("relaxed_express_rank_delta")
    if relaxed_express_rank_delta is not None:
        relaxed_express_rank_delta = float(relaxed_express_rank_delta)
    express_distance = conformal_config.get("express_distance", "endpoint")
    if express_distance not in {"endpoint", "hamming"}:
        raise ValueError(
            "conformal.express_distance must be one of ['endpoint', 'hamming'], "
            f"got {express_distance!r}"
        )
    distance_express_max_distance = conformal_config.get(
        "distance_express_max_distance",
        0.02,
    )
    if distance_express_max_distance is not None:
        distance_express_max_distance = float(distance_express_max_distance)
        if distance_express_max_distance < 0 or distance_express_max_distance > 1:
            raise ValueError("distance_express_max_distance must be in [0, 1]")
    distance_express_debug = bool(conformal_config.get("distance_express_debug", False))
    weighted_express_lambda = float(conformal_config.get("weighted_express_lambda", 1.0))
    weighted_express_distance_normalization = conformal_config.get(
        "weighted_express_distance_normalization",
        "rank",
    )
    weighted_express_max_distance = conformal_config.get("weighted_express_max_distance")
    if weighted_express_max_distance is not None:
        weighted_express_max_distance = float(weighted_express_max_distance)
    weighted_express_max_rank_pct = conformal_config.get("weighted_express_max_rank_pct", 0.05)
    if weighted_express_max_rank_pct is not None:
        weighted_express_max_rank_pct = float(weighted_express_max_rank_pct)
    weighted_express_debug = bool(conformal_config.get("weighted_express_debug", False))
    adaptive_weighted_express_low_distance_threshold = conformal_config.get(
        "adaptive_weighted_express_low_distance_threshold",
        0.01,
    )
    adaptive_weighted_express_low_distance_threshold = float(
        adaptive_weighted_express_low_distance_threshold
    )
    if (
        adaptive_weighted_express_low_distance_threshold < 0
        or adaptive_weighted_express_low_distance_threshold > 1
    ):
        raise ValueError("adaptive_weighted_express_low_distance_threshold must be in [0, 1]")
    adaptive_weighted_express_target_low_distance_count = int(
        conformal_config.get("adaptive_weighted_express_target_low_distance_count", 6)
    )
    if adaptive_weighted_express_target_low_distance_count <= 0:
        raise ValueError("adaptive_weighted_express_target_low_distance_count must be positive")
    adaptive_weighted_express_lambda_min = float(
        conformal_config.get("adaptive_weighted_express_lambda_min", 35.0)
    )
    adaptive_weighted_express_lambda_max = float(
        conformal_config.get("adaptive_weighted_express_lambda_max", 300.0)
    )
    if adaptive_weighted_express_lambda_min <= 0:
        raise ValueError("adaptive_weighted_express_lambda_min must be positive")
    if adaptive_weighted_express_lambda_max <= 0:
        raise ValueError("adaptive_weighted_express_lambda_max must be positive")
    if adaptive_weighted_express_lambda_max < adaptive_weighted_express_lambda_min:
        raise ValueError("adaptive_weighted_express_lambda_max must be >= lambda_min")
    adaptive_weighted_express_max_distance = conformal_config.get(
        "adaptive_weighted_express_max_distance",
        1.0,
    )
    if adaptive_weighted_express_max_distance is not None:
        adaptive_weighted_express_max_distance = float(adaptive_weighted_express_max_distance)
        if adaptive_weighted_express_max_distance < 0 or adaptive_weighted_express_max_distance > 1:
            raise ValueError("adaptive_weighted_express_max_distance must be in [0, 1] or null")
    adaptive_weighted_express_debug = bool(
        conformal_config.get("adaptive_weighted_express_debug", False)
    )
    weighted_neighborhood_express_lambda = float(
        conformal_config.get("weighted_neighborhood_express_lambda", 1.0)
    )
    weighted_neighborhood_express_distance_normalization = conformal_config.get(
        "weighted_neighborhood_express_distance_normalization",
        "rank",
    )
    weighted_neighborhood_express_max_distance = conformal_config.get(
        "weighted_neighborhood_express_max_distance"
    )
    if weighted_neighborhood_express_max_distance is not None:
        weighted_neighborhood_express_max_distance = float(
            weighted_neighborhood_express_max_distance
        )
    weighted_neighborhood_express_max_rank_pct = conformal_config.get(
        "weighted_neighborhood_express_max_rank_pct",
        0.05,
    )
    if weighted_neighborhood_express_max_rank_pct is not None:
        weighted_neighborhood_express_max_rank_pct = float(
            weighted_neighborhood_express_max_rank_pct
        )
    weighted_neighborhood_express_max_neighbors = conformal_config.get(
        "weighted_neighborhood_express_max_neighbors",
        200,
    )
    if weighted_neighborhood_express_max_neighbors is not None:
        weighted_neighborhood_express_max_neighbors = int(
            weighted_neighborhood_express_max_neighbors
        )
        if weighted_neighborhood_express_max_neighbors <= 0:
            raise ValueError(
                "weighted_neighborhood_express_max_neighbors must be positive or null"
            )
    weighted_neighborhood_express_debug = bool(
        conformal_config.get("weighted_neighborhood_express_debug", False)
    )

    results = {
        strategy: {
            "selected": 0,
            "miscovered": 0,
            "n_calibration": [],
            "interval_length": [],
            "infinite_interval": 0,
        }
        for strategy in strategies
    }

    total_hits = 0
    raw_rows = []
    selected_point_rows = []

    for run in range(n_runs):
        #if (run % 25 == 0):
        print(f"Run: {run}")
        conformal = Conformal(
            tau_0=selection_config["tau_0"],
            tau_1=selection_config["tau_1"],
            window_width=selection_config["window_width"],
            tau_tail=selection_config["tau_tail"],
            alpha=conformal_config["alpha"],
            randomized_calibration=conformal_config["randomized_calibration"],
            random_seed=seed + run,
        )
        data_df = sample_data_frame(n_samples, seed + run, data_path=data_config["path"])
        Y = values_from_column(data_df, data_config["label_column"])
        point_predictions_all = values_from_column(data_df, prediction_config["point_prediction_column"])
        scores_all = values_from_column(data_df, selection_config["score_column"])

        conformal.scores_off = scores_all[:n_off]
        scores_on = scores_all[n_off:]

        conformal.point_predictions_off = point_predictions_all[:n_off]
        point_predictions_on = point_predictions_all[n_off:]

        conformal.y_off = Y[:n_off]
        conformal.residuals_off = np.abs(Y[:n_off] - point_predictions_all[:n_off])
        Y_on = Y[n_off:]

        for t in range(n_on):
            score_t = scores_on[t]
            point_prediction_t = point_predictions_on[t]
            y_t = Y_on[t]
            current_bounds = conformal.selection_bounds(t)
            current_lower_bound, current_upper_bound = current_bounds
            s_t = int(conformal.select_t(score_t, t))
            if s_t:
                total_hits += 1
                residual_t = np.abs(y_t - point_prediction_t)
                selected_point_rows.append({
                    "run": run,
                    "t": t,
                    "score_t": score_t,
                    "residual_t": residual_t,
                    "selection_lower_bound": current_lower_bound,
                    "selection_upper_bound": current_upper_bound,
                })

                for strategy in strategies:
                    strategy_result = conformal.evaluate_strategy(
                        strategy=strategy,
                        score_t=score_t,
                        point_prediction_t=point_prediction_t,
                        y_t=y_t,
                        current_bounds=current_bounds,
                        k=k_express,
                        relaxed_express_target_size=relaxed_express_target_size,
                        relaxed_express_rank_delta=relaxed_express_rank_delta,
                        distance_express_max_distance=distance_express_max_distance,
                        distance_express_debug=distance_express_debug,
                        weighted_express_lambda=weighted_express_lambda,
                        weighted_express_distance_normalization=(
                            weighted_express_distance_normalization
                        ),
                        weighted_express_max_distance=weighted_express_max_distance,
                        weighted_express_max_rank_pct=weighted_express_max_rank_pct,
                        weighted_express_debug=weighted_express_debug,
                        adaptive_weighted_express_low_distance_threshold=(
                            adaptive_weighted_express_low_distance_threshold
                        ),
                        adaptive_weighted_express_target_low_distance_count=(
                            adaptive_weighted_express_target_low_distance_count
                        ),
                        adaptive_weighted_express_lambda_min=(
                            adaptive_weighted_express_lambda_min
                        ),
                        adaptive_weighted_express_lambda_max=(
                            adaptive_weighted_express_lambda_max
                        ),
                        adaptive_weighted_express_max_distance=(
                            adaptive_weighted_express_max_distance
                        ),
                        adaptive_weighted_express_debug=adaptive_weighted_express_debug,
                        weighted_neighborhood_express_lambda=(
                            weighted_neighborhood_express_lambda
                        ),
                        weighted_neighborhood_express_distance_normalization=(
                            weighted_neighborhood_express_distance_normalization
                        ),
                        weighted_neighborhood_express_max_distance=(
                            weighted_neighborhood_express_max_distance
                        ),
                        weighted_neighborhood_express_max_rank_pct=(
                            weighted_neighborhood_express_max_rank_pct
                        ),
                        weighted_neighborhood_express_max_neighbors=(
                            weighted_neighborhood_express_max_neighbors
                        ),
                        weighted_neighborhood_express_debug=(
                            weighted_neighborhood_express_debug
                        ),
                        express_distance=express_distance,
                    )

                    results[strategy]["selected"] += 1
                    results[strategy]["miscovered"] += int(strategy_result["miscovered"])
                    results[strategy]["n_calibration"].append(strategy_result["n_calibration"])
                    results[strategy]["interval_length"].append(strategy_result["interval_length"])
                    results[strategy]["infinite_interval"] += int(np.isinf(strategy_result["interval_length"]))
                    raw_rows.append({
                        "run": run,
                        "t": t,
                        "strategy": strategy,
                        "miscovered": int(strategy_result["miscovered"]),
                        "n_calibration": strategy_result["n_calibration"],
                        "interval_length": strategy_result["interval_length"],
                        "buffer": strategy_result["buffer"],
                        "score_t": score_t,
                        "sum_s_past": np.sum(conformal.s_past),
                        "selection_lower_bound": current_lower_bound,
                        "selection_upper_bound": current_upper_bound,
                        "relaxed_express_exact_matches": strategy_result.get(
                            "relaxed_express_exact_matches"
                        ),
                        "relaxed_express_chosen_size": strategy_result.get(
                            "relaxed_express_chosen_size"
                        ),
                        "relaxed_express_target_size": strategy_result.get(
                            "relaxed_express_target_size"
                        ),
                        "relaxed_express_distance_backend": strategy_result.get(
                            "relaxed_express_distance_backend"
                        ),
                        "relaxed_express_max_distance": strategy_result.get(
                            "relaxed_express_max_distance"
                        ),
                        "relaxed_express_mean_distance": strategy_result.get(
                            "relaxed_express_mean_distance"
                        ),
                        "relaxed_express_relaxation_needed": strategy_result.get(
                            "relaxed_express_relaxation_needed"
                        ),
                        "relaxed_express_added_nonexact": strategy_result.get(
                            "relaxed_express_added_nonexact"
                        ),
                        "distance_express_max_distance": strategy_result.get(
                            "distance_express_max_distance"
                        ),
                        "distance_express_distance_backend": strategy_result.get(
                            "distance_express_distance_backend"
                        ),
                        "distance_express_n_candidates_total": strategy_result.get(
                            "distance_express_n_candidates_total"
                        ),
                        "distance_express_chosen_size": strategy_result.get(
                            "distance_express_chosen_size"
                        ),
                        "distance_express_exact_matches": strategy_result.get(
                            "distance_express_exact_matches"
                        ),
                        "distance_express_max_chosen_distance": strategy_result.get(
                            "distance_express_max_chosen_distance"
                        ),
                        "distance_express_mean_chosen_distance": strategy_result.get(
                            "distance_express_mean_chosen_distance"
                        ),
                        "distance_express_median_chosen_distance": strategy_result.get(
                            "distance_express_median_chosen_distance"
                        ),
                        "distance_express_min_distance": strategy_result.get(
                            "distance_express_min_distance"
                        ),
                        "distance_express_median_distance": strategy_result.get(
                            "distance_express_median_distance"
                        ),
                        "distance_express_mean_distance": strategy_result.get(
                            "distance_express_mean_distance"
                        ),
                        "distance_express_max_distance_observed": strategy_result.get(
                            "distance_express_max_distance_observed"
                        ),
                        "weighted_express_lambda": strategy_result.get(
                            "weighted_express_lambda"
                        ),
                        "weighted_express_distance_backend": strategy_result.get(
                            "weighted_express_distance_backend"
                        ),
                        "weighted_express_distance_normalization": strategy_result.get(
                            "weighted_express_distance_normalization"
                        ),
                        "weighted_express_max_distance_cutoff": strategy_result.get(
                            "weighted_express_max_distance_cutoff"
                        ),
                        "weighted_express_max_rank_pct": strategy_result.get(
                            "weighted_express_max_rank_pct"
                        ),
                        "weighted_express_n_candidates_total": strategy_result.get(
                            "weighted_express_n_candidates_total"
                        ),
                        "weighted_express_n_positive_weights": strategy_result.get(
                            "weighted_express_n_positive_weights"
                        ),
                        "weighted_express_positive_weight_fraction": strategy_result.get(
                            "weighted_express_positive_weight_fraction"
                        ),
                        "weighted_express_sum_positive_weights": strategy_result.get(
                            "weighted_express_sum_positive_weights"
                        ),
                        "weighted_express_sum_raw_weights": strategy_result.get(
                            "weighted_express_sum_raw_weights"
                        ),
                        "weighted_express_finite_mass": strategy_result.get(
                            "weighted_express_finite_mass"
                        ),
                        "weighted_express_test_mass": strategy_result.get(
                            "weighted_express_test_mass"
                        ),
                        "weighted_express_min_distance": strategy_result.get(
                            "weighted_express_min_distance"
                        ),
                        "weighted_express_median_distance": strategy_result.get(
                            "weighted_express_median_distance"
                        ),
                        "weighted_express_mean_distance": strategy_result.get(
                            "weighted_express_mean_distance"
                        ),
                        "weighted_express_max_distance": strategy_result.get(
                            "weighted_express_max_distance"
                        ),
                        "weighted_express_min_weight": strategy_result.get(
                            "weighted_express_min_weight"
                        ),
                        "weighted_express_median_weight": strategy_result.get(
                            "weighted_express_median_weight"
                        ),
                        "weighted_express_mean_weight": strategy_result.get(
                            "weighted_express_mean_weight"
                        ),
                        "weighted_express_max_weight": strategy_result.get(
                            "weighted_express_max_weight"
                        ),
                        "weighted_express_n_eff": strategy_result.get(
                            "weighted_express_n_eff"
                        ),
                        "weighted_express_n_eff_finite": strategy_result.get(
                            "weighted_express_n_eff_finite"
                        ),
                        "weighted_express_weighted_mean_distance": strategy_result.get(
                            "weighted_express_weighted_mean_distance"
                        ),
                        "weighted_express_stress": strategy_result.get(
                            "weighted_express_stress"
                        ),
                        "weighted_express_infinite": strategy_result.get(
                            "weighted_express_infinite"
                        ),
                        "adaptive_weighted_express_low_distance_threshold": strategy_result.get(
                            "adaptive_weighted_express_low_distance_threshold"
                        ),
                        "adaptive_weighted_express_target_low_distance_count": strategy_result.get(
                            "adaptive_weighted_express_target_low_distance_count"
                        ),
                        "adaptive_weighted_express_lambda_min": strategy_result.get(
                            "adaptive_weighted_express_lambda_min"
                        ),
                        "adaptive_weighted_express_lambda_max": strategy_result.get(
                            "adaptive_weighted_express_lambda_max"
                        ),
                        "adaptive_weighted_express_lambda_t": strategy_result.get(
                            "adaptive_weighted_express_lambda_t"
                        ),
                        "adaptive_weighted_express_stress": strategy_result.get(
                            "adaptive_weighted_express_stress"
                        ),
                        "adaptive_weighted_express_n_low_distance": strategy_result.get(
                            "adaptive_weighted_express_n_low_distance"
                        ),
                        "adaptive_weighted_express_max_distance_cutoff": strategy_result.get(
                            "adaptive_weighted_express_max_distance_cutoff"
                        ),
                        "adaptive_weighted_express_distance_backend": strategy_result.get(
                            "adaptive_weighted_express_distance_backend"
                        ),
                        "adaptive_weighted_express_n_candidates_total": strategy_result.get(
                            "adaptive_weighted_express_n_candidates_total"
                        ),
                        "adaptive_weighted_express_n_positive_weights": strategy_result.get(
                            "adaptive_weighted_express_n_positive_weights"
                        ),
                        "adaptive_weighted_express_positive_weight_fraction": strategy_result.get(
                            "adaptive_weighted_express_positive_weight_fraction"
                        ),
                        "adaptive_weighted_express_sum_positive_weights": strategy_result.get(
                            "adaptive_weighted_express_sum_positive_weights"
                        ),
                        "adaptive_weighted_express_sum_raw_weights": strategy_result.get(
                            "adaptive_weighted_express_sum_raw_weights"
                        ),
                        "adaptive_weighted_express_finite_mass": strategy_result.get(
                            "adaptive_weighted_express_finite_mass"
                        ),
                        "adaptive_weighted_express_test_mass": strategy_result.get(
                            "adaptive_weighted_express_test_mass"
                        ),
                        "adaptive_weighted_express_min_distance": strategy_result.get(
                            "adaptive_weighted_express_min_distance"
                        ),
                        "adaptive_weighted_express_median_distance": strategy_result.get(
                            "adaptive_weighted_express_median_distance"
                        ),
                        "adaptive_weighted_express_mean_distance": strategy_result.get(
                            "adaptive_weighted_express_mean_distance"
                        ),
                        "adaptive_weighted_express_max_distance": strategy_result.get(
                            "adaptive_weighted_express_max_distance"
                        ),
                        "adaptive_weighted_express_min_weight": strategy_result.get(
                            "adaptive_weighted_express_min_weight"
                        ),
                        "adaptive_weighted_express_median_weight": strategy_result.get(
                            "adaptive_weighted_express_median_weight"
                        ),
                        "adaptive_weighted_express_mean_weight": strategy_result.get(
                            "adaptive_weighted_express_mean_weight"
                        ),
                        "adaptive_weighted_express_max_weight": strategy_result.get(
                            "adaptive_weighted_express_max_weight"
                        ),
                        "adaptive_weighted_express_n_eff": strategy_result.get(
                            "adaptive_weighted_express_n_eff"
                        ),
                        "adaptive_weighted_express_n_eff_finite": strategy_result.get(
                            "adaptive_weighted_express_n_eff_finite"
                        ),
                        "adaptive_weighted_express_weighted_mean_distance": strategy_result.get(
                            "adaptive_weighted_express_weighted_mean_distance"
                        ),
                        "adaptive_weighted_express_stress_weighted_distance": strategy_result.get(
                            "adaptive_weighted_express_stress_weighted_distance"
                        ),
                        "adaptive_weighted_express_infinite": strategy_result.get(
                            "adaptive_weighted_express_infinite"
                        ),
                        "weighted_neighborhood_express_lambda": strategy_result.get(
                            "weighted_neighborhood_express_lambda"
                        ),
                        "weighted_neighborhood_express_distance_backend": strategy_result.get(
                            "weighted_neighborhood_express_distance_backend"
                        ),
                        "weighted_neighborhood_express_distance_normalization": strategy_result.get(
                            "weighted_neighborhood_express_distance_normalization"
                        ),
                        "weighted_neighborhood_express_max_distance_cutoff": strategy_result.get(
                            "weighted_neighborhood_express_max_distance_cutoff"
                        ),
                        "weighted_neighborhood_express_max_rank_pct": strategy_result.get(
                            "weighted_neighborhood_express_max_rank_pct"
                        ),
                        "weighted_neighborhood_express_max_neighbors": strategy_result.get(
                            "weighted_neighborhood_express_max_neighbors"
                        ),
                        "weighted_neighborhood_express_n_candidates_total": strategy_result.get(
                            "weighted_neighborhood_express_n_candidates_total"
                        ),
                        "weighted_neighborhood_express_n_after_neighbor_cap": strategy_result.get(
                            "weighted_neighborhood_express_n_after_neighbor_cap"
                        ),
                        "weighted_neighborhood_express_neighbor_cap_active": strategy_result.get(
                            "weighted_neighborhood_express_neighbor_cap_active"
                        ),
                        "weighted_neighborhood_express_neighbor_cap_boundary_distance": strategy_result.get(
                            "weighted_neighborhood_express_neighbor_cap_boundary_distance"
                        ),
                        "weighted_neighborhood_express_n_positive_weights": strategy_result.get(
                            "weighted_neighborhood_express_n_positive_weights"
                        ),
                        "weighted_neighborhood_express_positive_weight_fraction": strategy_result.get(
                            "weighted_neighborhood_express_positive_weight_fraction"
                        ),
                        "weighted_neighborhood_express_sum_positive_weights": strategy_result.get(
                            "weighted_neighborhood_express_sum_positive_weights"
                        ),
                        "weighted_neighborhood_express_sum_raw_weights": strategy_result.get(
                            "weighted_neighborhood_express_sum_raw_weights"
                        ),
                        "weighted_neighborhood_express_finite_mass": strategy_result.get(
                            "weighted_neighborhood_express_finite_mass"
                        ),
                        "weighted_neighborhood_express_test_mass": strategy_result.get(
                            "weighted_neighborhood_express_test_mass"
                        ),
                        "weighted_neighborhood_express_min_distance": strategy_result.get(
                            "weighted_neighborhood_express_min_distance"
                        ),
                        "weighted_neighborhood_express_median_distance": strategy_result.get(
                            "weighted_neighborhood_express_median_distance"
                        ),
                        "weighted_neighborhood_express_mean_distance": strategy_result.get(
                            "weighted_neighborhood_express_mean_distance"
                        ),
                        "weighted_neighborhood_express_max_distance": strategy_result.get(
                            "weighted_neighborhood_express_max_distance"
                        ),
                        "weighted_neighborhood_express_min_weight": strategy_result.get(
                            "weighted_neighborhood_express_min_weight"
                        ),
                        "weighted_neighborhood_express_median_weight": strategy_result.get(
                            "weighted_neighborhood_express_median_weight"
                        ),
                        "weighted_neighborhood_express_mean_weight": strategy_result.get(
                            "weighted_neighborhood_express_mean_weight"
                        ),
                        "weighted_neighborhood_express_max_weight": strategy_result.get(
                            "weighted_neighborhood_express_max_weight"
                        ),
                        "weighted_neighborhood_express_n_eff": strategy_result.get(
                            "weighted_neighborhood_express_n_eff"
                        ),
                        "weighted_neighborhood_express_n_eff_finite": strategy_result.get(
                            "weighted_neighborhood_express_n_eff_finite"
                        ),
                        "weighted_neighborhood_express_weighted_mean_distance": strategy_result.get(
                            "weighted_neighborhood_express_weighted_mean_distance"
                        ),
                        "weighted_neighborhood_express_stress": strategy_result.get(
                            "weighted_neighborhood_express_stress"
                        ),
                        "weighted_neighborhood_express_infinite": strategy_result.get(
                            "weighted_neighborhood_express_infinite"
                        ),
                    })
            conformal.append_online_point(score_t, point_prediction_t, y_t, s_t, current_bounds)
        
    for strategy in strategies:
        selected = results[strategy]["selected"]
        if selected == 0:
            print(f"{strategy}: no selected points")
            continue

        miscoverage = results[strategy]["miscovered"] / selected
        avg_n_calibration = np.mean(results[strategy]["n_calibration"])
        median_interval_length = np.median(results[strategy]["interval_length"])
        infinite_fraction = results[strategy]["infinite_interval"] / selected

        print(
            f"{strategy}: "
            f"selected={selected}, "
            f"miscoverage={miscoverage:.3f}, "
            f"avg_n_calibration={avg_n_calibration:.2f}, "
            f"median_interval_length={median_interval_length:.3f}, "
            f"infinite_fraction={infinite_fraction:.3f}"
        )

    if weighted_express_debug:
        weighted_rows = [row for row in raw_rows if row["strategy"] == "WEIGHTED-EXPRESS"]
        if weighted_rows:
            print("WEIGHTED-EXPRESS diagnostics:")
            for key in [
                "weighted_express_sum_raw_weights",
                "weighted_express_n_candidates_total",
                "weighted_express_n_positive_weights",
                "weighted_express_positive_weight_fraction",
                "weighted_express_finite_mass",
                "weighted_express_test_mass",
                "weighted_express_median_distance",
                "weighted_express_mean_distance",
                "weighted_express_median_weight",
                "weighted_express_mean_weight",
                "weighted_express_n_eff",
                "weighted_express_n_eff_finite",
                "weighted_express_stress",
            ]:
                values = np.asarray([
                    row[key]
                    for row in weighted_rows
                    if row.get(key) is not None and not pd.isna(row.get(key))
                ], dtype=float)
                if len(values) == 0:
                    continue
                print(
                    f"  {key}: "
                    f"median={np.median(values):.6g}, "
                    f"mean={np.mean(values):.6g}, "
                    f"min={np.min(values):.6g}, "
                    f"max={np.max(values):.6g}"
                )

    if adaptive_weighted_express_debug:
        adaptive_weighted_rows = [
            row for row in raw_rows if row["strategy"] == "ADAPTIVE-WEIGHTED-EXPRESS"
        ]
        if adaptive_weighted_rows:
            print("ADAPTIVE-WEIGHTED-EXPRESS diagnostics:")
            for key in [
                "adaptive_weighted_express_lambda_t",
                "adaptive_weighted_express_stress",
                "adaptive_weighted_express_n_low_distance",
                "adaptive_weighted_express_n_candidates_total",
                "adaptive_weighted_express_n_positive_weights",
                "adaptive_weighted_express_sum_raw_weights",
                "adaptive_weighted_express_finite_mass",
                "adaptive_weighted_express_test_mass",
                "adaptive_weighted_express_n_eff_finite",
                "adaptive_weighted_express_weighted_mean_distance",
                "adaptive_weighted_express_infinite",
            ]:
                values = np.asarray([
                    row[key]
                    for row in adaptive_weighted_rows
                    if row.get(key) is not None and not pd.isna(row.get(key))
                ], dtype=float)
                if len(values) == 0:
                    continue
                print(
                    f"  {key}: "
                    f"median={np.median(values):.6g}, "
                    f"mean={np.mean(values):.6g}, "
                    f"min={np.min(values):.6g}, "
                    f"max={np.max(values):.6g}"
                )

    if distance_express_debug:
        distance_rows = [row for row in raw_rows if row["strategy"] == "DISTANCE-EXPRESS"]
        if distance_rows:
            print("DISTANCE-EXPRESS diagnostics:")
            for key in [
                "distance_express_chosen_size",
                "distance_express_exact_matches",
                "distance_express_max_chosen_distance",
                "distance_express_mean_chosen_distance",
                "distance_express_median_chosen_distance",
            ]:
                values = np.asarray([
                    row[key]
                    for row in distance_rows
                    if row.get(key) is not None and not pd.isna(row.get(key))
                ], dtype=float)
                if len(values) == 0:
                    continue
                print(
                    f"  {key}: "
                    f"median={np.median(values):.6g}, "
                    f"mean={np.mean(values):.6g}, "
                    f"min={np.min(values):.6g}, "
                    f"max={np.max(values):.6g}"
                )

    if weighted_neighborhood_express_debug:
        weighted_neighborhood_rows = [
            row for row in raw_rows if row["strategy"] == "WEIGHTED-NEIGHBORHOOD-EXPRESS"
        ]
        if weighted_neighborhood_rows:
            print("WEIGHTED-NEIGHBORHOOD-EXPRESS diagnostics:")
            for key in [
                "weighted_neighborhood_express_n_candidates_total",
                "weighted_neighborhood_express_n_after_neighbor_cap",
                "weighted_neighborhood_express_neighbor_cap_active",
                "weighted_neighborhood_express_n_positive_weights",
                "weighted_neighborhood_express_positive_weight_fraction",
                "weighted_neighborhood_express_sum_raw_weights",
                "weighted_neighborhood_express_finite_mass",
                "weighted_neighborhood_express_test_mass",
                "weighted_neighborhood_express_n_eff_finite",
                "weighted_neighborhood_express_stress",
            ]:
                values = np.asarray([
                    row[key]
                    for row in weighted_neighborhood_rows
                    if row.get(key) is not None and not pd.isna(row.get(key))
                ], dtype=float)
                if len(values) == 0:
                    continue
                print(
                    f"  {key}: "
                    f"median={np.median(values):.6g}, "
                    f"mean={np.mean(values):.6g}, "
                    f"min={np.min(values):.6g}, "
                    f"max={np.max(values):.6g}"
                )

    output_dir = dump_experiment_results(
        results,
        raw_rows,
        selected_point_rows,
        n_runs,
        output_root=output_root,
        run_name=run_name,
        config_path=config_path,
        resolved_config=config,
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    validate_output_name(args.suite_name, "--suite-name")
    validate_output_name(args.run_name, "--run-name")
    if args.run_name and not args.suite_name:
        raise ValueError("--run-name requires --suite-name")

    output_root = DEFAULT_RESULTS_ROOT
    if args.suite_name:
        output_root = DEFAULT_RESULTS_ROOT / args.suite_name

    config = load_config(args.config)
    run_experiment(
        config,
        config_path=args.config,
        output_root=output_root,
        run_name=args.run_name,
    )
