import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np

TAU_1 = 5.0
TAU_0 = 4000 #2250

WINDOW_WIDTH = 0.5 #1
TAU_TAIL = 6.25
ALPHA = 0.4
USE_RANDOMIZED_CALIBRATION = True


def dump_experiment_results(
    results,
    raw_rows,
    selected_point_rows,
    n_runs,
    output_root="results",
    run_name=None,
    config_path=None,
    resolved_config=None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"{timestamp}_{run_name}" if run_name else f"{timestamp}_{n_runs}_runs"
    output_dir = Path(output_root) / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if config_path is not None:
        shutil.copyfile(config_path, output_dir / "config.json")

    if resolved_config is not None:
        with (output_dir / "resolved_config.json").open("w") as f:
            json.dump(resolved_config, f, indent=2, sort_keys=True)

    aggregate_path = output_dir / "aggregate_results.csv"
    raw_path = output_dir / "raw_selected_events.csv"
    selected_points_path = output_dir / "selected_datapoints.csv"

    aggregate_fieldnames = [
        "strategy",
        "selected",
        "miscovered",
        "miscoverage",
        "avg_n_calibration",
        "median_interval_length",
        "infinite_fraction",
    ]

    with aggregate_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fieldnames)
        writer.writeheader()

        for strategy, strategy_results in results.items():
            selected = strategy_results["selected"]
            if selected == 0:
                writer.writerow({
                    "strategy": strategy,
                    "selected": 0,
                    "miscovered": 0,
                    "miscoverage": np.nan,
                    "avg_n_calibration": np.nan,
                    "median_interval_length": np.nan,
                    "infinite_fraction": np.nan,
                })
                continue

            writer.writerow({
                "strategy": strategy,
                "selected": selected,
                "miscovered": strategy_results["miscovered"],
                "miscoverage": strategy_results["miscovered"] / selected,
                "avg_n_calibration": np.mean(strategy_results["n_calibration"]),
                "median_interval_length": np.median(strategy_results["interval_length"]),
                "infinite_fraction": strategy_results["infinite_interval"] / selected,
            })

    raw_fieldnames = [
        "run",
        "t",
        "strategy",
        "miscovered",
        "n_calibration",
        "interval_length",
        "buffer",
        "score_t",
        "sum_s_past",
        "selection_lower_bound",
        "selection_upper_bound",

        "relaxed_express_exact_matches",
        "relaxed_express_chosen_size",
        "relaxed_express_target_size",
        "relaxed_express_distance_backend",
        "relaxed_express_max_distance",
        "relaxed_express_mean_distance",
        "relaxed_express_relaxation_needed",
        "relaxed_express_added_nonexact",

        "weighted_express_lambda",
        "weighted_express_distance_backend",
        "weighted_express_distance_normalization",
        "weighted_express_max_distance_cutoff",
        "weighted_express_max_rank_pct",
        "weighted_express_n_candidates_total",
        "weighted_express_n_positive_weights",
        "weighted_express_positive_weight_fraction",
        "weighted_express_sum_positive_weights",
        "weighted_express_sum_raw_weights",
        "weighted_express_finite_mass",
        "weighted_express_test_mass",
        "weighted_express_min_distance",
        "weighted_express_median_distance",
        "weighted_express_mean_distance",
        "weighted_express_max_distance",
        "weighted_express_min_weight",
        "weighted_express_median_weight",
        "weighted_express_mean_weight",
        "weighted_express_max_weight",
        "weighted_express_n_eff",
        "weighted_express_n_eff_finite",
        "weighted_express_weighted_mean_distance",
        "weighted_express_stress",
        "weighted_express_infinite",
    ]

    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    selected_point_fieldnames = [
        "run",
        "t",
        "score_t",
        "residual_t",
        "selection_lower_bound",
        "selection_upper_bound",
    ]

    with selected_points_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_point_fieldnames)
        writer.writeheader()
        writer.writerows(selected_point_rows)

    return output_dir

class Conformal:
    def __init__(
        self,
        tau_0=TAU_0,
        tau_1=TAU_1,
        window_width=WINDOW_WIDTH,
        tau_tail=TAU_TAIL,
        alpha=ALPHA,
        randomized_calibration=USE_RANDOMIZED_CALIBRATION,
        random_seed=None,
    ):
        self.tau_0 = tau_0
        self.tau_1 = tau_1
        self.window_width = window_width
        self.tau_tail = tau_tail
        self.alpha = alpha
        self.randomized_calibration = randomized_calibration

        self.x_off = np.array([])
        self.y_off = np.array([])
        self.scores_off = np.array([])
        self.point_predictions_off = np.array([])
        self.residuals_off = np.array([])

        # All online points: scores, selection decisions, and the rule used at arrival.
        self.scores_past = np.array([])
        self.s_past = np.array([])
        self.bounds_past = np.array([])
        self.bounds_past_lower = np.array([])
        self.residuals_past = np.array([])

        # Labeled online points only: safe for calibration.
        self.selected_scores_past = np.array([])
        self.selected_y_past = np.array([])
        self.selected_point_predictions_past = np.array([])
        self.selected_residuals_past = np.array([])
        self.selected_bounds_past = np.array([])
        self.selected_bounds_past_lower = np.array([])
        self.selected_history_indices = np.array([], dtype=int)

        self.rng = np.random.default_rng(random_seed)
        self._express_cache_key = None
        self._express_cache_value = None
        self._relaxed_express_last_diagnostics = None
        self._weighted_express_last_diagnostics = None

    def selected_count(self, j=None):
        past = self.s_past if j is None else self.s_past[:j]
        return float(np.sum(past))

    def selection_bound_lower(self, j, tau_0=None, tau_1=None):
        return self.selection_threshold(j, tau_0=tau_0, tau_1=tau_1)

    def selection_bound(self, j, tau_0=None, tau_1=None, window_width=None, tau_tail=None):#, t_jump=T_JUMP):
        lower = self.selection_threshold(j, tau_0=tau_0, tau_1=tau_1)
        tau_tail = self.tau_tail if tau_tail is None else tau_tail
        window_width = self.window_width if window_width is None else window_width
        if lower >= tau_tail:
            return np.inf
        return lower + window_width

    def selection_threshold(self, j, tau_0=None, tau_1=None):
        tau_0 = self.tau_0 if tau_0 is None else tau_0
        tau_1 = self.tau_1 if tau_1 is None else tau_1
        n_selected = self.selected_count(j)
        return tau_1 + (n_selected / tau_0)

    def selection_bounds(self, j, tau_0=None, tau_1=None, window_width=None, tau_tail=None):
        tau_tail = self.tau_tail if tau_tail is None else tau_tail
        window_width = self.window_width if window_width is None else window_width
        lower = self.selection_bound_lower(j, tau_0=tau_0, tau_1=tau_1)
        upper = np.inf if lower >= tau_tail else lower + window_width
        return lower, upper

    def select_at_bound(self, score, bound):
        return self.select_at_bounds(score, (-np.inf, bound))

    def select_at_bounds(self, score, bounds):
        lower, upper = bounds
        score = np.asarray(score)
        return (score >= lower) & (score <= upper)

    def select_past(self, score, j):
        bounds = (self.bounds_past_lower[j], self.bounds_past[j])
        return int(self.select_at_bounds(score, bounds))

    def select_t(self, score, j=None):
        t = len(self.scores_past) if j is None else j
        return int(self.select_at_bounds(score, self.selection_bounds(t)))

    def same_selection_signature(self, candidate_scores, score_t, lower_bounds, upper_bounds):
        # this is slow because it builds 2d matrix for each time t
        candidate_scores = np.asarray(candidate_scores)
        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)
        if lower_bounds.size == 0:
            return np.ones(candidate_scores.shape[0], dtype=bool)

        candidate_signature = self.select_at_bounds(
            candidate_scores.reshape(-1, 1),
            (lower_bounds.reshape(1, -1), upper_bounds.reshape(1, -1)),
        )
        test_signature = self.select_at_bounds(
            np.asarray([score_t]).reshape(-1, 1),
            (lower_bounds.reshape(1, -1), upper_bounds.reshape(1, -1)),
        )
        return np.all(candidate_signature == test_signature, axis=1)
    
    def same_selection_signature_fast(self, candidate_scores, score_t, lower_bounds, upper_bounds):
        # this is faster hopefully
        candidate_scores = np.asarray(candidate_scores)
        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)

        if lower_bounds.size == 0:
            return np.ones(candidate_scores.shape[0], dtype=bool)

        # Bounds are nondecreasing. A score's selection signature is the
        # contiguous time interval where lower <= score <= upper.
        candidate_start = np.searchsorted(upper_bounds, candidate_scores, side="left")
        candidate_end = np.searchsorted(lower_bounds, candidate_scores, side="right")
        test_start = np.searchsorted(upper_bounds, score_t, side="left")
        test_end = np.searchsorted(lower_bounds, score_t, side="right")

        return (candidate_start == test_start) & (candidate_end == test_end)

    def signature_distance_fast(self, candidate_scores, score_t, lower_bounds, upper_bounds):
        candidate_scores = np.asarray(candidate_scores)
        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)

        if lower_bounds.size == 0:
            return np.zeros(candidate_scores.shape[0], dtype=int)

        # Keep this convention in sync with same_selection_signature_fast:
        # distance zero is exactly the EXPRESS signature match.
        candidate_start = np.searchsorted(upper_bounds, candidate_scores, side="left")
        candidate_end = np.searchsorted(lower_bounds, candidate_scores, side="right")
        test_start = np.searchsorted(upper_bounds, score_t, side="left")
        test_end = np.searchsorted(lower_bounds, score_t, side="right")

        return np.abs(candidate_start - test_start) + np.abs(candidate_end - test_end)

    def signature_distance_hamming(self, candidate_scores, score_t, lower_bounds, upper_bounds):
        candidate_scores = np.asarray(candidate_scores)
        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)

        if lower_bounds.size == 0:
            return np.zeros(candidate_scores.shape[0], dtype=int)

        # This computes exact Hamming distance between full binary selection
        # signatures using the interval representation induced by the monotone
        # moving-band selection rule. It is not endpoint L1 distance; it is the
        # symmetric difference size between two interval signatures.
        candidate_start = np.searchsorted(upper_bounds, candidate_scores, side="left")
        candidate_end = np.searchsorted(lower_bounds, candidate_scores, side="right")
        test_start = np.searchsorted(upper_bounds, score_t, side="left")
        test_end = np.searchsorted(lower_bounds, score_t, side="right")

        candidate_len = candidate_end - candidate_start
        test_len = test_end - test_start

        overlap_start = np.maximum(candidate_start, test_start)
        overlap_end = np.minimum(candidate_end, test_end)
        overlap_len = np.maximum(0, overlap_end - overlap_start)

        return candidate_len + test_len - 2 * overlap_len

    def signature_distance_hamming_slow(self, candidate_scores, score_t, lower_bounds, upper_bounds):
        candidate_scores = np.asarray(candidate_scores)
        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)

        if lower_bounds.size == 0:
            return np.zeros(candidate_scores.shape[0], dtype=int)

        candidate_signature = self.select_at_bounds(
            candidate_scores.reshape(-1, 1),
            (lower_bounds.reshape(1, -1), upper_bounds.reshape(1, -1)),
        )
        test_signature = self.select_at_bounds(
            np.asarray([score_t]).reshape(-1, 1),
            (lower_bounds.reshape(1, -1), upper_bounds.reshape(1, -1)),
        )[0]

        return np.sum(candidate_signature != test_signature.reshape(1, -1), axis=1)

    def signature_distance(
        self,
        candidate_scores,
        score_t,
        lower_bounds,
        upper_bounds,
        distance_backend="endpoint",
    ):
        # Endpoint distance compares compressed interval endpoints of the selection
        # signature. Hamming distance compares full binary counterfactual
        # selection signatures directly; it is slower but more general.
        if distance_backend == "endpoint":
            return self.signature_distance_fast(
                candidate_scores,
                score_t,
                lower_bounds,
                upper_bounds,
            )
        if distance_backend == "hamming":
            return self.signature_distance_hamming(
                candidate_scores,
                score_t,
                lower_bounds,
                upper_bounds,
            )
        raise ValueError(
            "express_distance must be one of ['endpoint', 'hamming'], "
            f"got {distance_backend!r}"
        )

    def min_calibration_size_for_no_infinity(self, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        return int(np.ceil((1.0 / alpha) - 1.0))

    # With m calibration scores, the empirical conformal rank grid has spacing
    # approximately 1 / (m + 1). This helper returns the minimum m needed to make
    # that spacing at most delta.
    def min_calibration_size_for_rank_resolution(self, delta):
        if delta is None:
            return None
        if delta <= 0 or delta > 1:
            raise ValueError(f"delta must be in (0, 1], got {delta}")
        return int(np.ceil((1.0 / delta) - 1.0))

    def relaxed_express_target_size_from_delta(self, delta, alpha=None):
        no_infinity = self.min_calibration_size_for_no_infinity(alpha=alpha)
        if delta is None:
            return no_infinity
        resolution = self.min_calibration_size_for_rank_resolution(delta)
        return max(no_infinity, resolution)

    def quantile_threshold(self, calibration_scores, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        if len(calibration_scores) == 0:
            return np.inf

        calibration_scores = np.asarray(calibration_scores)
        n = len(calibration_scores)

        quantile_idx = int(np.ceil((n + 1) * (1 - alpha))) - 1

        if quantile_idx >= n:
            return np.inf

        return np.partition(calibration_scores, quantile_idx)[quantile_idx]

    def randomized_quantile_threshold(self, calibration_scores, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        xi = self.rng.uniform(0.0, 1.0)
        n = len(calibration_scores)

        if n == 0:
            return np.inf if xi > alpha else -np.inf

        calibration_scores = np.asarray(calibration_scores)

        required_strictly_greater = int(np.floor(alpha * (n + 1) - xi)) + 1

        if required_strictly_greater <= 0:
            return np.inf

        if required_strictly_greater > n:
            return -np.inf

        quantile_idx = n - required_strictly_greater
        return np.partition(calibration_scores, quantile_idx)[quantile_idx]

    def calibration_threshold(self, calibration_scores, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        if self.randomized_calibration:
            return self.randomized_quantile_threshold(calibration_scores, alpha=alpha)
        return self.quantile_threshold(calibration_scores, alpha=alpha)

    def weighted_quantile_threshold(self, calibration_scores, raw_weights, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        calibration_scores = np.asarray(calibration_scores, dtype=float)
        raw_weights = np.asarray(raw_weights, dtype=float)

        if len(calibration_scores) != len(raw_weights):
            raise ValueError("calibration_scores and raw_weights must have the same length")
        if len(calibration_scores) == 0:
            return np.inf
        if not np.all(np.isfinite(calibration_scores)):
            raise ValueError("Weighted EXPRESS calibration scores must be finite")
        if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights < 0):
            raise ValueError("Weighted EXPRESS raw weights must be finite and nonnegative")

        denom = 1.0 + np.sum(raw_weights)
        normalized_weights = raw_weights / denom

        order = np.argsort(calibration_scores, kind="stable")
        values = np.concatenate([calibration_scores[order], [np.inf]])
        masses = np.concatenate([normalized_weights[order], [1.0 / denom]])

        target = 1.0 - alpha
        cumulative = np.cumsum(masses)
        hit_idx = np.flatnonzero(cumulative >= target)

        if len(hit_idx) == 0:
            return np.inf
        return values[hit_idx[0]]

    def normalize_weighted_express_distances(self, distances, normalization, history_length):
        normalization = "history_length" if normalization is None else normalization
        distances = np.asarray(distances, dtype=float)

        if normalization == "none":
            return distances
        if normalization == "history_length":
            return distances / max(int(history_length), 1)
        if normalization == "rank":
            if len(distances) == 0:
                return distances
            unique_values = np.unique(distances)
            if len(unique_values) == 1:
                return np.zeros(len(distances), dtype=float)
            level_lookup = {value: rank for rank, value in enumerate(unique_values)}
            ranks = np.asarray([level_lookup[value] for value in distances], dtype=float)
            return ranks / (len(unique_values) - 1)

        raise ValueError(
            "weighted_express_distance_normalization must be one of "
            f"['rank', 'history_length', 'none'], got {normalization!r}"
        )

    def interval_length_from_threshold(self, threshold):
        if np.isposinf(threshold):
            return np.inf
        if np.isneginf(threshold):
            return 0.0
        return 2 * threshold

    # Calibration strategies
    def full(self):
        return np.concatenate([self.residuals_off, self.selected_residuals_past])

    def s_full(self, current_bounds):
        candidate_residuals = np.concatenate([self.residuals_off, self.selected_residuals_past])
        candidate_scores = np.concatenate([self.scores_off, self.selected_scores_past])
        selected_mask = self.select_at_bounds(candidate_scores, current_bounds)
        return candidate_residuals[selected_mask]

    def s_fix(self, current_bounds):
        selected_mask = self.select_at_bounds(self.scores_off, current_bounds)
        return self.residuals_off[selected_mask]

    # bao et al. 2024
    def ada_off(self, current_bounds):
        selected_mask = self.select_at_bounds(self.scores_off, current_bounds)
        return self.residuals_off[selected_mask]

    def ada_on(self, score_t, current_bounds):
        current_mask = self.select_at_bounds(self.selected_scores_past, current_bounds)
        test_selected_at_past_bounds = self.select_at_bounds(
            score_t,
            (self.selected_bounds_past_lower, self.selected_bounds_past),
        )
        history_match = test_selected_at_past_bounds
        selected_mask = current_mask & history_match
        return self.selected_residuals_past[selected_mask]

    def express(self, score_t, current_bounds):
        current_lower, current_upper = current_bounds
        cache_key = (
            len(self.scores_past),
            len(self.selected_scores_past),
            float(score_t),
            float(current_lower),
            float(current_upper),
        )
        if self._express_cache_key == cache_key:
            return self._express_cache_value

        candidate_residuals = np.concatenate([self.residuals_off, self.selected_residuals_past])
        candidate_scores = np.concatenate([self.scores_off, self.selected_scores_past])
        lower_bounds = np.append(self.bounds_past_lower, current_lower)
        upper_bounds = np.append(self.bounds_past, current_upper)
        selected_mask = self.same_selection_signature_fast(
            candidate_scores,
            score_t,
            lower_bounds,
            upper_bounds,
        )
        selected_residuals = candidate_residuals[selected_mask]
        self._express_cache_key = cache_key
        self._express_cache_value = selected_residuals
        return selected_residuals

    def relaxed_express(
        self,
        score_t,
        current_bounds,
        target_size=None,
        rng=None,
        distance_backend="endpoint",
    ):
        """
        Experimental controlled relaxation of EXPRESS.

        Uses all exact EXPRESS matches. If there are fewer than target_size
        calibration points, adds nearest nonmatching signature points until
        target_size is reached.

        This intentionally relaxes the EXPRESS exchangeability-preserving condition
        and should be treated as a diagnostic strategy, not as a method with the
        Sale-Ramdas validity guarantee.
        """
        current_lower, current_upper = current_bounds
        rng = self.rng if rng is None else rng

        candidate_residuals = np.concatenate([
            self.residuals_off,
            self.selected_residuals_past,
        ])
        candidate_scores = np.concatenate([
            self.scores_off,
            self.selected_scores_past,
        ])

        if target_size is None:
            target_size = self.min_calibration_size_for_no_infinity()
        requested_target_size = int(target_size)

        if len(candidate_residuals) == 0:
            self._relaxed_express_last_diagnostics = {
                "exact_matches": 0,
                "chosen_size": 0,
                "added_nonexact": 0,
                "target_size": requested_target_size,
                "distance_backend": distance_backend,
                "max_distance": np.nan,
                "mean_distance": np.nan,
                "relaxation_needed": False,
            }
            return candidate_residuals

        lower_bounds = np.append(self.bounds_past_lower, current_lower)
        upper_bounds = np.append(self.bounds_past, current_upper)

        distances = self.signature_distance(
            candidate_scores,
            score_t,
            lower_bounds,
            upper_bounds,
            distance_backend=distance_backend,
        )

        target_size = min(requested_target_size, len(candidate_residuals))

        exact_idx = np.flatnonzero(distances == 0)
        exact_match_count = int(len(exact_idx))

        if exact_match_count >= target_size:
            # IMPORTANT: keep all exact EXPRESS matches, not only target_size of them.
            chosen_idx = exact_idx
            added_nonexact = 0
        else:
            needed = target_size - exact_match_count
            nonexact_idx = np.flatnonzero(distances > 0)

            if needed <= 0 or len(nonexact_idx) == 0:
                added_idx = np.array([], dtype=int)
            else:
                nonexact_distances = distances[nonexact_idx]
                boundary_distance = np.partition(nonexact_distances, needed - 1)[needed - 1]

                strict_add_idx = nonexact_idx[nonexact_distances < boundary_distance]
                boundary_idx = nonexact_idx[nonexact_distances == boundary_distance]

                remaining = needed - len(strict_add_idx)

                if remaining <= 0:
                    sampled_boundary_idx = np.array([], dtype=int)
                elif remaining >= len(boundary_idx):
                    sampled_boundary_idx = boundary_idx
                else:
                    sampled_boundary_idx = rng.choice(
                        boundary_idx,
                        size=remaining,
                        replace=False,
                    )

                added_idx = np.concatenate([strict_add_idx, sampled_boundary_idx])

            chosen_idx = np.concatenate([exact_idx, added_idx])
            added_nonexact = int(len(chosen_idx) - exact_match_count)

        chosen_distances = distances[chosen_idx]

        self._relaxed_express_last_diagnostics = {
            "exact_matches": exact_match_count,
            "chosen_size": int(len(chosen_idx)),
            "added_nonexact": added_nonexact,
            "target_size": requested_target_size,
            "distance_backend": distance_backend,
            "max_distance": float(np.max(chosen_distances)) if len(chosen_distances) else np.nan,
            "mean_distance": float(np.mean(chosen_distances)) if len(chosen_distances) else np.nan,
            "relaxation_needed": exact_match_count < target_size,
        }

        return candidate_residuals[chosen_idx]

    def weighted_express(
        self,
        score_t,
        current_bounds,
        lambda_=1.0,
        distance_normalization="rank",
        max_distance=None,
        max_rank_pct=0.05,
        distance_backend="endpoint",
        debug=False,
    ):
        """
        Empirical compact-support soft relaxation of EXPRESS.

        This is inspired by weighted nonexchangeable conformal prediction, but
        the weights depend on the current EXPRESS signature distance. It is not
        a direct instantiation of a fixed-weight validity theorem.
        """
        del debug
        lambda_ = float(lambda_)
        if lambda_ < 0:
            raise ValueError(f"weighted_express_lambda must be nonnegative, got {lambda_}")
        if max_rank_pct is not None:
            max_rank_pct = float(max_rank_pct)
            if max_rank_pct < 0 or max_rank_pct > 1:
                raise ValueError(
                    f"weighted_express_max_rank_pct must be in [0, 1], got {max_rank_pct}"
                )
        if max_distance is not None:
            max_distance = float(max_distance)
        cutoff = max_rank_pct if distance_normalization == "rank" else max_distance

        current_lower, current_upper = current_bounds
        candidate_residuals = np.concatenate([
            self.residuals_off,
            self.residuals_past,
        ])
        candidate_scores = np.concatenate([
            self.scores_off,
            self.scores_past,
        ])

        if len(candidate_residuals) == 0:
            diagnostics = {
                "lambda": lambda_,
                "distance_backend": distance_backend,
                "distance_normalization": distance_normalization,
                "max_distance_cutoff": np.nan if cutoff is None else cutoff,
                "max_rank_pct": np.nan if max_rank_pct is None else max_rank_pct,
                "n_candidates_total": 0,
                "n_positive_weights": 0,
                "positive_weight_fraction": 0.0,
                "sum_positive_weights": 0.0,
                "sum_raw_weights": 0.0,
                "finite_mass": 0.0,
                "test_mass": 1.0,
                "min_distance": np.nan,
                "median_distance": np.nan,
                "mean_distance": np.nan,
                "max_distance": np.nan,
                "min_weight": np.nan,
                "median_weight": np.nan,
                "mean_weight": np.nan,
                "max_weight": np.nan,
                "n_eff": 0.0,
                "n_eff_finite": 0.0,
                "weighted_mean_distance": np.nan,
                "stress": 0.0,
                "infinite": True,
            }
            self._weighted_express_last_diagnostics = diagnostics
            return np.inf, 0, diagnostics

        lower_bounds = np.append(self.bounds_past_lower, current_lower)
        upper_bounds = np.append(self.bounds_past, current_upper)
        raw_distances = self.signature_distance(
            candidate_scores,
            score_t,
            lower_bounds,
            upper_bounds,
            distance_backend=distance_backend,
        )
        distances = self.normalize_weighted_express_distances(
            raw_distances,
            distance_normalization,
            history_length=len(lower_bounds),
        )

        raw_weights = np.exp(-lambda_ * distances)
        if cutoff is not None:
            raw_weights[distances > cutoff] = 0.0
        buffer = self.weighted_quantile_threshold(candidate_residuals, raw_weights)

        positive_weights = raw_weights > 0
        n_positive_weights = int(np.sum(positive_weights))
        sum_raw_weights = float(np.sum(raw_weights))
        denom = 1.0 + sum_raw_weights
        normalized_weights = raw_weights / denom
        finite_weight_sum = float(np.sum(normalized_weights))
        weight_square_sum = float(np.sum(normalized_weights ** 2))
        weight_square_sum_raw = float(np.sum(raw_weights ** 2))

        diagnostics = {
            "lambda": lambda_,
            "distance_backend": distance_backend,
            "distance_normalization": distance_normalization,
            "max_distance_cutoff": np.nan if cutoff is None else float(cutoff),
            "max_rank_pct": np.nan if max_rank_pct is None else float(max_rank_pct),
            "n_candidates_total": int(len(candidate_residuals)),
            "n_positive_weights": n_positive_weights,
            "positive_weight_fraction": float(n_positive_weights / len(raw_weights)),
            "sum_positive_weights": float(np.sum(raw_weights[positive_weights])),
            "sum_raw_weights": sum_raw_weights,
            "finite_mass": float(sum_raw_weights / denom),
            "test_mass": float(1.0 / denom),
            "min_distance": float(np.min(distances)),
            "median_distance": float(np.median(distances)),
            "mean_distance": float(np.mean(distances)),
            "max_distance": float(np.max(distances)),
            "min_weight": float(np.min(raw_weights)),
            "median_weight": float(np.median(raw_weights)),
            "mean_weight": float(np.mean(raw_weights)),
            "max_weight": float(np.max(raw_weights)),
            "n_eff": float(1.0 / weight_square_sum) if weight_square_sum > 0 else 0.0,
            "n_eff_finite": (
                float((sum_raw_weights ** 2) / weight_square_sum_raw)
                if sum_raw_weights > 0 and weight_square_sum_raw > 0
                else 0.0
            ),
            "weighted_mean_distance": (
                float(np.sum(normalized_weights * distances) / finite_weight_sum)
                if finite_weight_sum > 0
                else np.nan
            ),
            "stress": float(np.sum(normalized_weights * distances)),
            "infinite": bool(np.isinf(buffer)),
        }
        self._weighted_express_last_diagnostics = diagnostics
        return buffer, n_positive_weights, diagnostics
    
    def k_express(self, score_t, k, current_bounds):
        current_lower, current_upper = current_bounds
        recent_start = max(0, len(self.scores_past) - k)
        recent_selected_mask = self.selected_history_indices >= recent_start
        candidate_residuals = np.concatenate([
            self.residuals_off,
            self.selected_residuals_past[recent_selected_mask],
        ])
        candidate_scores = np.concatenate([
            self.scores_off,
            self.selected_scores_past[recent_selected_mask],
        ])
        lower_bounds = np.append(self.bounds_past_lower[recent_start:], current_lower)
        upper_bounds = np.append(self.bounds_past[recent_start:], current_upper)
        selected_mask = self.same_selection_signature_fast(
            candidate_scores,
            score_t,
            lower_bounds,
            upper_bounds,
        )
        return candidate_residuals[selected_mask]

    def express_m(self, score_t, k, current_bounds):
        t = len(self.scores_past) + 1
        if t == 0:
            return np.inf, 0

        alpha_sf = (1 / np.sqrt(t)) * self.alpha
        alpha_ex = (1 - (1 / np.sqrt(t))) * self.alpha

        scores_sf = self.s_fix(current_bounds)
        threshold_sf = self.calibration_threshold(scores_sf, alpha=alpha_sf)

        scores_ex = self.express(score_t, current_bounds)
        threshold_ex = self.calibration_threshold(scores_ex, alpha=alpha_ex)

        return min(threshold_sf, threshold_ex), len(scores_sf) + len(scores_ex)

    def append_online_point(self, score_t, point_prediction_t, y_t, s_t, bounds_t):
        self._express_cache_key = None
        self._express_cache_value = None
        lower_t, upper_t = bounds_t
        history_index = len(self.scores_past)
        self.scores_past = np.append(self.scores_past, score_t)
        self.s_past = np.append(self.s_past, s_t)
        self.bounds_past_lower = np.append(self.bounds_past_lower, lower_t)
        self.bounds_past = np.append(self.bounds_past, upper_t)
        self.residuals_past = np.append(
            self.residuals_past,
            np.abs(y_t - point_prediction_t),
        )

        if s_t:
            self.selected_scores_past = np.append(self.selected_scores_past, score_t)
            self.selected_y_past = np.append(self.selected_y_past, y_t)
            self.selected_point_predictions_past = np.append(
                self.selected_point_predictions_past,
                point_prediction_t,
            )
            self.selected_residuals_past = np.append(
                self.selected_residuals_past,
                np.abs(y_t - point_prediction_t),
            )
            self.selected_bounds_past_lower = np.append(self.selected_bounds_past_lower, lower_t)
            self.selected_bounds_past = np.append(self.selected_bounds_past, upper_t)
            self.selected_history_indices = np.append(self.selected_history_indices, history_index)

    def evaluate_strategy(
        self,
        strategy,
        score_t,
        y_t,
        point_prediction_t,
        current_bounds,
        k=5,
        relaxed_express_target_size=None,
        relaxed_express_rank_delta=None,
        weighted_express_lambda=1.0,
        weighted_express_distance_normalization="rank",
        weighted_express_max_distance=None,
        weighted_express_max_rank_pct=0.05,
        weighted_express_debug=False,
        express_distance="endpoint",
    ):
        if strategy == "FULL":
            scores_cal = self.full()
        elif strategy == "S-FULL":
            scores_cal = self.s_full(current_bounds)
        elif strategy == "S-FIX":
            scores_cal = self.s_fix(current_bounds)
        elif strategy == "ADA":
            scores_off = self.ada_off(current_bounds)
            scores_on = self.ada_on(score_t, current_bounds)
            scores_cal = np.concatenate([scores_off, scores_on])
        elif strategy == "EXPRESS":
            scores_cal = self.express(score_t, current_bounds)
        elif strategy == "RELAXED-EXPRESS":
            target_size = relaxed_express_target_size
            if target_size is None:
                target_size = self.relaxed_express_target_size_from_delta(
                    relaxed_express_rank_delta
                )

            scores_cal = self.relaxed_express(
                score_t,
                current_bounds,
                target_size=target_size,
                distance_backend=express_distance,
            )
        elif strategy == "K-EXPRESS":
            scores_cal = self.k_express(score_t, k, current_bounds)
        elif strategy == "WEIGHTED-EXPRESS":
            buffer, n_calibration, diagnostics = self.weighted_express(
                score_t,
                current_bounds,
                lambda_=weighted_express_lambda,
                distance_normalization=weighted_express_distance_normalization,
                max_distance=weighted_express_max_distance,
                max_rank_pct=weighted_express_max_rank_pct,
                distance_backend=express_distance,
                debug=weighted_express_debug,
            )
            covered = np.abs(y_t - point_prediction_t) <= buffer

            return {
                "miscovered": not covered,
                "n_calibration": n_calibration,
                "interval_length": self.interval_length_from_threshold(buffer),
                "buffer": buffer,
                "weighted_express_lambda": diagnostics.get("lambda", np.nan),
                "weighted_express_distance_backend": diagnostics.get(
                    "distance_backend",
                    np.nan,
                ),
                "weighted_express_distance_normalization": diagnostics.get(
                    "distance_normalization",
                    np.nan,
                ),
                "weighted_express_max_distance_cutoff": diagnostics.get(
                    "max_distance_cutoff",
                    np.nan,
                ),
                "weighted_express_max_rank_pct": diagnostics.get("max_rank_pct", np.nan),
                "weighted_express_n_candidates_total": diagnostics.get(
                    "n_candidates_total",
                    np.nan,
                ),
                "weighted_express_n_positive_weights": diagnostics.get(
                    "n_positive_weights",
                    np.nan,
                ),
                "weighted_express_positive_weight_fraction": diagnostics.get(
                    "positive_weight_fraction",
                    np.nan,
                ),
                "weighted_express_sum_positive_weights": diagnostics.get(
                    "sum_positive_weights",
                    np.nan,
                ),
                "weighted_express_sum_raw_weights": diagnostics.get("sum_raw_weights", np.nan),
                "weighted_express_finite_mass": diagnostics.get("finite_mass", np.nan),
                "weighted_express_test_mass": diagnostics.get("test_mass", np.nan),
                "weighted_express_min_distance": diagnostics.get("min_distance", np.nan),
                "weighted_express_median_distance": diagnostics.get(
                    "median_distance",
                    np.nan,
                ),
                "weighted_express_mean_distance": diagnostics.get("mean_distance", np.nan),
                "weighted_express_max_distance": diagnostics.get("max_distance", np.nan),
                "weighted_express_min_weight": diagnostics.get("min_weight", np.nan),
                "weighted_express_median_weight": diagnostics.get("median_weight", np.nan),
                "weighted_express_mean_weight": diagnostics.get("mean_weight", np.nan),
                "weighted_express_max_weight": diagnostics.get("max_weight", np.nan),
                "weighted_express_n_eff": diagnostics.get("n_eff", np.nan),
                "weighted_express_n_eff_finite": diagnostics.get("n_eff_finite", np.nan),
                "weighted_express_weighted_mean_distance": diagnostics.get(
                    "weighted_mean_distance",
                    np.nan,
                ),
                "weighted_express_stress": diagnostics.get("stress", np.nan),
                "weighted_express_infinite": diagnostics.get("infinite", np.nan),
            }
        elif strategy == "EXPRESS-M":
            buffer, n_calibration = self.express_m(score_t, k, current_bounds)
            covered = np.abs(y_t - point_prediction_t) <= buffer

            return {
                "miscovered": not covered,
                "n_calibration": n_calibration,
                "interval_length": self.interval_length_from_threshold(buffer),
                "buffer": buffer,
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        buffer = self.calibration_threshold(scores_cal)
        covered = np.abs(y_t - point_prediction_t) <= buffer

        result = {
            "miscovered": not covered,
            "n_calibration": len(scores_cal),
            "interval_length": self.interval_length_from_threshold(buffer),
            "buffer": buffer,
        }

        if strategy == "RELAXED-EXPRESS":
            diagnostics = self._relaxed_express_last_diagnostics or {}
            result.update({
                "relaxed_express_exact_matches": diagnostics.get("exact_matches", np.nan),
                "relaxed_express_chosen_size": diagnostics.get("chosen_size", np.nan),
                "relaxed_express_target_size": diagnostics.get("target_size", np.nan),
                "relaxed_express_distance_backend": diagnostics.get("distance_backend", np.nan),
                "relaxed_express_max_distance": diagnostics.get("max_distance", np.nan),
                "relaxed_express_mean_distance": diagnostics.get("mean_distance", np.nan),
                "relaxed_express_relaxation_needed": diagnostics.get("relaxation_needed", np.nan),
                "relaxed_express_added_nonexact": diagnostics.get("added_nonexact", np.nan),
            })

        return result
