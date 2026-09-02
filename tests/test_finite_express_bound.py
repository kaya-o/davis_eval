import math
import numpy as np

from src.conformal import (
    Conformal,
    compute_relaxed_point_coverage_gap_bound,
    coverage_gap_bound_result_fields,
)


def test_relaxed_point_coverage_gap_bound_with_added_nonexact_points():
    result = compute_relaxed_point_coverage_gap_bound(7, 2, 0.1)

    assert result["bound_alpha"] == 0.1
    assert result["bound_nominal_coverage"] == 0.9
    assert result["bound_m"] == 7
    assert result["bound_b"] == 2
    assert result["bound_total_size"] == 9
    assert result["bound_k_prime"] == 9
    assert result["bound_effective_exact_rank_cutoff_raw"] == 7
    assert result["bound_effective_exact_rank_cutoff"] == 7
    assert result["bound_coverage_lower"] == 7 / 8
    assert math.isclose(result["bound_gap"], 0.025)
    assert math.isclose(result["bound_gap_positive"], 0.025)
    assert math.isclose(result["bound_gap_simple"], 0.025)
    assert result["bound_b_over_m_plus_1"] == 2 / 8
    assert result["bound_vacuous"] is False


def test_relaxed_point_coverage_gap_bound_vacuous_case():
    result = compute_relaxed_point_coverage_gap_bound(0, 9, 0.1)

    assert result["bound_k_prime"] == 9
    assert result["bound_effective_exact_rank_cutoff_raw"] == 0
    assert result["bound_effective_exact_rank_cutoff"] == 0
    assert result["bound_coverage_lower"] == 0.0
    assert result["bound_gap"] == 0.9
    assert result["bound_vacuous"] is True


def test_relaxed_point_coverage_gap_bound_exact_only_conservative_case():
    result = compute_relaxed_point_coverage_gap_bound(5, 0, 0.4)

    assert result["bound_k_prime"] == 4
    assert result["bound_effective_exact_rank_cutoff"] == 4
    assert result["bound_coverage_lower"] == 4 / 6
    assert math.isclose(result["bound_gap"], 0.6 - (4 / 6))
    assert result["bound_gap_positive"] == 0.0
    assert result["bound_gap_simple"] == 0.0
    assert result["bound_vacuous"] is False


def test_relaxed_point_coverage_gap_bound_returns_python_scalars():
    result = compute_relaxed_point_coverage_gap_bound(7, 2, 0.1)

    for key, value in result.items():
        if key == "bound_vacuous":
            assert type(value) is bool
        elif key in {
            "bound_m",
            "bound_b",
            "bound_total_size",
            "bound_k_prime",
            "bound_effective_exact_rank_cutoff_raw",
            "bound_effective_exact_rank_cutoff",
        }:
            assert type(value) is int
        else:
            assert type(value) is float


def test_relaxed_point_coverage_gap_bound_rejects_invalid_inputs():
    invalid_cases = [
        (-1, 0, 0.1),
        (1.5, 0, 0.1),
        (1, -1, 0.1),
        (1, 0, 0.0),
        (1, 0, 1.0),
    ]
    for m, b, alpha in invalid_cases:
        try:
            compute_relaxed_point_coverage_gap_bound(m, b, alpha)
        except ValueError:
            continue
        raise AssertionError((m, b, alpha))


def test_coverage_gap_bound_result_fields_supports_prefixed_sources():
    diagnostics = {
        "relaxed_express_bound_alpha": 0.1,
        "relaxed_express_bound_gap_positive": 0.25,
    }

    fields = coverage_gap_bound_result_fields(
        "relaxed_express",
        diagnostics,
        source_prefix="relaxed_express",
    )

    assert fields["relaxed_express_bound_alpha"] == 0.1
    assert fields["relaxed_express_bound_gap_positive"] == 0.25
    assert math.isnan(fields["relaxed_express_bound_m"])


def test_relaxed_express_populates_coverage_gap_bound_diagnostics():
    c = Conformal(alpha=0.1, randomized_calibration=False, random_seed=123)
    c.scores_off = np.array([4.5, 3.0, 6.0], dtype=float)
    c.residuals_off = np.array([0.1, 0.2, 0.3], dtype=float)
    c.scores_past = np.array([], dtype=float)
    c.residuals_past = np.array([], dtype=float)
    c.bounds_past_lower = np.array([], dtype=float)
    c.bounds_past = np.array([], dtype=float)

    result = c.evaluate_strategy(
        strategy="RELAXED-EXPRESS",
        score_t=4.5,
        y_t=4.6,
        point_prediction_t=4.5,
        current_bounds=(4.0, 5.0),
        relaxed_express_max_distance=1.0,
        express_distance="endpoint",
    )

    expected_bound = compute_relaxed_point_coverage_gap_bound(1, 2, 0.1)

    assert result["relaxed_express_exact_matches"] == 1
    assert result["relaxed_express_chosen_size"] == 3
    assert result["relaxed_express_bound_m"] == expected_bound["bound_m"]
    assert result["relaxed_express_bound_b"] == expected_bound["bound_b"]
    assert result["relaxed_express_bound_total_size"] == expected_bound["bound_total_size"]
    assert result["relaxed_express_bound_gap"] == expected_bound["bound_gap"]
    assert result["relaxed_express_bound_gap_positive"] == expected_bound["bound_gap_positive"]
