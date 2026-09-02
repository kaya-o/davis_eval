import numpy as np

from src.conformal import Conformal
from src.pipeline import SUPPORTED_STRATEGIES


def make_cap_fixture():
    c = Conformal(random_seed=123)
    c.scores_off = np.array([4.0, 4.5, 3.5, 5.5], dtype=float)
    c.residuals_off = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    c.scores_past = np.array([0.5, 4.25, 4.75, 2.5], dtype=float)
    c.residuals_past = np.array([10.0, 11.0, 12.0, 13.0], dtype=float)
    c.bounds_past_lower = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    c.bounds_past = c.bounds_past_lower + 1.0
    return c


def candidate_scores(c):
    return np.concatenate([c.scores_off, c.scores_past])


def candidate_residuals(c):
    return np.concatenate([c.residuals_off, c.residuals_past])


def brute_force_cap_mask(c, score_t, current_bounds):
    scores = candidate_scores(c)
    selected_indices = [
        i
        for i in range(len(c.scores_past))
        if bool(c.select_at_bounds(c.scores_past[i], current_bounds))
    ]

    mask = []
    for score_s in scores:
        if not bool(c.select_at_bounds(score_s, current_bounds)):
            mask.append(False)
            continue

        ok = True
        for i in selected_indices:
            bounds_i = (c.bounds_past_lower[i], c.bounds_past[i])
            pi_s = bool(c.select_at_bounds(score_s, bounds_i))
            pi_t = bool(c.select_at_bounds(score_t, bounds_i))
            if pi_s != pi_t:
                ok = False
                break
        mask.append(ok)

    return np.asarray(mask, dtype=bool)


def brute_force_express_mask(c, score_t, current_bounds):
    scores = candidate_scores(c)
    mask = []
    for score_s in scores:
        if bool(c.select_at_bounds(score_s, current_bounds)) != bool(
            c.select_at_bounds(score_t, current_bounds)
        ):
            mask.append(False)
            continue

        ok = True
        for i in range(len(c.scores_past)):
            bounds_i = (c.bounds_past_lower[i], c.bounds_past[i])
            pi_s = bool(c.select_at_bounds(score_s, bounds_i))
            pi_t = bool(c.select_at_bounds(score_t, bounds_i))
            if pi_s != pi_t:
                ok = False
                break
        mask.append(ok)

    return np.asarray(mask, dtype=bool)


def test_cap_matches_brute_force_definition():
    c = make_cap_fixture()
    score_t = 4.5
    current_bounds = (4.0, 5.0)

    expected_mask = brute_force_cap_mask(c, score_t, current_bounds)
    expected_residuals = candidate_residuals(c)[expected_mask]
    actual_residuals = c.cap(score_t, current_bounds)

    assert np.array_equal(actual_residuals, expected_residuals)


def test_cap_is_subset_of_current_rule_selected_candidates():
    c = make_cap_fixture()
    score_t = 4.5
    current_bounds = (4.0, 5.0)

    cap_mask = brute_force_cap_mask(c, score_t, current_bounds)
    current_mask = c.select_at_bounds(candidate_scores(c), current_bounds)

    assert np.all(cap_mask <= current_mask)


def test_express_is_subset_of_cap_when_n_t_on_is_strict_subset():
    c = make_cap_fixture()
    score_t = 4.5
    current_bounds = (4.0, 5.0)

    n_t_on_mask = c.select_at_bounds(c.scores_past, current_bounds)
    assert 0 < int(np.sum(n_t_on_mask)) < len(c.scores_past)

    cap_mask = brute_force_cap_mask(c, score_t, current_bounds)
    express_mask = brute_force_express_mask(c, score_t, current_bounds)

    assert np.all(express_mask <= cap_mask)
    assert np.any(cap_mask & ~express_mask)


def test_cap_equals_express_when_n_t_on_contains_all_past_online_indices():
    c = make_cap_fixture()
    c.scores_past = np.array([4.1, 4.2, 4.3, 4.4], dtype=float)
    score_t = 4.5
    current_bounds = (4.0, 5.0)

    n_t_on_mask = c.select_at_bounds(c.scores_past, current_bounds)
    assert np.all(n_t_on_mask)

    cap_mask = brute_force_cap_mask(c, score_t, current_bounds)
    express_mask = brute_force_express_mask(c, score_t, current_bounds)

    assert np.array_equal(cap_mask, express_mask)


def test_cap_reduces_to_current_rule_selection_when_n_t_on_is_empty():
    c = make_cap_fixture()
    c.scores_past = np.array([0.5, 1.5, 2.5, 3.5], dtype=float)
    score_t = 4.5
    current_bounds = (4.0, 5.0)

    n_t_on_mask = c.select_at_bounds(c.scores_past, current_bounds)
    assert not np.any(n_t_on_mask)

    cap_mask = brute_force_cap_mask(c, score_t, current_bounds)
    current_mask = c.select_at_bounds(candidate_scores(c), current_bounds)

    assert np.array_equal(cap_mask, current_mask)


def test_evaluate_strategy_cap_returns_common_result_and_diagnostics():
    c = make_cap_fixture()
    score_t = 4.5
    current_bounds = (4.0, 5.0)

    result = c.evaluate_strategy(
        strategy="CAP",
        score_t=score_t,
        y_t=5.0,
        point_prediction_t=4.25,
        current_bounds=current_bounds,
    )

    for key in [
        "miscovered",
        "n_calibration",
        "interval_length",
        "buffer",
        "cap_chosen_size",
        "cap_n_current_selected_online",
    ]:
        assert key in result

    assert result["n_calibration"] == result["cap_chosen_size"]


def test_pipeline_accepts_cap_strategy():
    assert "CAP" in SUPPORTED_STRATEGIES
