# DAVIS online selective conformal experiments

This repository contains code submitted as supplementary material along with my masters thesis titled `Evaluation of Calibration-Selection Strategies for Online Selective Conformal Prediction`. It simulates online selection on the DAVIS drug–target affinity data and compares conformal calibration strategies on the selected test points. Experiments are configured with JSON, launched through `src.pipeline`, and written as reproducible CSV/JSON result directories.

This repository was only tested on linux with Python 3.9. Same functionality on other systems or Python versions is not guaranteed.

## Quick start

Run commands from the repository root. Relative data, configuration, and result paths are interpreted from the current working directory.

The required DAVIS data file is not redistributed with this repository. Download
`davis_other_data_models.csv` from the
[PEMI data directory](https://github.com/mingyi811/PEMI/tree/main/data) and place
it at `data/davis_other_data_models.csv`. From the repository root, this can be
done manually or with:

```bash
mkdir -p data
curl -L \
  https://raw.githubusercontent.com/mingyi811/PEMI/main/data/davis_other_data_models.csv \
  -o data/davis_other_data_models.csv
```

Then create the Python environment and run an experiment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.pipeline --config configs/all_strategies_standard.json
```

The example configuration runs every currently implemented strategy once over the standard 20,000-step DAVIS online stream. It uses the difficult setting (`alpha = 0.1`, `tau_0 = 2000`, `window_width = 1`) and is intended as a readable starting point. This comprehensive run is compute-intensive rather than an instant smoke test. For a quick plumbing check, make a copy with a much smaller `data.n_on`; do not interpret that shortened run as a full-horizon comparison. Increase `n_runs` for a final experiment or remove strategies that are not part of the comparison.

The pipeline prints the result directory when it finishes. With the example above, it has this form:

```text
results/YYYYMMDD_HHMMSS_1_runs/
```

The main summary is `aggregate_results.csv`; event-level data are in `raw_selected_events.csv`. Plots are not generated automatically.

## Contents

- [How an experiment works](#how-an-experiment-works)
- [Running experiments](#running-experiments)
- [Configuration loading rules](#configuration-loading-rules)
- [All-strategies reference configuration](#all-strategies-reference-configuration)
- [General configuration reference](#general-configuration-reference)
- [Strategy reference](#strategy-reference)
- [Result directories and dumped data](#result-directories-and-dumped-data)
- [Reproducibility and interpretation](#reproducibility-and-interpretation)
- [Visualizations](#visualizations)
- [Validation and tests](#validation-and-tests)
- [Troubleshooting](#troubleshooting)

## How an experiment works

For each run, the pipeline samples `n_off + n_on` rows without replacement from the configured CSV using seed `seed + run`. The first `n_off` rows form the offline sample and the remaining `n_on` rows form the online stream.

At online time `t`, the selection band is

```text
lower_t = tau_1 + (number selected before t) / tau_0
upper_t = lower_t + window_width
```

The point is selected when its configured selection score lies in the inclusive interval `[lower_t, upper_t]`. Conformal intervals are evaluated only at selected online points.

For a selected point with point prediction `prediction_t`, a strategy normally produces a finite, nonnegative residual threshold `buffer`. The prediction interval is

```text
[prediction_t - buffer, prediction_t + buffer]
```

and the dumped `interval_length` is the full width `2 * buffer`. A `+inf` buffer gives an infinite interval. Under randomized calibration, `-inf` represents an empty interval and is dumped with length `0`. Each strategy changes how calibration residuals are selected or weighted; the online selection rule itself is common to all strategies in the same run.

This is a full-feedback simulation: the outcome and residual of every past online point are stored, including points that were not selected. `FULL`, `CAP`, exact `EXPRESS`, and several EXPRESS variants can therefore use all past online residuals; `K-EXPRESS` deliberately restricts that history. The full-feedback assumption must be reconsidered in an application where labels are observed only after selection.

## Running experiments

### One experiment

The recommended command is:

```bash
python -m src.pipeline --config configs/my_experiment.json
```

The equivalent script invocation is also supported:

```bash
python src/pipeline.py --config configs/my_experiment.json
```

Always supply `--config`. The code-level default path, `configs/davis_experiment.json`, is not present in this repository.

### Command-line options

| Option | Meaning |
|---|---|
| `--config PATH` | User JSON configuration. It is copied into the result directory. |
| `--suite-name NAME` | Put the run under `results/NAME/`. `NAME` must be a simple directory name, not a path. |
| `--run-name NAME` | Give a suite child a readable suffix. Requires `--suite-name`; a timestamp is prepended automatically. |
| `-h`, `--help` | Show the CLI help. |

Without a suite, output is written to:

```text
results/YYYYMMDD_HHMMSS_<n_runs>_runs/
```

With both suite options, output is written to:

```text
results/<suite-name>/YYYYMMDD_HHMMSS_<run-name>/
```

With `--suite-name` but no `--run-name`, the child keeps the ordinary run suffix:

```text
results/<suite-name>/YYYYMMDD_HHMMSS_<n_runs>_runs/
```

`experiment_name` does not control the directory name.

### A suite of related experiments

In this repository a "suite" is just a directory under `results` that contains multiple experiment results. This is just for convenience of organization, so that results of related experiments can be put in the same directory, such as a parameter sweep. The suite helper creates a timestamped directory and metadata file:

```bash
suite_path="$(python -m src.scripts.create_suite example_sweep)"
suite_name="$(basename "$suite_path")"
```
But a directory can also be manually created under `results` and directory name passed to `--suite-name` option.

Run each configuration into that suite with a unique run name:

```bash
python -m src.pipeline \
  --config configs/sweep/value_1.json \
  --suite-name "$suite_name" \
  --run-name value_1

python -m src.pipeline \
  --config configs/sweep/value_2.json \
  --suite-name "$suite_name" \
  --run-name value_2
```

The helper only creates the suite directory and `suite_metadata.json`; it does not execute a list of configurations.

## Configuration loading rules

Configurations are JSON objects. The user JSON is recursively merged over `DEFAULT_CONFIG` in `src/pipeline.py`.

- A missing field inherits its code default.
- A nested object is merged field by field.
- A list, including `strategies`, replaces the default list completely.
- JSON `null` is an explicit value and overrides a non-null default.
- Unknown or misspelled keys are not rejected; they can be silently ignored by the experiment.
- Strategy identifiers are case-sensitive. Avoid duplicate identifiers.
- Many strategy parameters are parsed even when their strategy is absent, so an invalid unused value can still stop the run.

For reproducibility, inspect `resolved_config.json` after every run. It contains the deep-merged, default-filled configuration supplied to execution. Runtime casts such as converting a numeric string to `int` or `float` are not written back, so use correctly typed JSON rather than treating this file as a normalized schema. `config.json` is only the original user file.

There is no jump-rule or `tau_tail` parameter in the current implementation.

## All-strategies reference configuration

The maintained example is [`configs/all_strategies_standard.json`](configs/all_strategies_standard.json). It includes all 14 current strategies but only the parameters needed to make their intended behavior clear.

```json
{
  "experiment_name": "all_strategies_standard",
  "n_runs": 1,
  "seed": 42,
  "data": {
    "path": "data/davis_other_data_models.csv",
    "n_off": 500,
    "n_on": 20000,
    "label_column": "Label"
  },
  "selection": {
    "score_column": "muhat_2",
    "tau_1": 5.0,
    "tau_0": 2000,
    "window_width": 1.0
  },
  "prediction": {
    "point_prediction_column": "muhat_1"
  },
  "conformal": {
    "alpha": 0.1,
    "randomized_calibration": true,
    "express_distance": "hamming",
    "random_calibration_size": 9,
    "k_express": 7500,
    "finite_express_target_size": null,
    "finite_express_rank_delta": null,
    "relaxed_express_max_distance": 0.02,
    "weighted_express_lambda": 25.0,
    "weighted_express_distance_normalization": "history_length",
    "weighted_express_max_distance": null,
    "adaptive_weighted_express_low_distance_threshold": 0.0,
    "adaptive_weighted_express_stress_mode": "sigmoid",
    "adaptive_weighted_express_stress_midpoint_count": 75,
    "adaptive_weighted_express_stress_slope": 0.1,
    "adaptive_weighted_express_stress_count_source": "express_calibration",
    "adaptive_weighted_express_lambda_min": 1.0,
    "adaptive_weighted_express_lambda_max": 100.0,
    "adaptive_weighted_express_max_distance": 0.05,
    "weighted_neighborhood_express_lambda": 25.0,
    "weighted_neighborhood_express_distance_normalization": "history_length",
    "weighted_neighborhood_express_max_distance": null,
    "weighted_neighborhood_express_max_neighbors": 200
  },
  "strategies": [
    "FULL",
    "RANDOM",
    "S-FULL",
    "S-FIX",
    "ADA",
    "CAP",
    "EXPRESS",
    "FINITE-EXPRESS",
    "RELAXED-EXPRESS",
    "K-EXPRESS",
    "WEIGHTED-EXPRESS",
    "ADAPTIVE-WEIGHTED-EXPRESS",
    "WEIGHTED-NEIGHBORHOOD-EXPRESS",
    "EXPRESS-M"
  ]
}
```

At `alpha = 0.1`, FINITE-EXPRESS's automatic minimum is 9, so the example uses 9 points for the RANDOM control. The WEIGHTED-NEIGHBORHOOD settings are an interpretable control: they match WEIGHTED-EXPRESS except for the 200-neighbor cap. These are example values, not claims of universal optimality.

## General configuration reference

### Top-level options

| Key | Code default | Meaning |
|---|---:|---|
| `experiment_name` | `"davis_muhat2_moving_window"` | Free-form metadata saved in the resolved configuration. It does not name the output directory. |
| `n_runs` | `10` | Number of independent repetitions. Use a positive integer. |
| `seed` | `42` | Base random seed. Run `r` uses `seed + r`. |
| `data` | object | Input data and offline/online split; see below. |
| `selection` | object | Online selection policy; see below. |
| `prediction` | object | Point-prediction source; see below. |
| `conformal` | object | Shared and strategy-specific conformal parameters. |
| `strategies` | eight-strategy list | Ordered, case-sensitive strategy identifiers to evaluate. Set this explicitly. |

If omitted, `strategies` defaults to:

```json
["FULL", "S-FULL", "S-FIX", "ADA", "CAP", "EXPRESS", "K-EXPRESS", "EXPRESS-M"]
```

### `data`

| Key | Code default | Meaning |
|---|---:|---|
| `path` | `"data/davis_other_data_models.csv"` | Input CSV path, relative to the directory from which the command is run. |
| `n_off` | `500` | Nonnegative integer number of sampled rows assigned to the initial offline set. |
| `n_on` | `20000` | Nonnegative integer number of sampled rows assigned to the online stream. |
| `label_column` | `"Label"` | Response column used to compute absolute residuals and coverage. |

Requirements:

- The CSV must contain at least `n_off + n_on` rows.
- The configured label, prediction, and score columns must be numeric and should be finite.
- The standard DAVIS file contains `Label`, `muhat_1`, `muhat_2`, and `muhat_3`; additional columns are allowed.

### `selection`

| Key | Code default | Meaning |
|---|---:|---|
| `score_column` | `"muhat_2"` | CSV column used by the online selection rule. |
| `tau_1` | `5.0` | Initial lower edge of the selection band. |
| `tau_0` | `4000` | Drift scale: each previous selection raises the lower edge by `1 / tau_0`. Use a positive value. |
| `window_width` | `0.5` | Width of the inclusive selection band. Use a nonnegative value. |

The frequently used difficult setting is `tau_0 = 2000` and `window_width = 1`; these are not the code defaults and must be explicit.

### `prediction`

| Key | Code default | Meaning |
|---|---:|---|
| `point_prediction_column` | `"muhat_1"` | CSV column used as the center of every prediction interval. Calibration scores are absolute residuals `abs(label - prediction)`. |

Any of `label_column`, `score_column`, and `point_prediction_column` may name an existing CSV column. The aliases `"ensemble_variance"`, `"variance"`, and `"variance_score"` are also recognized and compute the row-wise variance of `muhat_1`, `muhat_2`, and `muhat_3`.

### Shared `conformal` options

| Key | Code default | Meaning |
|---|---:|---|
| `alpha` | `0.4` | Target miscoverage; use `0 < alpha < 1`. The difficult setting uses `0.1`. |
| `randomized_calibration` | `true` | Use the randomized conformal rank for ordinary unweighted strategies. Weighted-family quantiles do not use this switch. |
| `express_distance` | `"endpoint"` | Signature-distance backend: `"endpoint"` or `"hamming"`. It affects RELAXED and weighted EXPRESS variants, not exact EXPRESS; FINITE-EXPRESS always uses Hamming. |

`randomized_calibration` does not turn off randomness in RANDOM sampling, FINITE boundary ties, or WEIGHTED-NEIGHBORHOOD boundary ties.

### Signature distances

Selection signatures record whether a candidate would be selected by each historical selection rule.

- `"hamming"` is the exact number of disagreements between two binary signatures.
- `"endpoint"` is the L1 distance between the start and end indices of the signatures' compressed selected intervals.

For Hamming distance, zero means an exact EXPRESS match. RELAXED-EXPRESS and ADAPTIVE-WEIGHTED-EXPRESS divide the raw distance by the number of past-plus-current rules. WEIGHTED-EXPRESS and WEIGHTED-NEIGHBORHOOD-EXPRESS use their configured normalization.

## Strategy reference

### `FULL`

Uses all offline residuals and all previous online residuals. It has no strategy-specific parameter.

This is a full-information baseline because it uses outcomes from unselected online points.

### `RANDOM`

Samples without replacement from the same candidate pool as FULL.

Exactly one of the following must be non-null whenever `RANDOM` is enabled:

| Key | Default | Meaning |
|---|---:|---|
| `random_calibration_size` | `null` | Fixed requested sample size, an integer at least zero. The actual size is capped by the available pool. |
| `random_calibration_track_strategy` | `null` | Strategy identifier whose event-level `n_calibration` determines RANDOM's requested size. It cannot be `"RANDOM"` and need not be listed in `strategies`. |

When tracking a weighted strategy, the tracked value is its positive-weight count, not an ordinary set cardinality.

### `S-FULL`

Uses offline points and previously selected online points whose scores satisfy the current selection rule. It has no strategy-specific parameter.

### `S-FIX`

Uses only offline points whose scores satisfy the current selection rule. It has no strategy-specific parameter.

### `ADA`

Uses offline points satisfying the current rule plus previously selected online points that satisfy both the current-rule and historical-rule compatibility conditions. It has no strategy-specific parameter.

### `CAP`

Implements the decision-driven CAP calibration rule. It starts from the full offline-plus-past-online holdout pool, requires a candidate to satisfy the current rule, and checks historical agreement over the past online observations selected by the current rule. It has no strategy-specific parameter.

### `EXPRESS`

Uses candidates from the full offline-plus-past-online pool whose counterfactual selection signature exactly matches the current point under every past and current rule. It has no strategy-specific parameter.

`express_distance` does not affect exact EXPRESS.

### `FINITE-EXPRESS`

Keeps every exact EXPRESS match. If there are fewer than a requested target, it adds the nearest non-exact candidates by raw Hamming distance until the target or available pool is reached. Ties at the augmentation boundary are sampled uniformly.

| Key | Default | Meaning |
|---|---:|---|
| `finite_express_target_size` | `null` | Explicit nonnegative target. A non-null value takes precedence over `finite_express_rank_delta`. It is a floor when enough candidates exist, not a cap: all exact matches are retained. |
| `finite_express_rank_delta` | `null` | Desired conformal rank-grid resolution `delta`, with `0 < delta <= 1`. Used only when the explicit target is null. |

When the explicit target is null, the target is

```text
max(ceil(1 / alpha - 1), ceil(1 / delta - 1))
```

with the second term omitted when `delta` is null. Thus `alpha = 0.1` and both parameters null give target 9.

The target controls a minimum cardinality when the candidate pool permits it. It does not bound Hamming distance or interval-length error. FINITE-EXPRESS is an empirical relaxation and should not be described as inheriting exact EXPRESS's finite-sample guarantee.

### `RELAXED-EXPRESS`

Includes every candidate whose signature distance, divided by the number of past-plus-current rules, does not exceed a cutoff. It does not force a minimum calibration size.

| Key | Default | Meaning |
|---|---:|---|
| `relaxed_express_max_distance` | `0.02` | Inclusive normalized-distance cutoff in `[0, 1]`; `null` includes the entire pool. |
| `relaxed_express_debug` | `false` | Print aggregate RELAXED diagnostics after the run. Detailed diagnostics are dumped regardless. |

The distance backend is `express_distance`. With Hamming distance, normalized values lie in `[0, 1]`. Endpoint distance divided by history length can exceed 1 even though the configured cutoff is restricted to `[0, 1]`.

This is an empirical relaxation and does not inherit the exact EXPRESS guarantee.

### `K-EXPRESS`

Applies exact EXPRESS matching to all offline points plus only the most recent `k` online observations and their corresponding rules.

| Key | Default | Meaning |
|---|---:|---|
| `k_express` | `7500` | Nonnegative online-history length. Offline points are always retained in the candidate pool. |

A presentation label such as “7500-EXPRESS” means `K-EXPRESS` with `k_express: 7500`.

### `WEIGHTED-EXPRESS`

Uses the full candidate pool with exponential signature-distance weights

```text
weight_i = exp(-weighted_express_lambda * distance_i)
```

and a weighted conformal quantile with an additional test-point mass at positive infinity.

| Key | Default | Meaning |
|---|---:|---|
| `weighted_express_lambda` | `1.0` | Nonnegative exponential decay rate. Larger values concentrate mass nearer the current signature. |
| `weighted_express_distance_normalization` | `"rank"` | One of `"rank"`, `"history_length"`, `"none"`, or `null`. `null` behaves as `"history_length"`. |
| `weighted_express_max_distance` | `null` | Active cutoff, in the selected normalization's units, for `"history_length"` and `"none"`; values above it receive weight zero. `null` means no explicit cutoff. Use a nonnegative value. |
| `weighted_express_max_rank_pct` | `0.05` | Cutoff in `[0, 1]` used only with `"rank"`; `null` means no explicit rank cutoff. |
| `weighted_express_debug` | `false` | Print aggregate weighted diagnostics after the run. |

Normalization modes:

- `"rank"`: map distinct raw-distance levels to evenly spaced values from 0 to 1.
- `"history_length"`: divide raw distance by the number of past-plus-current rules.
- `"none"`: use raw signature distance.

Only one cutoff is active: `max_rank_pct` under rank normalization and `max_distance` otherwise. The inactive cutoff is ignored. Even without an explicit cutoff, exponential underflow can make extremely small weights numerically zero.

The dumped `n_calibration` is the number of strictly positive weights, not an unweighted calibration-set size or effective sample size. Weighted quantiles ignore `randomized_calibration`. This strategy is an empirical relaxation.

### `ADAPTIVE-WEIGHTED-EXPRESS`

Uses history-length-normalized signature distances and changes the exponential decay rate according to calibration-availability stress. With stress `eta_t` in `[0, 1]`,

```text
lambda_t = lambda_max^(1 - eta_t) * lambda_min^eta_t
```

High stress therefore moves `lambda_t` toward `lambda_min`, allowing more distant candidates to contribute.

| Key | Default | Meaning |
|---|---:|---|
| `adaptive_weighted_express_low_distance_threshold` | `0.01` | Normalized-distance threshold in `[0, 1]` used to count near points. |
| `adaptive_weighted_express_target_low_distance_count` | `6` | Positive target count used by linear stress. It is still parsed and validated in sigmoid mode. |
| `adaptive_weighted_express_stress_mode` | `"linear"` | `"linear"` or `"sigmoid"`. |
| `adaptive_weighted_express_stress_midpoint_count` | `6` | Nonnegative count at which sigmoid stress equals `0.5`. Used only by sigmoid stress. |
| `adaptive_weighted_express_stress_slope` | `0.8` | Positive sigmoid slope. Used only by sigmoid stress. |
| `adaptive_weighted_express_stress_count_source` | `"low_distance"` | `"low_distance"` uses the near-point count; `"express_calibration"` uses the exact EXPRESS calibration count. |
| `adaptive_weighted_express_lambda_min` | `35.0` | Positive lower endpoint of the adaptive decay rate. |
| `adaptive_weighted_express_lambda_max` | `300.0` | Positive upper endpoint, at least `lambda_min`. |
| `adaptive_weighted_express_max_distance` | `1.0` | Inclusive normalized-distance cutoff in `[0, 1]`; `null` applies no explicit cutoff. |
| `adaptive_weighted_express_debug` | `false` | Print aggregate adaptive diagnostics after the run. |

The stress formulas are:

```text
linear:  eta_t = clip((target_count - count_t) / target_count, 0, 1)
sigmoid: eta_t = 1 / (1 + exp(slope * (count_t - midpoint_count)))
```

When `stress_count_source` is `"express_calibration"`, `low_distance_threshold` does not control stress, but the near-point count is still computed and dumped as a diagnostic. The strategy uses `express_distance`, always normalizes by history length, and has no separate normalization option. With endpoint distance, the history-normalized value can exceed 1 even though `adaptive_weighted_express_max_distance` is restricted to `[0, 1]`; use `null` when no cutoff is intended.

Its `n_calibration` is the positive-weight count, and its weighted quantile ignores `randomized_calibration`. This strategy is an empirical relaxation.

### `WEIGHTED-NEIGHBORHOOD-EXPRESS`

First retains at most the nearest `K` candidates by raw signature distance, resolving a boundary tie randomly, and then applies the same weighted core as WEIGHTED-EXPRESS.

| Key | Default | Meaning |
|---|---:|---|
| `weighted_neighborhood_express_lambda` | `1.0` | Nonnegative exponential decay rate. |
| `weighted_neighborhood_express_distance_normalization` | `"rank"` | `"rank"`, `"history_length"`, `"none"`, or `null`. |
| `weighted_neighborhood_express_max_distance` | `null` | Nonnegative active cutoff, in the selected normalization's units, outside rank normalization; `null` means no explicit cutoff. |
| `weighted_neighborhood_express_max_rank_pct` | `0.05` | Cutoff in `[0, 1]` used only under rank normalization; `null` disables it. |
| `weighted_neighborhood_express_max_neighbors` | `200` | Positive nearest-candidate cap; `null` disables the neighbor cap. |
| `weighted_neighborhood_express_debug` | `false` | Print aggregate neighborhood diagnostics after the run. |

Distance normalization, active-cutoff selection, weighted quantiles, and `n_calibration` have the same meanings as in WEIGHTED-EXPRESS. This strategy is an empirical relaxation.

### `EXPRESS-M`

Computes an S-FIX threshold and an EXPRESS threshold with a time-varying split of `alpha`, then uses the smaller threshold. For dumped zero-based timestep `t`, the implementation uses `T = t + 1`:

```text
T = t + 1
alpha_SF = alpha / sqrt(T)
alpha_EX = (1 - 1 / sqrt(T)) * alpha
```

It has no effective strategy-specific option. `k_express` is passed into the current function signature but is not used by EXPRESS-M. Its reported `n_calibration` is `len(S-FIX) + len(EXPRESS)`; overlapping candidates can therefore be counted twice.

## Result directories and dumped data

### Directory layout

A completed pipeline run contains five core files:

```text
results/<run-directory>/
├── config.json
├── resolved_config.json
├── aggregate_results.csv
├── raw_selected_events.csv
└── selected_datapoints.csv
```

| File | Purpose |
|---|---|
| `config.json` | Byte-for-byte copy of the supplied user configuration. |
| `resolved_config.json` | Deep-merged effective configuration, including inherited defaults. |
| `aggregate_results.csv` | One pooled summary row per configured strategy. |
| `raw_selected_events.csv` | One event-level row per selected `(run, t, strategy)`. |
| `selected_datapoints.csv` | One strategy-independent row per selected `(run, t)`. |

Suites created with `src.scripts.create_suite` additionally contain `suite_metadata.json`, with the human-readable `suite_name` and local ISO `created_at` timestamp. Passing `--suite-name` directly does not create this metadata file. Visualization scripts generally write a separate `vis/` directory, but the pipeline itself does not create figures.

### `aggregate_results.csv`

Metrics are pooled over all selected events from all runs; they are not averages of run-level metrics.

| Column | Meaning |
|---|---|
| `strategy` | Strategy identifier. |
| `selected` | Total number of evaluated selected events across all runs. |
| `miscovered` | Number of selected events whose residual exceeded `buffer`. |
| `miscoverage` | `miscovered / selected`. |
| `avg_n_calibration` | Arithmetic mean of event-level `n_calibration`. |
| `median_interval_length` | Median of all event-level interval lengths pooled together. Infinite intervals are included. |
| `infinite_fraction` | Fraction of selected events with infinite interval length. |

If no online point is selected, counts are zero and the derived metrics are written as NaN.

For weighted-family strategies, `avg_n_calibration` averages positive-weight counts. It is not a meaningful unweighted calibration-set size or calibration mass. For EXPRESS-M it averages the sum of its two component counts, which can double-count overlapping candidates.

### `selected_datapoints.csv`

This file has one row for every selected event, independent of how many strategies were evaluated.

| Column | Meaning |
|---|---|
| `run` | Zero-based run index. |
| `t` | Zero-based online timestep. |
| `score_t` | Selection score of the selected point. |
| `residual_t` | Absolute prediction residual. |
| `selection_lower_bound` | Lower selection-band edge at arrival. |
| `selection_upper_bound` | Upper selection-band edge at arrival. |

The original label and point prediction are not dumped separately.

### `raw_selected_events.csv`

This is the detailed analysis file. It contains one row for each configured strategy at each selected event. Unselected timesteps are gaps: they do not receive rows.

Common columns:

| Column | Meaning |
|---|---|
| `run`, `t`, `strategy` | Zero-based run/timestep and strategy identifier. |
| `miscovered` | `1` if the selected response falls outside the interval, otherwise `0`. |
| `n_calibration` | Ordinary calibration cardinality for most unweighted strategies; positive-weight count for weighted strategies; summed component counts for EXPRESS-M. |
| `buffer` | Interval half-width. It can be `inf` or `-inf` under randomized calibration with too little calibration mass. |
| `interval_length` | Full symmetric interval width. Positive infinity remains `inf`; a negative-infinite threshold maps to length `0`. |
| `score_t` | Selection score at the selected event. |
| `sum_s_past` | Number of online points selected before `t`. |
| `selection_lower_bound`, `selection_upper_bound` | Selection band used at `t`. |

The remainder is a fixed, wide diagnostic schema. Columns irrelevant to a row's strategy are empty and are normally read as NaN by pandas.

| Prefix | Contents |
|---|---|
| `random_*` | Requested, available, and chosen RANDOM sizes. |
| `cap_*` | CAP pool composition, compatibility counts, and chosen offline/online counts. |
| `finite_express_*` | Exact/added counts, target, Hamming distances, relaxation indicator, and coverage-gap diagnostics. |
| `relaxed_express_*` | Cutoff/backend, chosen/exact counts, distance summaries, and coverage-gap diagnostics. |
| `weighted_express_*` | Lambda, normalization/cutoff, candidate and positive-weight counts, raw/normalized mass, distance and weight summaries, effective sample sizes, stress, and infinity flag. |
| `adaptive_weighted_express_*` | Controller settings, stress source/count, `lambda_t`, exact/near availability, cutoff, weighted diagnostics, and infinity flag. |
| `weighted_neighborhood_express_*` | Weighted diagnostics plus neighbor-cap size, activity, and boundary distance. |

Coverage-bound diagnostics use the `finite_express_bound_*` and `relaxed_express_bound_*` prefixes. Inspect the CSV header or `raw_fieldnames` in `src/conformal.py` when an exact diagnostic schema is required. Treat the wide raw file as a strategy-diagnostics table rather than a tidy table with one schema per method.

### Reading dumped values

```python
from pathlib import Path
import pandas as pd

result_dir = Path("results/YYYYMMDD_HHMMSS_1_runs")
aggregate = pd.read_csv(result_dir / "aggregate_results.csv")
events = pd.read_csv(result_dir / "raw_selected_events.csv")
selected = pd.read_csv(result_dir / "selected_datapoints.csv")

express_events = events.loc[events["strategy"] == "EXPRESS"]
run_level_miscoverage = express_events.groupby("run")["miscovered"].mean()
```

CSVs may contain literal `inf` and `-inf`; non-applicable diagnostics are empty. With unique strategy identifiers, the raw file can be large because its row count is exactly

```text
number of selected events * number of configured strategies.
```

## Reproducibility and interpretation

- Each run samples data with `seed + run`, so the same data/configuration and software version are reproducible.
- Keep strategy order fixed when comparing exact reruns. Ordinary randomized thresholds, RANDOM sampling, and randomized tie-breaking share a generator, so reordering strategies can change which random draws a strategy receives.
- `randomized_calibration` applies to ordinary unweighted conformal thresholds. Weighted-family methods use their deterministic weighted quantile.
- The pooled median interval length includes infinite intervals. It is infinite whenever at least half of the pooled event lengths are infinite.
- A weighted method's positive-weight count is not its effective sample size. Use the dumped `*_n_eff_finite`, finite mass, and test mass diagnostics when studying weighted calibration.
- FINITE-EXPRESS, RELAXED-EXPRESS, WEIGHTED-EXPRESS, ADAPTIVE-WEIGHTED-EXPRESS, and WEIGHTED-NEIGHBORHOOD-EXPRESS are empirical relaxations in this implementation; do not attribute the exact EXPRESS finite-sample guarantee to them without a separate argument.
- The output schema has no explicit version field. Preserve the source revision together with final results.
- Output timestamps have one-second resolution. Avoid launching two jobs with the same destination and suffix in the same second.

## Visualizations

Experiment execution and visualization are separate. Scripts live under `src/scripts/`; most accept either `--result-dir` or `--suite-dir`. For example:

```bash
python src/scripts/sale_ramdas_style_vis.py \
  --result-dir results/YYYYMMDD_HHMMSS_100_runs \
  --hide-title

python src/scripts/selected_datapoints_vis.py \
  --result-dir results/YYYYMMDD_HHMMSS_100_runs \
  --panels \
  --n-runs 8
```

Check a script's `--help` before using it; sweep-specific scripts expect particular strategy and configuration fields.

Disclaimer: Visualization scripts provided in this repository are mostly one-off scripts to create specific figures for the thesis. They weren't maintained over the commit history and might be using older data dump formats or older configs. Therefore, exact reproduction of thesis figures isn't guaranteed. However, thesis results can be verified independently by reproducing the experiments presented and by means of own visualization from the dumped experiment data.

## Validation and tests

Check that JSON parses before starting a long job:

```bash
python -m json.tool configs/my_experiment.json >/dev/null
```

Inspect the effective merged configuration without running an experiment:

```bash
python -c 'import json; from src.pipeline import load_config; print(json.dumps(load_config("configs/my_experiment.json"), indent=2))'
```

Install the test runner, which is not part of `requirements.txt`, and run the suite with:

```bash
python -m pip install pytest
python -m pytest -q
```

Test warnings can be ignored safely.

## Troubleshooting

| Problem | Check |
|---|---|
| The pipeline cannot find the default config | Always pass `--config`; `configs/davis_experiment.json` is not present. |
| A data/config path cannot be found | Run from the repository root or use the correct path relative to the current working directory. |
| `Requested n=... exceeds available samples` | Ensure `data.n_off + data.n_on` does not exceed the CSV row count. |
| `RANDOM requires exactly one...` | Set exactly one of `random_calibration_size` and `random_calibration_track_strategy`; leave the other null or omit it. |
| An unknown strategy error appears | Use one of the 14 exact, uppercase identifiers documented above. |
| A misspelled option seems to have no effect | Unknown keys are not rejected. Compare `config.json` with `resolved_config.json` and the tables above. |
| All summary metrics are NaN | No online point was selected. Check `tau_1`, `tau_0`, `window_width`, and the selection-score distribution. |
| Median interval length is infinite | Infinite event lengths are included in the pooled median. Inspect `infinite_fraction` and event-level `buffer`. |
| Weighted calibration size looks unexpectedly large | `n_calibration` counts positive numerical weights, not weighted mass or effective sample size. |
| No figure appears after the run | The pipeline writes data only. Run the appropriate script under `src/scripts/`. |
