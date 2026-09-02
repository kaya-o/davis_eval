#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULT_DIR="${1:-results/20260608_130243_10_runs}"
RUN="${2:-0}"
OUTPUT_DIR="${3:-${RESULT_DIR}/vis}"
X_MAX="${4:-}"

cd "${PROJECT_ROOT}"

ARGS=(
  --result-dir "${RESULT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --run "${RUN}"
)

if [[ -n "${X_MAX}" ]]; then
  ARGS+=(--x-max "${X_MAX}")
fi

python src/scripts/single_run_selected_event_points_vis.py "${ARGS[@]}"
