#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULT_DIR="${1:-results/20260608_115243_10_runs}"
OUTPUT_DIR="${2:-${RESULT_DIR}/vis}"
BIN_WIDTH="${3:-250}"

cd "${PROJECT_ROOT}"

python src/scripts/binned_selected_event_adaptive_diagnostics_vis.py \
  --result-dir "${RESULT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --bin-width "${BIN_WIDTH}"
