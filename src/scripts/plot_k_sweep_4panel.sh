#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SUITE_DIR="${1:-results/suite_20260528_101421_k_sweep}"
OUTPUT="${2:-${SUITE_DIR}/vis/k_sweep_metrics_4panels.png}"

cd "${PROJECT_ROOT}"

python src/scripts/k_sweep_4panel_vis.py \
  --suite-dir "${SUITE_DIR}" \
  --output "${OUTPUT}"
