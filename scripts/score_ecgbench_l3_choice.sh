#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash score_ecgbench_l3_choice.sh --predictions PRED.jsonl --run-name NAME [--dataset-root ROOT] [--output-dir DIR]

Scores ECGBench-L3 close-ended CSN+G12EC predictions with paper terminology:
Consistent-Correct, Inconsistency, Consistent-Error,
Correctable Inconsistency, and Always-wrong Inconsistency.
EOF
}

PREDICTIONS=""
RUN_NAME=""
DATASET_ROOT="${DATASET_ROOT:-/data/ecg_l3_gemlike_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_ROOT}/results/ecgbench_l3_choice}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --predictions) PREDICTIONS="${2:-}"; shift 2 ;;
    --run-name) RUN_NAME="${2:-}"; shift 2 ;;
    --dataset-root) DATASET_ROOT="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${PREDICTIONS}" || -z "${RUN_NAME}" ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PUBLIC_ROOT}/tools/evaluate_ecgbench_l3_close_ended.py" \
  --predictions "${PREDICTIONS}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}"
