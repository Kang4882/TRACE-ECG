#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash score_ecgbench_broad_outputs.sh --result-root RESULT_ROOT --output-json OUT.json [--golden-root GOLDEN_ROOT]

Aggregates official ECG-Bench broad subset outputs. This is a scorer wrapper,
not a model inference launcher.
EOF
}

RESULT_ROOT=""
OUTPUT_JSON=""
GOLDEN_ROOT="${GOLDEN_ROOT:-/data/ECGBench}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-root) RESULT_ROOT="${2:-}"; shift 2 ;;
    --output-json) OUTPUT_JSON="${2:-}"; shift 2 ;;
    --golden-root) GOLDEN_ROOT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${RESULT_ROOT}" || -z "${OUTPUT_JSON}" ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PUBLIC_ROOT}/tools/evaluate_ecgbench_outputs.py" \
  --result-root "${RESULT_ROOT}" \
  --golden-root "${GOLDEN_ROOT}" \
  --output-json "${OUTPUT_JSON}"
