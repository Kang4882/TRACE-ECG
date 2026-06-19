#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/apply_gem_patch.sh /path/to/GEM"
  exit 1
fi

GEM_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -d "${GEM_ROOT}/llava" ]]; then
  echo "Not a GEM/LLaVA-style checkout: ${GEM_ROOT}"
  exit 1
fi

echo "[TRACE-ECG] applying tracked GEM patch"
git -C "${GEM_ROOT}" apply "${REPO_ROOT}/patches/gem_trace_ecg_tracked_changes.patch"

echo "[TRACE-ECG] copying new files"
rsync -a "${REPO_ROOT}/patches/new_files/" "${GEM_ROOT}/"

echo "[TRACE-ECG] patch applied"
