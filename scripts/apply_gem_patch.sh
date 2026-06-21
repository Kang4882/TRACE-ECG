#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/apply_gem_patch.sh /path/to/GEM

Applies the TRACE-ECG GEM patch. The command is safe to re-run:
- if the tracked patch is already applied, it reports that state;
- if the patch can be applied cleanly, it applies it;
- otherwise it stops and asks you to verify the upstream GEM commit.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

GEM_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_FILE="${REPO_ROOT}/patches/gem_trace_ecg_tracked_changes.patch"
NEW_FILES_DIR="${REPO_ROOT}/patches/new_files/"

if [[ ! -d "${GEM_ROOT}/llava" ]]; then
  echo "Not a GEM/LLaVA-style checkout: ${GEM_ROOT}" >&2
  exit 1
fi
if ! git -C "${GEM_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "GEM_ROOT must be a git checkout so patch state can be verified: ${GEM_ROOT}" >&2
  exit 1
fi

if git -C "${GEM_ROOT}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[TRACE-ECG] tracked GEM patch is already applied"
elif git -C "${GEM_ROOT}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[TRACE-ECG] applying tracked GEM patch"
  git -C "${GEM_ROOT}" apply "${PATCH_FILE}"
else
  cat >&2 <<EOF
[TRACE-ECG] patch does not apply cleanly.

Use the verified upstream GEM commit, then retry:
  git clone https://github.com/lanxiang1017/GEM.git /path/to/GEM
  git -C /path/to/GEM checkout c8a580faae819c57c008e94fa080f5d3c6881769
  bash scripts/apply_gem_patch.sh /path/to/GEM

If you intentionally use a different GEM commit, inspect:
  git -C "${GEM_ROOT}" apply --check "${PATCH_FILE}"
EOF
  exit 1
fi

echo "[TRACE-ECG] copying new files"
if command -v rsync >/dev/null 2>&1; then
  rsync -a "${NEW_FILES_DIR}" "${GEM_ROOT}/"
else
  cp -a "${NEW_FILES_DIR}/." "${GEM_ROOT}/"
fi

echo "[TRACE-ECG] patch ready"
