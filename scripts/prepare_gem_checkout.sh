#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prepare_gem_checkout.sh /path/to/GEM

Clones or resets a clean GEM checkout to the verified upstream commit and then
applies the TRACE-ECG patch.

Environment overrides:
  GEM_REPO_URL=https://github.com/lanxiang1017/GEM.git
  GEM_UPSTREAM_COMMIT=c8a580faae819c57c008e94fa080f5d3c6881769
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

DEST="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_FILE="${REPO_ROOT}/patches/gem_trace_ecg_tracked_changes.patch"
GEM_REPO_URL="${GEM_REPO_URL:-https://github.com/lanxiang1017/GEM.git}"
GEM_UPSTREAM_COMMIT="${GEM_UPSTREAM_COMMIT:-c8a580faae819c57c008e94fa080f5d3c6881769}"

if [[ -d "${DEST}/.git" ]]; then
  if git -C "${DEST}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "[TRACE-ECG] destination already contains the tracked patch"
    bash "${SCRIPT_DIR}/apply_gem_patch.sh" "${DEST}"
    exit 0
  fi
  if [[ -n "$(git -C "${DEST}" status --porcelain)" ]]; then
    echo "Existing GEM checkout has uncommitted changes and is not already patched: ${DEST}" >&2
    echo "Use a clean checkout or commit/stash your changes first." >&2
    exit 1
  fi
elif [[ -e "${DEST}" ]]; then
  echo "Destination exists but is not a git checkout: ${DEST}" >&2
  exit 1
else
  mkdir -p "$(dirname "${DEST}")"
  git clone "${GEM_REPO_URL}" "${DEST}"
fi

echo "[TRACE-ECG] checking out verified GEM commit: ${GEM_UPSTREAM_COMMIT}"
git -C "${DEST}" fetch --depth 1 origin "${GEM_UPSTREAM_COMMIT}"
git -C "${DEST}" checkout --detach "${GEM_UPSTREAM_COMMIT}"

bash "${SCRIPT_DIR}/apply_gem_patch.sh" "${DEST}"
