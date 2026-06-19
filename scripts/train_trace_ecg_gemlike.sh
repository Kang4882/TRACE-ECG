#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat >&2 <<'EOF'
[deprecated] scripts/train_trace_ecg_gemlike.sh is kept only for backward
compatibility. TRACE-ECG is now the audit framework, not a training objective.
Forwarding to scripts/train_target_margin_triplet_lora.sh, the appendix-only
Target-Margin Triplet LoRA diagnostic launcher.
EOF

exec bash "${SCRIPT_DIR}/train_target_margin_triplet_lora.sh" "$@"
