#!/bin/bash
# Thin wrapper that launches a single 3-stage cell via
# `tools/run_one_full_sharegpt_cell.sh` with the v4 + DEFAULT max-num-seqs
# config baked in.
#
# Usage:
#   tools/run_v4_cell.sh <cell_tag> <model> [<extra-server-args>]
#
# Examples:
#   tools/run_v4_cell.sh apr26-m2-main Qwen/Qwen3-8B
#   tools/run_v4_cell.sh apr26-r3-prefix-off Qwen/Qwen3-8B "--no-prefix-caching"
#   tools/run_v4_cell.sh apr26-triton Qwen/Qwen3-8B "--attention-backend TRITON_ATTN"
#
# Env overrides:
#   REUSE_PROFILE=<path>   skip Stage 2, reuse this profile pack
#   VALIDATE_ONLY=1 + REAL_BASELINE_DIR=<dir>  skip Stage 1
#   STUB_DIR=<path>        cuda stubs (default ~/cuda_stubs)

set -uo pipefail

CELL_TAG="${1:?cell_tag required}"
BENCH_MODEL="${2:?model required}"
EXTRA_SERVER_ARGS="${3:-}"
shift $(( $# > 3 ? 3 : $# ))

# Sanity-check: refuse --max-num-seqs in extra args.
if echo "$EXTRA_SERVER_ARGS" | grep -qE '\-\-max-num-seqs|\-\-max_num_seqs'; then
    echo "FATAL: do not pass --max-num-seqs in EXTRA_SERVER_ARGS." >&2
    echo "       v4 mandates DEFAULT max-num-seqs. See MEMORY.md." >&2
    exit 1
fi

export CELL_TAG BENCH_MODEL EXTRA_SERVER_ARGS

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$REPO_ROOT/tools/run_one_full_sharegpt_cell.sh"
