#!/usr/bin/env bash
# Invoke from the graph follow-up worktree after the direct boundary job exits.
set -euo pipefail
source scripts/dev-env.sh
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_DECODE_GRAPH=0
export M40LLM_ATTN_HEAD128_KERNEL=native
export M40LLM_CACHE_EXPERIMENT=off
experiment_prefix=docs/experiments/2026-10-31-graph-dispatch
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
run_check() {
    local label=$1
    shift
    if "$@" > "${experiment_prefix}-${label}.log" 2>&1; then
        printf '%s\t0\n' "$label" >> "${experiment_prefix}-regression-exits.tsv"
    else
        local result=$?
        printf '%s\t%s\n' "$label" "$result" >> "${experiment_prefix}-regression-exits.tsv"
        tail -25 "${experiment_prefix}-${label}.log"
        return "$result"
    fi
}
export M40LLM_ENABLE_CUBLAS=1
run_check affected cargo test --release --features cuda \
    --test attention_parity_cuda_grid --test attention_last_token \
    --test attention_batched_varlen --test attention_prefill_varlen \
    --test cuda_transfer_order --test forward_with_layer_smoke \
    --test forward_minimal --test forward_parity_toy --test mlp_wrappers \
    --test q8_native_mlp_cache --test q8_attention_projection_runtime \
    -- --include-ignored --nocapture --test-threads=1
run_check parallel cargo test --release --features cuda --test head128_attention_parallel \
    test_parallel_head128_matches_dense_and_cpu \
    -- --include-ignored --nocapture --test-threads=1
run_check parallel-runtime cargo test --release --features cuda --test head128_attention_parallel \
    test_runtime_reuses_workspace_and_rejects_invalid_configs \
    -- --include-ignored --nocapture --test-threads=1
run_check ffn-ownership cargo test --release --features cuda --test q8_ffn_runtime \
    test_q8_ffn_full_forward_reuses_workspace_and_keeps_native_bits \
    -- --include-ignored --nocapture --test-threads=1
export M40LLM_ENABLE_CUBLAS=0
run_check state-cublas-off cargo test --release --features cuda --lib decode_session::graph_tests \
    -- --include-ignored --nocapture --test-threads=1
run_check short-cublas-off cargo test --release --features cuda --test attention_graph_dispatch \
    test_graph_attention_short_dispatch \
    -- --include-ignored --nocapture --test-threads=1
run_check parity-cublas-off cargo test --release --features cuda \
    --test attention_parity_cuda_grid --test forward_parity_toy --test mlp_wrappers \
    -- --include-ignored --nocapture --test-threads=1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
