// src/cuda.rs
#![allow(clippy::uninlined_format_args)]

#[cfg(feature = "cuda")]
use crate::kv_selection::{
    KvAttentionGroupStats, KvAttentionTelemetrySummary, KvAttentionTopEntry,
};
#[cfg(feature = "cuda")]
use anyhow::anyhow;
use anyhow::Result;
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::ffi::CStr;
#[cfg(feature = "cuda")]
use std::sync::Once;

#[allow(unused_imports)]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
mod ffi {
    use super::*;
    #[repr(C)]
    pub struct M40llmCudaContext {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct M40llmKVCache {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct M40llmCudaGraphExec {
        _private: [u8; 0],
    }
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct M40llmAttentionGroupStats {
        pub prob_mass: f32,
        pub logit_max: f32,
        pub logit_mean: f32,
        pub count: u32,
    }
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct M40llmAttentionTopEntry {
        pub group: u32,
        pub block_index: u32,
        pub token_position: u32,
        pub score: f32,
        pub probability: f32,
    }
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct M40llmAttentionBlockMass {
        pub block_index: u32,
        pub prob_mass: f32,
        pub logit_max: f32,
        pub logit_mean: f32,
        pub count: u32,
    }
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct M40llmAttentionTelemetry {
        pub recent: M40llmAttentionGroupStats,
        pub selected_old_exact: M40llmAttentionGroupStats,
        pub summary: M40llmAttentionGroupStats,
        pub representatives: M40llmAttentionGroupStats,
        pub other: M40llmAttentionGroupStats,
        pub needle_block_mass: f32,
        pub selected_block_masses: [M40llmAttentionBlockMass; 64],
        pub selected_block_mass_count: u32,
        pub top_entries: [M40llmAttentionTopEntry; 8],
        pub top_entry_count: u32,
    }

    extern "C" {
        pub fn m40llm_current_device_props(
            name_buf: *mut ::std::os::raw::c_char,
            buf_len: usize,
            major: *mut i32,
            minor: *mut i32,
            device_id: *mut i32,
        ) -> i32;
        pub fn m40llm_device_malloc(
            ctx: *mut M40llmCudaContext,
            bytes: usize,
            out_ptr: *mut *mut c_void,
        ) -> i32;
        pub fn m40llm_device_free(ctx: *mut M40llmCudaContext, ptr: *mut c_void) -> i32;
        pub fn m40llm_memcpy_h2d(
            ctx: *mut M40llmCudaContext,
            dst_device: *mut c_void,
            src_host: *const c_void,
            bytes: usize,
        ) -> i32;
        pub fn m40llm_memcpy_d2h(
            ctx: *mut M40llmCudaContext,
            dst_host: *mut c_void,
            src_device: *const c_void,
            bytes: usize,
        ) -> i32;
        pub fn m40llm_memcpy_d2d_async(
            ctx: *mut M40llmCudaContext,
            dst_device: *mut c_void,
            src_device: *const c_void,
            bytes: usize,
            stream_kind: u32,
        ) -> i32;

        pub fn m40llm_validate_device_ptr(ptr: *const c_void) -> i32;

        pub fn m40llm_create_context(device_id: i32) -> *mut M40llmCudaContext;
        pub fn m40llm_destroy_context(ctx: *mut M40llmCudaContext);
        pub fn m40llm_stream_synchronize(ctx: *mut M40llmCudaContext, stream_kind: u32) -> i32;
        pub fn m40llm_stream_wait_for_stream(
            ctx: *mut M40llmCudaContext,
            waiting_stream_kind: u32,
            signal_stream_kind: u32,
        ) -> i32;
        pub fn m40llm_cuda_graph_begin_capture(
            ctx: *mut M40llmCudaContext,
            stream_kind: u32,
        ) -> i32;
        pub fn m40llm_cuda_graph_end_capture(
            ctx: *mut M40llmCudaContext,
            stream_kind: u32,
            out_graph: *mut *mut M40llmCudaGraphExec,
        ) -> i32;
        pub fn m40llm_cuda_graph_cancel_capture(
            ctx: *mut M40llmCudaContext,
            stream_kind: u32,
        ) -> i32;
        pub fn m40llm_cuda_graph_launch(
            ctx: *mut M40llmCudaContext,
            graph: *mut M40llmCudaGraphExec,
            stream_kind: u32,
        ) -> i32;
        pub fn m40llm_cuda_graph_launch_timed_sync(
            ctx: *mut M40llmCudaContext,
            graph: *mut M40llmCudaGraphExec,
            stream_kind: u32,
            elapsed_ms: *mut f32,
        ) -> i32;
        pub fn m40llm_cuda_graph_destroy(graph: *mut M40llmCudaGraphExec);

        pub fn m40llm_upload_weights(
            ctx: *mut M40llmCudaContext,
            host_ptr: *const c_void,
            num_bytes: usize,
            out_device_ptr: *mut *mut c_void,
        ) -> i32;

        pub fn m40llm_gemm_f16_storage_f32_compute(
            ctx: *mut M40llmCudaContext,
            d_A: *const c_void,
            d_B: *const c_void,
            d_C: *mut c_void,
            M: i32,
            N: i32,
            K: i32,
        ) -> i32;

        pub fn m40llm_gemm_f32xf16_f32(
            ctx: *mut M40llmCudaContext,
            d_A_f32: *const c_void,
            d_B_f16: *const c_void,
            d_C_f32: *mut c_void,
            M: i32,
            N: i32,
            K: i32,
        ) -> i32;

        pub fn m40llm_gemm_f32xf16_gguf_f32(
            ctx: *mut M40llmCudaContext,
            d_A_f32: *const c_void,
            d_B_f16: *const c_void,
            d_C_f32: *mut c_void,
            M: i32,
            N: i32,
            K: i32,
        ) -> i32;
        pub fn m40llm_gemm_f32xf32_f32_async(
            ctx: *mut M40llmCudaContext,
            d_A_f32: *const c_void,
            d_B_f32_colmajor_nt: *const c_void,
            d_C_f32: *mut c_void,
            M: i32,
            N: i32,
            K: i32,
        ) -> i32;
        pub fn m40llm_materialize_gguf_f16_to_f32_colmajor_nt(
            ctx: *mut M40llmCudaContext,
            d_B_f16: *const c_void,
            d_B_f32_colmajor_nt: *mut c_void,
            N: i32,
            K: i32,
        ) -> i32;

        pub fn m40llm_gemm_f16xf16_f32(
            ctx: *mut M40llmCudaContext,
            d_A_f16: *const c_void,
            d_B_f16: *const c_void,
            d_C_f32: *mut c_void,
            M: i32,
            N: i32,
            K: i32,
        ) -> i32;

        pub fn m40llm_rms_norm_f32_async(
            ctx: *mut M40llmCudaContext,
            d_in: *const c_void,
            d_out: *mut c_void,
            rows: u32,
            dim: u32,
            eps: f32,
        ) -> i32;
        pub fn m40llm_rms_norm_f32_weighted_async(
            ctx: *mut M40llmCudaContext,
            d_in: *const c_void,
            d_weight: *const c_void,
            d_out: *mut c_void,
            rows: u32,
            dim: u32,
            eps: f32,
            weight_dtype: u32,
        ) -> i32;

        pub fn m40llm_rope_f32_async(
            ctx: *mut M40llmCudaContext,
            d_q: *mut c_void,
            d_k: *mut c_void,
            rows: u32,
            num_heads: u32,
            head_dim: u32,
            past_len: u32,
            freq_base: f32,
            freq_scale: f32,
        ) -> i32;
        pub fn m40llm_rope_f32_inplace_async(
            ctx: *mut M40llmCudaContext,
            d_x: *mut c_void,
            rows: u32,
            num_heads: u32,
            head_dim: u32,
            past_len: u32,
            freq_base: f32,
            freq_scale: f32,
        ) -> i32;
        pub fn m40llm_rope_f32_inplace_position_dev_async(
            ctx: *mut M40llmCudaContext,
            d_x: *mut c_void,
            rows: u32,
            num_heads: u32,
            head_dim: u32,
            past_len_dev: *const u32,
            freq_base: f32,
            freq_scale: f32,
        ) -> i32;
        pub fn m40llm_residual_add_f32_async(
            ctx: *mut M40llmCudaContext,
            d_a_f32: *const c_void,
            d_b_f32: *const c_void,
            d_out_f32: *mut c_void,
            n: usize,
        ) -> i32;
        pub fn m40llm_swiglu_f32_async(
            ctx: *mut M40llmCudaContext,
            d_gate_f32: *const c_void,
            d_up_f32: *const c_void,
            d_out_f32: *mut c_void,
            n: usize,
        ) -> i32;

        pub fn m40llm_kvcache_create(
            ctx: *mut M40llmCudaContext,
            max_seq_len: u32,
            max_batch_size: u32,
            num_heads: u32,
            head_dim: u32,
        ) -> *mut M40llmKVCache;
        pub fn m40llm_kvcache_create_compressed(
            ctx: *mut M40llmCudaContext,
            max_seq_len: u32,
            max_batch_size: u32,
            num_heads: u32,
            head_dim: u32,
            recent_window: u32,
            block_size: u32,
            top_blocks: u32,
            representatives: u32,
            representative_policy: u32,
            q8_old_backing: u32,
        ) -> *mut M40llmKVCache;
        pub fn m40llm_kvcache_append_token(
            ctx: *mut M40llmCudaContext,
            kv: *mut M40llmKVCache,
            seq_id: u32,
            k_dev: *const c_void,
            v_dev: *const c_void,
        ) -> i32;
        pub fn m40llm_kvcache_append_token_f32_async(
            ctx: *mut M40llmCudaContext,
            kv: *mut M40llmKVCache,
            seq_id: u32,
            k_dev_f32: *const c_void,
            v_dev_f32: *const c_void,
        ) -> i32;
        pub fn m40llm_kvcache_append_token_f32_rope_k_async(
            ctx: *mut M40llmCudaContext,
            kv: *mut M40llmKVCache,
            seq_id: u32,
            k_dev_f32: *const c_void,
            v_dev_f32: *const c_void,
            past_len: u32,
            freq_base: f32,
            freq_scale: f32,
        ) -> i32;
        pub fn m40llm_kvcache_append_token_f32_rope_k_at_async(
            ctx: *mut M40llmCudaContext,
            kv: *mut M40llmKVCache,
            seq_id: u32,
            k_dev_f32: *const c_void,
            v_dev_f32: *const c_void,
            position: u32,
            past_len: u32,
            freq_base: f32,
            freq_scale: f32,
        ) -> i32;
        pub fn m40llm_kvcache_append_token_f32_rope_k_position_dev_async(
            ctx: *mut M40llmCudaContext,
            kv: *mut M40llmKVCache,
            seq_id: u32,
            k_dev_f32: *const c_void,
            v_dev_f32: *const c_void,
            position_dev: *const u32,
            freq_base: f32,
            freq_scale: f32,
        ) -> i32;
        pub fn m40llm_kvcache_reset(ctx: *mut M40llmCudaContext, kv: *mut M40llmKVCache) -> i32;
        pub fn m40llm_kvcache_debug_read_token(
            ctx: *mut M40llmCudaContext,
            kv: *mut M40llmKVCache,
            seq_id: u32,
            token: u32,
            out_k_f16: *mut c_void,
            out_v_f16: *mut c_void,
        ) -> i32;
        pub fn m40llm_kvcache_debug_read_compressed_state(
            ctx: *mut M40llmCudaContext,
            kv: *mut M40llmKVCache,
            seq_id: u32,
            out_seq_len: *mut u32,
            out_block_counts: *mut u32,
            out_recent_k_f16: *mut c_void,
            out_recent_v_f16: *mut c_void,
            out_summary_k_acc: *mut f32,
            out_summary_v_acc: *mut f32,
            out_summary_k_f16: *mut c_void,
            out_summary_v_f16: *mut c_void,
            out_rep_k_f16: *mut c_void,
            out_rep_v_f16: *mut c_void,
            out_rep_positions: *mut u32,
        ) -> i32;
        pub fn m40llm_kvcache_debug_select_old_blocks(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            seq_len: u32,
            recent_window: u32,
            block_size: u32,
            top_blocks: u32,
            out_blocks_host: *mut u32,
            out_scores_host: *mut f32,
            out_start_host: *mut u32,
            out_end_host: *mut u32,
            out_count: *mut u32,
            max_out: u32,
            out_total_old_blocks: *mut u32,
        ) -> i32;
        pub fn m40llm_kvcache_debug_attention_telemetry(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            mode: u32,
            seq_id: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            seq_len: u32,
            recent_window: u32,
            block_size: u32,
            top_blocks: u32,
            needle_block: u32,
            out: *mut M40llmAttentionTelemetry,
        ) -> i32;
        pub fn m40llm_kvcache_build_compressed_from_dense(
            ctx: *mut M40llmCudaContext,
            compressed: *mut M40llmKVCache,
            dense: *const M40llmKVCache,
            seq_len: u32,
        ) -> i32;

        pub fn m40llm_attention_last_token_f32(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            d_q_f32: *const c_void,
            seq_len: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_last_token_f32_gqa_async(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            seq_len: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_last_token_f32_gqa_seq_len_dev_async(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            d_seq_len: *const u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_last_token_f32_gqa_dense_recent_async(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            seq_len: u32,
            recent_window: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_last_token_f32_gqa_block_select_exact_async(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            seq_len: u32,
            recent_window: u32,
            block_size: u32,
            top_blocks: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_last_token_f32_gqa_block_select_exact_staged_async(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            seq_len: u32,
            recent_window: u32,
            block_size: u32,
            top_blocks: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_last_token_f32_gqa_block_select_exact_staged_with_buffers_async(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            seq_len: u32,
            recent_window: u32,
            block_size: u32,
            top_blocks: u32,
            staged_k_dev: *mut c_void,
            staged_v_dev: *mut c_void,
            staged_positions_dev: *mut c_void,
            staged_counts_dev: *mut c_void,
            staged_capacity_tokens: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_kvcache_build_q8_old_from_dense(
            ctx: *mut M40llmCudaContext,
            kv: *mut M40llmKVCache,
            seq_id: u32,
            seq_len: u32,
            recent_window: u32,
        ) -> i32;
        pub fn m40llm_attention_last_token_f32_gqa_block_select_exact_staged_q8_old_with_buffers_async(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            seq_len: u32,
            recent_window: u32,
            block_size: u32,
            top_blocks: u32,
            staged_k_dev: *mut c_void,
            staged_v_dev: *mut c_void,
            staged_positions_dev: *mut c_void,
            staged_counts_dev: *mut c_void,
            staged_capacity_tokens: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_last_token_f32_gqa_block_select_exact_q8_old_direct_async(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            seq_len: u32,
            recent_window: u32,
            block_size: u32,
            top_blocks: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_last_token_f32_gqa_block_summary_lossy_async(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            seq_len: u32,
            recent_window: u32,
            block_size: u32,
            top_blocks: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_last_token_f32_gqa_compressed_recent_only_async(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            seq_len: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_last_token_f32_gqa_batched(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            d_seq_ids: *const u32,
            d_seq_lens: *const u32,
            batch_size: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_last_token_f32_gqa_batched_async(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            d_seq_ids: *const u32,
            d_seq_lens: *const u32,
            batch_size: u32,
            d_q_f32: *const c_void,
            q_heads: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_prefill_f32_gqa_varlen_head64(
            ctx: *mut M40llmCudaContext,
            d_q_f32: *const c_void,
            d_k_f32: *const c_void,
            d_v_f32: *const c_void,
            d_q_offsets: *const u32,
            d_kv_offsets: *const u32,
            d_q_lens: *const u32,
            d_kv_lens: *const u32,
            batch_size: u32,
            q_heads: u32,
            kv_heads: u32,
            d_out_f32: *mut c_void,
        ) -> i32;
        pub fn m40llm_attention_prefill_f32_gqa_varlen_head64_async(
            ctx: *mut M40llmCudaContext,
            d_q_f32: *const c_void,
            d_k_f32: *const c_void,
            d_v_f32: *const c_void,
            d_q_offsets: *const u32,
            d_kv_offsets: *const u32,
            d_q_lens: *const u32,
            d_kv_lens: *const u32,
            batch_size: u32,
            q_heads: u32,
            kv_heads: u32,
            d_out_f32: *mut c_void,
        ) -> i32;

        pub fn m40llm_kvcache_destroy(kv: *mut M40llmKVCache);

        pub fn m40llm_start_persistent_decode(ctx: *mut M40llmCudaContext) -> i32;
        pub fn m40llm_stop_persistent_decode(ctx: *mut M40llmCudaContext) -> i32;
        pub fn m40llm_persistent_decode_submit_vec(
            ctx: *mut M40llmCudaContext,
            d_in_f32: *const c_void,
            d_out_f32: *mut c_void,
            n: u32,
            scale: f32,
            bias: f32,
            iterations: u32,
            out_command_id: *mut u32,
        ) -> i32;
        pub fn m40llm_persistent_decode_poll(
            ctx: *mut M40llmCudaContext,
            out_status: *mut u32,
            out_command_id: *mut u32,
        ) -> i32;
        // Utility conversion/dequant kernels
        pub fn m40llm_f16_to_f32(
            ctx: *mut M40llmCudaContext,
            d_in_f16: *const c_void,
            d_out_f32: *mut c_void,
            n: usize,
        ) -> i32;
        pub fn m40llm_q80_to_f32(
            ctx: *mut M40llmCudaContext,
            d_in_q80: *const c_void,
            d_out_f32: *mut c_void,
            n: usize,
        ) -> i32;
    }
}

#[cfg(not(feature = "cuda"))]
mod ffi {
    #[allow(dead_code)]
    #[repr(C)]
    pub struct M40llmCudaContext {
        _private: [u8; 0],
    }

    #[allow(dead_code)]
    #[repr(C)]
    pub struct M40llmKVCache {
        _private: [u8; 0],
    }
}

#[derive(Debug, Clone)]
pub struct DeviceProps {
    pub name: String,
    pub major: i32,
    pub minor: i32,
    pub device_id: i32,
}

// Global allocation tracker (bytes) for diagnostics only
pub(crate) static TOTAL_DEVICE_BYTES: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "cuda")]
fn alloc_log_enabled() -> bool {
    std::env::var("M40LLM_ALLOC_LOG").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
fn gemm_log_enabled() -> bool {
    std::env::var("M40LLM_GEMM_LOG").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
fn log_gemm_backend_once(once: &'static Once, name: &str, backend: &str) {
    if gemm_log_enabled() {
        once.call_once(|| eprintln!("[cuda] {name} backend: {backend}"));
    }
}

#[cfg(feature = "cuda")]
fn record_sync_kernel(op: &'static str) {
    crate::profile::record_launch(op);
    crate::profile::record_stream_sync(op);
}

#[cfg(feature = "cuda")]
fn record_async_kernel(op: &'static str) {
    crate::profile::record_launch(op);
}

#[cfg(feature = "cuda")]
fn record_sync_gemm(op: &'static str) {
    if cfg!(have_cublas) {
        crate::profile::record_cublas_call(op);
    } else {
        crate::profile::record_launch(op);
    }
    crate::profile::record_stream_sync(op);
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct AllocInfo {
    size: usize,
    tag: Option<String>,
}

// Public-safe wrapper types usable in both CUDA and non-CUDA builds
#[derive(Debug, Clone)]
pub struct CudaContext {
    #[allow(dead_code)]
    inner: Arc<CudaContextInner>,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DeviceBuffer {
    ctx: CudaContext,
    ptr: NonNull<c_void>,
    bytes: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for DeviceBuffer {}
#[cfg(feature = "cuda")]
unsafe impl Sync for DeviceBuffer {}

#[cfg(feature = "cuda")]
impl DeviceBuffer {
    #[track_caller]
    pub fn new_tagged(ctx: &CudaContext, bytes: usize, tag: &str) -> Result<Self> {
        let ptr = ctx.device_malloc_tagged(bytes, tag)?;
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| anyhow!("device_malloc_tagged returned null for tag={tag}"))?;
        Ok(Self {
            ctx: ctx.clone(),
            ptr,
            bytes,
        })
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct ExactBlockStagingWorkspace {
    q_heads: u32,
    head_dim: u32,
    capacity_tokens: u32,
    staged_k: DeviceBuffer,
    staged_v: DeviceBuffer,
    staged_positions: DeviceBuffer,
    staged_counts: DeviceBuffer,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct ExactBlockStagingPtrs {
    pub staged_k: *mut c_void,
    pub staged_v: *mut c_void,
    pub staged_positions: *mut c_void,
    pub staged_counts: *mut c_void,
    pub capacity_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactOldBacking {
    Dense,
    Q8,
}

impl ExactOldBacking {
    pub fn from_env() -> Self {
        match std::env::var("M40LLM_KV_EXACT_OLD_BACKING").ok().as_deref() {
            Some("q8") | Some("Q8") => Self::Q8,
            _ => Self::Dense,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Q8 => "q8",
        }
    }
}

#[cfg(feature = "cuda")]
unsafe impl Send for ExactBlockStagingWorkspace {}
#[cfg(feature = "cuda")]
unsafe impl Sync for ExactBlockStagingWorkspace {}

#[cfg(feature = "cuda")]
impl ExactBlockStagingWorkspace {
    pub fn new(
        ctx: &CudaContext,
        q_heads: u32,
        head_dim: u32,
        capacity_tokens: u32,
    ) -> Result<Self> {
        if q_heads == 0 || head_dim == 0 || capacity_tokens == 0 {
            anyhow::bail!(
                "ExactBlockStagingWorkspace invalid shape q_heads={q_heads} head_dim={head_dim} capacity_tokens={capacity_tokens}"
            );
        }
        let elems = (q_heads as usize)
            .checked_mul(capacity_tokens as usize)
            .and_then(|v| v.checked_mul(head_dim as usize))
            .ok_or_else(|| anyhow!("exact block staging element count overflow"))?;
        let kv_bytes = elems
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| anyhow!("exact block staging K/V byte size overflow"))?;
        let positions_bytes = (q_heads as usize)
            .checked_mul(capacity_tokens as usize)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u32>()))
            .ok_or_else(|| anyhow!("exact block staging positions byte size overflow"))?;
        let counts_bytes = (q_heads as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| anyhow!("exact block staging counts byte size overflow"))?;
        Ok(Self {
            q_heads,
            head_dim,
            capacity_tokens,
            staged_k: DeviceBuffer::new_tagged(ctx, kv_bytes, "kv_staging:k_f16")?,
            staged_v: DeviceBuffer::new_tagged(ctx, kv_bytes, "kv_staging:v_f16")?,
            staged_positions: DeviceBuffer::new_tagged(
                ctx,
                positions_bytes,
                "kv_staging:positions_u32",
            )?,
            staged_counts: DeviceBuffer::new_tagged(ctx, counts_bytes, "kv_staging:counts_u32")?,
        })
    }

    pub fn ptrs(&self) -> ExactBlockStagingPtrs {
        ExactBlockStagingPtrs {
            staged_k: self.staged_k.as_mut_ptr(),
            staged_v: self.staged_v.as_mut_ptr(),
            staged_positions: self.staged_positions.as_mut_ptr(),
            staged_counts: self.staged_counts.as_mut_ptr(),
            capacity_tokens: self.capacity_tokens,
        }
    }

    pub fn q_heads(&self) -> u32 {
        self.q_heads
    }

    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    pub fn capacity_tokens(&self) -> u32 {
        self.capacity_tokens
    }

    pub fn bytes(&self) -> usize {
        self.staged_k.bytes()
            + self.staged_v.bytes()
            + self.staged_positions.bytes()
            + self.staged_counts.bytes()
    }
}

#[cfg(feature = "cuda")]
impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device_free(self.ptr.as_ptr());
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaGraphExec {
    ctx: CudaContext,
    raw: NonNull<ffi::M40llmCudaGraphExec>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaGraphExec {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaGraphExec {}

#[cfg(feature = "cuda")]
impl CudaGraphExec {
    pub fn launch(&self, stream: CudaStream) -> Result<()> {
        let _g = self.ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_cuda_graph_launch(
                self.ctx.inner.raw.as_ptr(),
                self.raw.as_ptr(),
                stream.ffi_kind(),
            )
        };
        if rc != 0 {
            return Err(anyhow!("m40llm_cuda_graph_launch failed: {rc}"));
        }
        crate::profile::record_launch("cuda_graph_launch");
        Ok(())
    }

    pub fn launch_timed_sync(&self, stream: CudaStream) -> Result<f32> {
        let _g = self.ctx.inner.lock.lock().unwrap();
        let mut elapsed_ms = 0.0f32;
        let rc = unsafe {
            ffi::m40llm_cuda_graph_launch_timed_sync(
                self.ctx.inner.raw.as_ptr(),
                self.raw.as_ptr(),
                stream.ffi_kind(),
                &mut elapsed_ms as *mut _,
            )
        };
        if rc != 0 {
            return Err(anyhow!("m40llm_cuda_graph_launch_timed_sync failed: {rc}"));
        }
        crate::profile::record_launch("cuda_graph_launch");
        crate::profile::record_stream_sync("cuda_graph_launch_timed_sync");
        Ok(elapsed_ms)
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        unsafe {
            ffi::m40llm_cuda_graph_destroy(self.raw.as_ptr());
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct CudaGraphExec;

#[cfg(not(feature = "cuda"))]
impl CudaGraphExec {
    pub fn launch(&self, stream: CudaStream) -> Result<()> {
        let _ = stream;
        anyhow::bail!("CUDA graph launch is unavailable without the cuda feature")
    }

    pub fn launch_timed_sync(&self, stream: CudaStream) -> Result<f32> {
        let _ = stream;
        anyhow::bail!("CUDA graph timed launch is unavailable without the cuda feature")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaStream {
    Prefill,
    Decode,
}

impl CudaStream {
    #[cfg(feature = "cuda")]
    #[inline]
    fn ffi_kind(self) -> u32 {
        match self {
            Self::Prefill => 0,
            Self::Decode => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistentDecodeStatus {
    Idle,
    Pending,
    Done,
    Unknown(u32),
}

impl From<u32> for PersistentDecodeStatus {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Idle,
            1 => Self::Pending,
            2 => Self::Done,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistentDecodePoll {
    pub status: PersistentDecodeStatus,
    pub command_id: u32,
}

#[derive(Debug)]
struct CudaContextInner {
    device_id: i32,
    #[allow(dead_code)]
    lock: Mutex<()>,
    #[cfg(feature = "cuda")]
    raw: NonNull<ffi::M40llmCudaContext>,
    #[cfg(feature = "cuda")]
    weights_ptr: Mutex<Option<NonNull<c_void>>>,
    #[cfg(feature = "cuda")]
    alloc_map: Mutex<HashMap<*mut c_void, AllocInfo>>, // per-allocation info (size, tag)
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaContextInner {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaContextInner {}

impl CudaContext {
    #[inline]
    pub fn total_device_bytes() -> usize {
        TOTAL_DEVICE_BYTES.load(Ordering::SeqCst)
    }
    pub fn new(device_id: i32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let ptr = unsafe { ffi::m40llm_create_context(device_id) };
            let raw =
                NonNull::new(ptr).ok_or_else(|| anyhow!("m40llm_create_context returned null"))?;
            Ok(Self {
                inner: Arc::new(CudaContextInner {
                    device_id,
                    lock: Mutex::new(()),
                    raw,
                    weights_ptr: Mutex::new(None),
                    alloc_map: Mutex::new(HashMap::new()),
                }),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                inner: Arc::new(CudaContextInner {
                    device_id,
                    lock: Mutex::new(()),
                    #[cfg(feature = "cuda")]
                    raw: NonNull::dangling(),
                    #[cfg(feature = "cuda")]
                    weights_ptr: Mutex::new(None),
                    #[cfg(feature = "cuda")]
                    alloc_map: Mutex::new(HashMap::new()),
                }),
            })
        }
    }

    #[allow(dead_code)]
    pub fn device_id(&self) -> i32 {
        self.inner.device_id
    }

    /// Convenience: create a context that auto-selects Tesla M40 (sm_52) when available.
    /// Equivalent to `CudaContext::new(-1)`.
    pub fn new_m40() -> Result<Self> {
        Self::new(-1)
    }

    pub fn current_device_props(&self) -> Result<DeviceProps> {
        #[cfg(feature = "cuda")]
        {
            let mut name_buf = [0i8; 128];
            let mut major: i32 = 0;
            let mut minor: i32 = 0;
            let mut device_id: i32 = -1;
            let rc = unsafe {
                ffi::m40llm_current_device_props(
                    name_buf.as_mut_ptr(),
                    name_buf.len(),
                    &mut major as *mut _,
                    &mut minor as *mut _,
                    &mut device_id as *mut _,
                )
            };
            if rc != 0 {
                return Err(anyhow!("m40llm_current_device_props failed: {rc}"));
            }
            let cname = unsafe { CStr::from_ptr(name_buf.as_ptr()) };
            Ok(DeviceProps {
                name: cname.to_string_lossy().into_owned(),
                major,
                minor,
                device_id,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(DeviceProps {
                name: "stub".into(),
                major: 0,
                minor: 0,
                device_id: self.inner.device_id,
            })
        }
    }

    pub fn synchronize_stream(&self, stream: CudaStream) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = unsafe {
                ffi::m40llm_stream_synchronize(self.inner.raw.as_ptr(), stream.ffi_kind())
            };
            if rc != 0 {
                return Err(anyhow!("m40llm_stream_synchronize failed: {rc}"));
            }
            crate::profile::record_stream_sync("stream_synchronize");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = stream;
            Ok(())
        }
    }

    pub fn stream_wait_for_stream(
        &self,
        waiting: CudaStream,
        signal: CudaStream,
        op: &'static str,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = unsafe {
                ffi::m40llm_stream_wait_for_stream(
                    self.inner.raw.as_ptr(),
                    waiting.ffi_kind(),
                    signal.ffi_kind(),
                )
            };
            if rc != 0 {
                return Err(anyhow!(
                    "m40llm_stream_wait_for_stream failed for {op}: {rc}"
                ));
            }
            crate::profile::record_stream_wait(op);
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (waiting, signal, op);
            Ok(())
        }
    }

    pub fn capture_graph(
        &self,
        stream: CudaStream,
        capture: impl FnOnce() -> Result<()>,
    ) -> Result<CudaGraphExec> {
        #[cfg(feature = "cuda")]
        {
            {
                let _g = self.inner.lock.lock().unwrap();
                let rc = unsafe {
                    ffi::m40llm_cuda_graph_begin_capture(self.inner.raw.as_ptr(), stream.ffi_kind())
                };
                if rc != 0 {
                    return Err(anyhow!("m40llm_cuda_graph_begin_capture failed: {rc}"));
                }
            }

            if let Err(err) = capture() {
                let _g = self.inner.lock.lock().unwrap();
                unsafe {
                    let _ = ffi::m40llm_cuda_graph_cancel_capture(
                        self.inner.raw.as_ptr(),
                        stream.ffi_kind(),
                    );
                }
                return Err(err);
            }

            let mut out: *mut ffi::M40llmCudaGraphExec = std::ptr::null_mut();
            let _g = self.inner.lock.lock().unwrap();
            let rc = unsafe {
                ffi::m40llm_cuda_graph_end_capture(
                    self.inner.raw.as_ptr(),
                    stream.ffi_kind(),
                    &mut out as *mut _,
                )
            };
            if rc != 0 || out.is_null() {
                return Err(anyhow!(
                    "m40llm_cuda_graph_end_capture failed: rc={rc}, out_null={}",
                    out.is_null()
                ));
            }
            crate::profile::record_launch("cuda_graph_instantiate");
            Ok(CudaGraphExec {
                ctx: self.clone(),
                raw: NonNull::new(out).ok_or_else(|| anyhow!("graph capture returned null"))?,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = stream;
            capture()?;
            anyhow::bail!("CUDA graph capture is unavailable without the cuda feature")
        }
    }

    #[cfg(feature = "cuda")]
    fn synchronize_stream_for_op(&self, stream: CudaStream, op: &'static str) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        let rc =
            unsafe { ffi::m40llm_stream_synchronize(self.inner.raw.as_ptr(), stream.ffi_kind()) };
        if rc != 0 {
            return Err(anyhow!("m40llm_stream_synchronize failed for {op}: {rc}"));
        }
        crate::profile::record_stream_sync(op);
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl CudaContext {
    #[track_caller]
    fn device_malloc_inner(&self, bytes: usize, tag: Option<&str>) -> Result<*mut c_void> {
        let _g = self.inner.lock.lock().unwrap();
        let mut out: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            ffi::m40llm_device_malloc(self.inner.raw.as_ptr(), bytes, &mut out as *mut _)
        };
        if rc != 0 || out.is_null() {
            let props = self.current_device_props().ok();
            return Err(anyhow!(
                "m40llm_device_malloc failed: rc={rc}, bytes={bytes}, total_before={}{}",
                TOTAL_DEVICE_BYTES.load(Ordering::SeqCst),
                props
                    .map(|p| format!(
                        ", device='{}' sm_{}{} id {}",
                        p.name, p.major, p.minor, p.device_id
                    ))
                    .unwrap_or_default()
            ));
        }
        TOTAL_DEVICE_BYTES.fetch_add(bytes, Ordering::SeqCst);
        if let Ok(mut map) = self.inner.alloc_map.lock() {
            map.insert(
                out,
                AllocInfo {
                    size: bytes,
                    tag: tag.map(|s| s.to_string()),
                },
            );
        }
        crate::profile::record_device_alloc("device_malloc", bytes);
        if alloc_log_enabled() {
            let caller = std::panic::Location::caller();
            let total = TOTAL_DEVICE_BYTES.load(Ordering::SeqCst);
            let mut msg = format!(
                "[cuda] device_malloc: {} bytes (total={}) at {}:{}",
                bytes,
                total,
                caller.file(),
                caller.line()
            );
            if std::env::var("M40LLM_ALLOC_BT").ok().as_deref() == Some("1") {
                let bt = std::backtrace::Backtrace::capture();
                msg.push_str(&format!("\n{:?}", bt));
            }
            eprintln!(
                "{}{}",
                msg,
                tag.map(|t| format!(" tag={}", t)).unwrap_or_default()
            );
        }
        Ok(out)
    }

    #[track_caller]
    pub fn device_malloc(&self, bytes: usize) -> Result<*mut c_void> {
        self.device_malloc_inner(bytes, None)
    }

    #[track_caller]
    pub fn device_malloc_tagged(&self, bytes: usize, tag: &str) -> Result<*mut c_void> {
        self.device_malloc_inner(bytes, Some(tag))
    }
    /// # Safety
    /// `ptr` must be a valid device pointer previously allocated by `device_malloc` or the CUDA runtime.
    /// The memory must not be used after this call and must belong to this context/device.
    #[track_caller]
    pub unsafe fn device_free(&self, ptr: *mut c_void) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        // Pre-read for log, then free
        let before = TOTAL_DEVICE_BYTES.load(Ordering::SeqCst);
        let rc = unsafe { ffi::m40llm_device_free(self.inner.raw.as_ptr(), ptr) };
        if rc != 0 {
            return Err(anyhow!("m40llm_device_free failed: {rc}"));
        }
        crate::profile::record_device_free("device_free");
        // Decrement tracked total if we know this allocation size
        let mut dec = 0usize;
        let mut tag: Option<String> = None;
        if let Ok(mut map) = self.inner.alloc_map.lock() {
            if let Some(info) = map.remove(&ptr) {
                dec = info.size;
                tag = info.tag;
                TOTAL_DEVICE_BYTES.fetch_sub(info.size, Ordering::SeqCst);
            }
        }
        if alloc_log_enabled() {
            let after = TOTAL_DEVICE_BYTES.load(Ordering::SeqCst);
            let caller = std::panic::Location::caller();
            let mut msg = format!(
                "[cuda] device_free: ptr={:?} dec={} (total {} -> {}) at {}:{}",
                ptr,
                dec,
                before,
                after,
                caller.file(),
                caller.line()
            );
            if let Some(t) = &tag {
                msg.push_str(&format!(" tag={}", t));
            }
            if std::env::var("M40LLM_ALLOC_BT").ok().as_deref() == Some("1") {
                let bt = std::backtrace::Backtrace::capture();
                msg.push_str(&format!("\n{:?}", bt));
            }
            eprintln!("{}", msg);
        }
        Ok(())
    }
    /// # Safety
    /// `dst_device` must be a valid, writable device pointer to at least `bytes` bytes on this context's device.
    /// `src_host` must be a valid, readable host pointer to at least `bytes` bytes.
    pub unsafe fn memcpy_h2d(
        &self,
        dst_device: *mut c_void,
        src_host: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        let rc =
            unsafe { ffi::m40llm_memcpy_h2d(self.inner.raw.as_ptr(), dst_device, src_host, bytes) };
        if rc != 0 {
            return Err(anyhow!("m40llm_memcpy_h2d failed: {rc}"));
        }
        crate::profile::record_h2d_copy("memcpy_h2d", bytes);
        Ok(())
    }
    /// # Safety
    /// `dst_host` must be a valid, writable host pointer to at least `bytes` bytes.
    /// `src_device` must be a valid, readable device pointer to at least `bytes` bytes on this context's device.
    pub unsafe fn memcpy_d2h(
        &self,
        dst_host: *mut c_void,
        src_device: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        let rc =
            unsafe { ffi::m40llm_memcpy_d2h(self.inner.raw.as_ptr(), dst_host, src_device, bytes) };
        if rc != 0 {
            return Err(anyhow!("m40llm_memcpy_d2h failed: {rc}"));
        }
        crate::profile::record_d2h_copy("memcpy_d2h", bytes);
        Ok(())
    }

    /// # Safety
    /// `dst_device` and `src_device` must be valid device pointers to at least
    /// `bytes` bytes on this context's device. Regions must not overlap.
    pub unsafe fn memcpy_d2d_async(
        &self,
        dst_device: *mut c_void,
        src_device: *const c_void,
        bytes: usize,
        stream: CudaStream,
    ) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_memcpy_d2d_async(
                self.inner.raw.as_ptr(),
                dst_device,
                src_device,
                bytes,
                stream.ffi_kind(),
            )
        };
        if rc != 0 {
            return Err(anyhow!("m40llm_memcpy_d2d_async failed: {rc}"));
        }
        Ok(())
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn validate_device_ptr(&self, ptr: *const c_void) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        let rc = unsafe { ffi::m40llm_validate_device_ptr(ptr) };
        if rc != 0 {
            return Err(anyhow!("m40llm_validate_device_ptr failed: {rc}"));
        }
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
impl CudaContext {
    #[allow(dead_code)]
    pub fn device_malloc(&self, _bytes: usize) -> Result<*mut c_void> {
        let _g = self.inner.lock.lock().unwrap();
        let _ = self.inner.device_id;
        Ok(std::ptr::null_mut())
    }
    #[allow(dead_code)]
    pub fn device_free(&self, _ptr: *mut c_void) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        let _ = self.inner.device_id;
        Ok(())
    }
    #[allow(dead_code)]
    pub fn memcpy_h2d(
        &self,
        _dst_device: *mut c_void,
        _src_host: *const c_void,
        _bytes: usize,
    ) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        Ok(())
    }
    #[allow(dead_code)]
    pub fn memcpy_d2h(
        &self,
        _dst_host: *mut c_void,
        _src_device: *const c_void,
        _bytes: usize,
    ) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        Ok(())
    }

    #[allow(dead_code)]
    pub fn validate_device_ptr(&self, _ptr: *const c_void) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        Ok(())
    }
}

impl CudaContext {
    pub fn create_kvcache(
        &self,
        max_seq_len: u32,
        max_batch_size: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<*mut ffi::M40llmKVCache> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let kv = unsafe {
                ffi::m40llm_kvcache_create(
                    self.inner.raw.as_ptr(),
                    max_seq_len,
                    max_batch_size,
                    num_heads,
                    head_dim,
                )
            };
            if kv.is_null() {
                return Err(anyhow::anyhow!("m40llm_kvcache_create returned null"));
            }
            Ok(kv)
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _g = self.inner.lock.lock().unwrap();
            let _ = (max_seq_len, max_batch_size, num_heads, head_dim);
            Ok(std::ptr::null_mut())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_compressed_kvcache(
        &self,
        max_seq_len: u32,
        max_batch_size: u32,
        num_heads: u32,
        head_dim: u32,
        recent_window: u32,
        block_size: u32,
        top_blocks: u32,
        representatives: u32,
        representative_policy: crate::kv_compression::KvRepresentativePolicy,
        exact_old_backing: ExactOldBacking,
    ) -> Result<*mut ffi::M40llmKVCache> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let kv = unsafe {
                ffi::m40llm_kvcache_create_compressed(
                    self.inner.raw.as_ptr(),
                    max_seq_len,
                    max_batch_size,
                    num_heads,
                    head_dim,
                    recent_window,
                    block_size,
                    top_blocks,
                    representatives,
                    representative_policy.as_ffi(),
                    u32::from(exact_old_backing == ExactOldBacking::Q8),
                )
            };
            if kv.is_null() {
                return Err(anyhow::anyhow!(
                    "m40llm_kvcache_create_compressed returned null"
                ));
            }
            Ok(kv)
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _g = self.inner.lock.lock().unwrap();
            let _ = (
                max_seq_len,
                max_batch_size,
                num_heads,
                head_dim,
                recent_window,
                block_size,
                top_blocks,
                representatives,
                representative_policy,
                exact_old_backing,
            );
            Ok(std::ptr::null_mut())
        }
    }

    pub fn upload_weights(&self, data: &[u8]) -> Result<*mut c_void> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            // Free any previously uploaded weights to avoid leaks on re-upload
            if let Some(prev) = self.inner.weights_ptr.lock().unwrap().take() {
                // Adjust tracked totals using recorded size
                let mut wbytes = 0usize;
                if let Ok(mut m) = self.inner.alloc_map.lock() {
                    if let Some(info) = m.remove(&prev.as_ptr()) {
                        wbytes = info.size;
                    }
                }
                if wbytes > 0 {
                    TOTAL_DEVICE_BYTES.fetch_sub(wbytes, Ordering::SeqCst);
                }
                unsafe {
                    let _ = ffi::m40llm_device_free(self.inner.raw.as_ptr(), prev.as_ptr());
                }
                crate::profile::record_device_free("upload_weights.free_prev");
                eprintln!(
                    "[cuda] upload_weights: freed prev {} bytes (total={})",
                    wbytes,
                    TOTAL_DEVICE_BYTES.load(Ordering::SeqCst)
                );
            }
            let mut d_ptr: *mut c_void = std::ptr::null_mut();
            let rc = unsafe {
                ffi::m40llm_upload_weights(
                    self.inner.raw.as_ptr(),
                    data.as_ptr() as *const _,
                    data.len(),
                    &mut d_ptr as *mut _,
                )
            };
            if rc != 0 || d_ptr.is_null() {
                return Err(anyhow!(
                    "m40llm_upload_weights failed: rc={rc}, bytes={}, total_before={}",
                    data.len(),
                    TOTAL_DEVICE_BYTES.load(Ordering::SeqCst)
                ));
            }
            // Upload uses cudaMalloc + copy under the hood; conservatively track bytes
            TOTAL_DEVICE_BYTES.fetch_add(data.len(), Ordering::SeqCst);
            crate::profile::record_device_alloc("upload_weights", data.len());
            crate::profile::record_h2d_copy("upload_weights", data.len());
            if let Ok(mut map) = self.inner.alloc_map.lock() {
                map.insert(
                    d_ptr,
                    AllocInfo {
                        size: data.len(),
                        tag: Some("weights".into()),
                    },
                );
            }
            let total = TOTAL_DEVICE_BYTES.load(Ordering::SeqCst);
            eprintln!(
                "[cuda] upload_weights: {} bytes (total={}) tag=weights",
                data.len(),
                total
            );
            // Track ownership inside the context so it can be freed on drop
            let mut slot = self.inner.weights_ptr.lock().unwrap();
            *slot = NonNull::new(d_ptr);
            Ok(d_ptr)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = data;
            Ok(std::ptr::null_mut())
        }
    }

    /// # Safety
    /// `d_a_f32`, `d_b_f16`, and `d_c_f32` must be valid device pointers on this context's device.
    /// Dimensions m, n, k must match the underlying buffer shapes.
    pub unsafe fn gemm_f32xf16_f32(
        &self,
        d_a_f32: *const c_void,
        d_b_f16: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            static GEMM_LOG: Once = Once::new();
            log_gemm_backend_once(
                &GEMM_LOG,
                "m40llm_gemm_f32xf16_f32",
                if cfg!(have_cublas) {
                    "cuBLAS first; CUDA kernel fallback if cuBLAS rejects this mixed-type call"
                } else {
                    "CUDA kernel fallback"
                },
            );
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_gemm_f32xf16_f32(
                self.inner.raw.as_ptr(),
                d_a_f32,
                d_b_f16,
                d_c_f32,
                m,
                n,
                k,
            );
            if rc != 0 {
                return Err(anyhow!("m40llm_gemm_f32xf16_f32 failed: {rc}"));
            }
            record_sync_gemm("gemm_f32xf16_f32");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_a_f32, d_b_f16, d_c_f32, m, n, k);
            Ok(())
        }
    }

    /// # Safety
    /// `d_a_f32`, `d_b_f16`, and `d_c_f32` must be valid device pointers on this context's device.
    /// `d_b_f16` must be a GGUF F16 tensor with logical shape [k, n], where dimension 0 is fastest.
    pub unsafe fn gemm_f32xf16_gguf_f32(
        &self,
        d_a_f32: *const c_void,
        d_b_f16: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            static GEMM_LOG: Once = Once::new();
            log_gemm_backend_once(
                &GEMM_LOG,
                "m40llm_gemm_f32xf16_gguf_f32",
                if cfg!(have_cublas) {
                    "cuBLAS first for GGUF dimension-0-fastest weights; CUDA kernel fallback"
                } else {
                    "CUDA GGUF-layout kernel fallback"
                },
            );
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_gemm_f32xf16_gguf_f32(
                self.inner.raw.as_ptr(),
                d_a_f32,
                d_b_f16,
                d_c_f32,
                m,
                n,
                k,
            );
            if rc != 0 {
                return Err(anyhow!("m40llm_gemm_f32xf16_gguf_f32 failed: {rc}"));
            }
            record_sync_gemm("gemm_f32xf16_gguf_f32");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_a_f32, d_b_f16, d_c_f32, m, n, k);
            Ok(())
        }
    }

    /// # Safety
    /// `d_b_f32_colmajor_nt` must contain the transpose of a logical [k,n] GGUF
    /// weight, stored as column-major [n,k]. This computes row-major C = A * B.
    pub unsafe fn gemm_f32xf32_f32(
        &self,
        d_a_f32: *const c_void,
        d_b_f32_colmajor_nt: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            static GEMM_LOG: Once = Once::new();
            log_gemm_backend_once(
                &GEMM_LOG,
                "m40llm_gemm_f32xf32_f32",
                if cfg!(have_cublas) {
                    "cuBLAS sgemm materialized-f32"
                } else {
                    "unavailable without cuBLAS"
                },
            );
            self.gemm_f32xf32_f32_async(d_a_f32, d_b_f32_colmajor_nt, d_c_f32, m, n, k)?;
            self.synchronize_stream_for_op(CudaStream::Prefill, "gemm_f32xf32_f32")?;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_a_f32, d_b_f32_colmajor_nt, d_c_f32, m, n, k);
            Ok(())
        }
    }

    /// # Safety
    /// Enqueues materialized-FP32 GEMM on the prefill stream without synchronizing.
    pub unsafe fn gemm_f32xf32_f32_async(
        &self,
        d_a_f32: *const c_void,
        d_b_f32_colmajor_nt: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            static GEMM_LOG: Once = Once::new();
            log_gemm_backend_once(
                &GEMM_LOG,
                "m40llm_gemm_f32xf32_f32_async",
                if cfg!(have_cublas) {
                    "cuBLAS sgemm materialized-f32 async enqueue"
                } else {
                    "unavailable without cuBLAS"
                },
            );
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_gemm_f32xf32_f32_async(
                self.inner.raw.as_ptr(),
                d_a_f32,
                d_b_f32_colmajor_nt,
                d_c_f32,
                m,
                n,
                k,
            );
            if rc != 0 {
                return Err(anyhow!("m40llm_gemm_f32xf32_f32_async failed: {rc}"));
            }
            crate::profile::record_cublas_call("gemm_f32xf32_f32");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_a_f32, d_b_f32_colmajor_nt, d_c_f32, m, n, k);
            Ok(())
        }
    }

    /// # Safety
    /// `d_b_f16` must be a GGUF F16 tensor with logical shape [k,n], where
    /// dimension 0 is fastest. Output must have room for n*k f32 values.
    pub unsafe fn materialize_gguf_f16_to_f32_colmajor_nt(
        &self,
        d_b_f16: *const c_void,
        d_b_f32_colmajor_nt: *mut c_void,
        n: i32,
        k: i32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_materialize_gguf_f16_to_f32_colmajor_nt(
                self.inner.raw.as_ptr(),
                d_b_f16,
                d_b_f32_colmajor_nt,
                n,
                k,
            );
            if rc != 0 {
                return Err(anyhow!(
                    "m40llm_materialize_gguf_f16_to_f32_colmajor_nt failed: {rc}"
                ));
            }
            record_sync_kernel("materialize_gguf_f16_to_f32_colmajor_nt");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_b_f16, d_b_f32_colmajor_nt, n, k);
            Ok(())
        }
    }

    /// # Safety
    /// A f16 × B f16 → C f32 row-major GEMM. Device pointers must be valid on this context's device.
    pub unsafe fn gemm_f16xf16_f32(
        &self,
        d_a_f16: *const c_void,
        d_b_f16: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            static GEMM_LOG: Once = Once::new();
            log_gemm_backend_once(
                &GEMM_LOG,
                "m40llm_gemm_f16xf16_f32",
                if cfg!(have_cublas) {
                    "cuBLAS"
                } else {
                    "CUDA kernel fallback"
                },
            );
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_gemm_f16xf16_f32(
                self.inner.raw.as_ptr(),
                d_a_f16,
                d_b_f16,
                d_c_f32,
                m,
                n,
                k,
            );
            if rc != 0 {
                return Err(anyhow!("m40llm_gemm_f16xf16_f32 failed: {rc}"));
            }
            record_sync_gemm("gemm_f16xf16_f32");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_a_f16, d_b_f16, d_c_f32, m, n, k);
            Ok(())
        }
    }

    /// # Safety
    /// `d_a`, `d_b`, and `d_c` must be valid device pointers on this context's device.
    /// Dimensions m, n, k must match the underlying buffer shapes.
    pub unsafe fn gemm_f16_f32(
        &self,
        d_a: *const c_void,
        d_b: *const c_void,
        d_c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            static GEMM_LOG: Once = Once::new();
            log_gemm_backend_once(
                &GEMM_LOG,
                "m40llm_gemm_f16_storage_f32_compute",
                if cfg!(have_cublas) {
                    "cuBLAS"
                } else {
                    "CUDA kernel fallback"
                },
            );
            let _g = self.inner.lock.lock().unwrap();
            let rc = unsafe {
                ffi::m40llm_gemm_f16_storage_f32_compute(
                    self.inner.raw.as_ptr(),
                    d_a,
                    d_b,
                    d_c,
                    m,
                    n,
                    k,
                )
            };
            if rc != 0 {
                return Err(anyhow!("m40llm_gemm_f16_storage_f32_compute failed: {rc}"));
            }
            record_sync_gemm("gemm_f16_storage_f32_compute");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_a, d_b, d_c, m, n, k);
            Ok(())
        }
    }

    /// # Safety
    /// Packed buffers must use [total_tokens, heads, 64] row-major f32 layout.
    /// Offset/lens arrays must contain `batch_size` u32 entries.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_prefill_f32_gqa_varlen_head64(
        &self,
        d_q_f32: *const c_void,
        d_k_f32: *const c_void,
        d_v_f32: *const c_void,
        d_q_offsets: *const u32,
        d_kv_offsets: *const u32,
        d_q_lens: *const u32,
        d_kv_lens: *const u32,
        batch_size: u32,
        q_heads: u32,
        kv_heads: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_attention_prefill_f32_gqa_varlen_head64(
                self.inner.raw.as_ptr(),
                d_q_f32,
                d_k_f32,
                d_v_f32,
                d_q_offsets,
                d_kv_offsets,
                d_q_lens,
                d_kv_lens,
                batch_size,
                q_heads,
                kv_heads,
                d_out_f32,
            );
            if rc != 0 {
                return Err(anyhow!(
                    "m40llm_attention_prefill_f32_gqa_varlen_head64 failed: {rc}"
                ));
            }
            record_sync_kernel("attention_prefill_f32_gqa_varlen_head64");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                d_q_f32,
                d_k_f32,
                d_v_f32,
                d_q_offsets,
                d_kv_offsets,
                d_q_lens,
                d_kv_lens,
                batch_size,
                q_heads,
                kv_heads,
                d_out_f32,
            );
            Ok(())
        }
    }

    /// # Safety
    /// Enqueues packed prefill attention on the prefill stream and returns after
    /// launch validation. Call `synchronize_stream(CudaStream::Prefill)` before
    /// reading `d_out_f32` or reusing input/output buffers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_prefill_f32_gqa_varlen_head64_async(
        &self,
        d_q_f32: *const c_void,
        d_k_f32: *const c_void,
        d_v_f32: *const c_void,
        d_q_offsets: *const u32,
        d_kv_offsets: *const u32,
        d_q_lens: *const u32,
        d_kv_lens: *const u32,
        batch_size: u32,
        q_heads: u32,
        kv_heads: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_attention_prefill_f32_gqa_varlen_head64_async(
                self.inner.raw.as_ptr(),
                d_q_f32,
                d_k_f32,
                d_v_f32,
                d_q_offsets,
                d_kv_offsets,
                d_q_lens,
                d_kv_lens,
                batch_size,
                q_heads,
                kv_heads,
                d_out_f32,
            );
            if rc != 0 {
                return Err(anyhow!(
                    "m40llm_attention_prefill_f32_gqa_varlen_head64_async failed: {rc}"
                ));
            }
            record_async_kernel("attention_prefill_f32_gqa_varlen_head64");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                d_q_f32,
                d_k_f32,
                d_v_f32,
                d_q_offsets,
                d_kv_offsets,
                d_q_lens,
                d_kv_lens,
                batch_size,
                q_heads,
                kv_heads,
                d_out_f32,
            );
            Ok(())
        }
    }

    /// # Safety
    /// `d_in` and `d_out` must be valid device pointers to `rows * dim` f32 elements.
    pub unsafe fn rms_norm_f32(
        &self,
        d_in: *const c_void,
        d_out: *mut c_void,
        rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.rms_norm_f32_async(d_in, d_out, rows, dim, eps)?;
            self.synchronize_stream_for_op(CudaStream::Decode, "rms_norm_f32")
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in, d_out, rows, dim, eps);
            Ok(())
        }
    }

    /// # Safety
    /// Enqueues RMSNorm on the decode stream. Callers must synchronize before reading `d_out`.
    pub unsafe fn rms_norm_f32_async(
        &self,
        d_in: *const c_void,
        d_out: *mut c_void,
        rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_rms_norm_f32_async(
                self.inner.raw.as_ptr(),
                d_in,
                d_out,
                rows,
                dim,
                eps,
            );
            if rc != 0 {
                return Err(anyhow!("m40llm_rms_norm_f32_async failed: {rc}"));
            }
            record_async_kernel("rms_norm_f32");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in, d_out, rows, dim, eps);
            Ok(())
        }
    }

    /// # Safety
    /// `d_in`, `d_weight`, and `d_out` must be valid device pointers. `d_weight`
    /// must contain `dim` elements, with dtype code 0=F16 and 1=F32.
    pub unsafe fn rms_norm_f32_weighted(
        &self,
        d_in: *const c_void,
        d_weight: *const c_void,
        d_out: *mut c_void,
        rows: u32,
        dim: u32,
        eps: f32,
        weight_dtype: u32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.rms_norm_f32_weighted_async(d_in, d_weight, d_out, rows, dim, eps, weight_dtype)?;
            self.synchronize_stream_for_op(CudaStream::Decode, "rms_norm_f32_weighted")
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in, d_weight, d_out, rows, dim, eps, weight_dtype);
            Ok(())
        }
    }

    /// # Safety
    /// Enqueues weighted RMSNorm on the decode stream. Callers must synchronize before reading `d_out`.
    pub unsafe fn rms_norm_f32_weighted_async(
        &self,
        d_in: *const c_void,
        d_weight: *const c_void,
        d_out: *mut c_void,
        rows: u32,
        dim: u32,
        eps: f32,
        weight_dtype: u32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_rms_norm_f32_weighted_async(
                self.inner.raw.as_ptr(),
                d_in,
                d_weight,
                d_out,
                rows,
                dim,
                eps,
                weight_dtype,
            );
            if rc != 0 {
                return Err(anyhow!("m40llm_rms_norm_f32_weighted_async failed: {rc}"));
            }
            record_async_kernel("rms_norm_f32_weighted");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in, d_weight, d_out, rows, dim, eps, weight_dtype);
            Ok(())
        }
    }

    /// # Safety
    /// Applies in-place RoPE rotation to Q/K shaped [rows, num_heads * head_dim] (row-major f32).
    pub unsafe fn rope_f32(
        &self,
        d_q: *mut c_void,
        d_k: *mut c_void,
        rows: u32,
        num_heads: u32,
        head_dim: u32,
        past_len: u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.rope_f32_async(
                d_q, d_k, rows, num_heads, head_dim, past_len, freq_base, freq_scale,
            )?;
            self.synchronize_stream_for_op(CudaStream::Decode, "rope_f32")
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                d_q, d_k, rows, num_heads, head_dim, past_len, freq_base, freq_scale,
            );
            Ok(())
        }
    }

    /// # Safety
    /// Enqueues RoPE on the decode stream. Callers must synchronize before reusing Q/K.
    pub unsafe fn rope_f32_async(
        &self,
        d_q: *mut c_void,
        d_k: *mut c_void,
        rows: u32,
        num_heads: u32,
        head_dim: u32,
        past_len: u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_rope_f32_async(
                self.inner.raw.as_ptr(),
                d_q,
                d_k,
                rows,
                num_heads,
                head_dim,
                past_len,
                freq_base,
                freq_scale,
            );
            if rc != 0 {
                return Err(anyhow!("m40llm_rope_f32_async failed: {rc}"));
            }
            record_async_kernel("rope_f32");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                d_q, d_k, rows, num_heads, head_dim, past_len, freq_base, freq_scale,
            );
            Ok(())
        }
    }

    /// # Safety
    /// Applies in-place RoPE rotation to a tensor shaped [rows, num_heads * head_dim] (row-major f32).
    pub unsafe fn rope_f32_inplace(
        &self,
        d_x: *mut c_void,
        rows: u32,
        num_heads: u32,
        head_dim: u32,
        past_len: u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.rope_f32_inplace_async(
                d_x, rows, num_heads, head_dim, past_len, freq_base, freq_scale,
            )?;
            self.synchronize_stream_for_op(CudaStream::Decode, "rope_f32_inplace")
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                d_x, rows, num_heads, head_dim, past_len, freq_base, freq_scale,
            );
            Ok(())
        }
    }

    /// # Safety
    /// Enqueues in-place RoPE on the decode stream. Callers must synchronize before reusing `d_x`.
    pub unsafe fn rope_f32_inplace_async(
        &self,
        d_x: *mut c_void,
        rows: u32,
        num_heads: u32,
        head_dim: u32,
        past_len: u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_rope_f32_inplace_async(
                self.inner.raw.as_ptr(),
                d_x,
                rows,
                num_heads,
                head_dim,
                past_len,
                freq_base,
                freq_scale,
            );
            if rc != 0 {
                return Err(anyhow!("m40llm_rope_f32_inplace_async failed: {rc}"));
            }
            record_async_kernel("rope_f32_inplace");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                d_x, rows, num_heads, head_dim, past_len, freq_base, freq_scale,
            );
            Ok(())
        }
    }

    /// # Safety
    /// Enqueues in-place RoPE on the decode stream using a device-resident
    /// `past_len` parameter. Callers must synchronize before reusing `d_x`.
    pub unsafe fn rope_f32_inplace_position_dev_async(
        &self,
        d_x: *mut c_void,
        rows: u32,
        num_heads: u32,
        head_dim: u32,
        past_len_dev: *const u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_rope_f32_inplace_position_dev_async(
                self.inner.raw.as_ptr(),
                d_x,
                rows,
                num_heads,
                head_dim,
                past_len_dev,
                freq_base,
                freq_scale,
            );
            if rc != 0 {
                return Err(anyhow!(
                    "m40llm_rope_f32_inplace_position_dev_async failed: {rc}"
                ));
            }
            record_async_kernel("rope_f32_inplace_position_dev");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                d_x,
                rows,
                num_heads,
                head_dim,
                past_len_dev,
                freq_base,
                freq_scale,
            );
            Ok(())
        }
    }

    /// # Safety
    /// `d_a`, `d_b`, and `d_out` must be valid device pointers to `n` f32 elements.
    pub unsafe fn residual_add_f32(
        &self,
        d_a: *const c_void,
        d_b: *const c_void,
        d_out: *mut c_void,
        n: usize,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.residual_add_f32_async(d_a, d_b, d_out, n)?;
            self.synchronize_stream_for_op(CudaStream::Decode, "residual_add_f32")
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_a, d_b, d_out, n);
            Ok(())
        }
    }

    /// # Safety
    /// Enqueues residual add on the decode stream. Callers must synchronize before reading `d_out`.
    pub unsafe fn residual_add_f32_async(
        &self,
        d_a: *const c_void,
        d_b: *const c_void,
        d_out: *mut c_void,
        n: usize,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc =
                ffi::m40llm_residual_add_f32_async(self.inner.raw.as_ptr(), d_a, d_b, d_out, n);
            if rc != 0 {
                return Err(anyhow!("m40llm_residual_add_f32_async failed: {rc}"));
            }
            record_async_kernel("residual_add_f32");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_a, d_b, d_out, n);
            Ok(())
        }
    }

    /// # Safety
    /// `d_gate`, `d_up`, and `d_out` must be valid device pointers to `n` f32 elements.
    pub unsafe fn swiglu_f32(
        &self,
        d_gate: *const c_void,
        d_up: *const c_void,
        d_out: *mut c_void,
        n: usize,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.swiglu_f32_async(d_gate, d_up, d_out, n)?;
            self.synchronize_stream_for_op(CudaStream::Decode, "swiglu_f32")
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_gate, d_up, d_out, n);
            Ok(())
        }
    }

    /// # Safety
    /// Enqueues SwiGLU on the decode stream. Callers must synchronize before reading `d_out`.
    pub unsafe fn swiglu_f32_async(
        &self,
        d_gate: *const c_void,
        d_up: *const c_void,
        d_out: *mut c_void,
        n: usize,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_swiglu_f32_async(self.inner.raw.as_ptr(), d_gate, d_up, d_out, n);
            if rc != 0 {
                return Err(anyhow!("m40llm_swiglu_f32_async failed: {rc}"));
            }
            record_async_kernel("swiglu_f32");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_gate, d_up, d_out, n);
            Ok(())
        }
    }

    pub fn start_persistent_decode(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = unsafe { ffi::m40llm_start_persistent_decode(self.inner.raw.as_ptr()) };
            if rc != 0 {
                return Err(anyhow!("m40llm_start_persistent_decode failed: {rc}"));
            }
            record_async_kernel("persistent_decode_start");
        }
        Ok(())
    }

    /// # Safety
    /// `d_in_f32` and `d_out_f32` must point to `n` f32 values on this context's device.
    pub unsafe fn persistent_decode_submit_vec(
        &self,
        d_in_f32: *const c_void,
        d_out_f32: *mut c_void,
        n: u32,
        scale: f32,
        bias: f32,
        iterations: u32,
    ) -> Result<u32> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let mut command_id = 0u32;
            let rc = ffi::m40llm_persistent_decode_submit_vec(
                self.inner.raw.as_ptr(),
                d_in_f32,
                d_out_f32,
                n,
                scale,
                bias,
                iterations,
                &mut command_id as *mut u32,
            );
            if rc != 0 {
                return Err(anyhow!("m40llm_persistent_decode_submit_vec failed: {rc}"));
            }
            record_async_kernel("persistent_decode_submit_vec");
            Ok(command_id)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in_f32, d_out_f32, n, scale, bias, iterations);
            Ok(0)
        }
    }

    pub fn persistent_decode_poll(&self) -> Result<PersistentDecodePoll> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let mut status = 0u32;
            let mut command_id = 0u32;
            let rc = unsafe {
                ffi::m40llm_persistent_decode_poll(
                    self.inner.raw.as_ptr(),
                    &mut status as *mut u32,
                    &mut command_id as *mut u32,
                )
            };
            if rc != 0 {
                return Err(anyhow!("m40llm_persistent_decode_poll failed: {rc}"));
            }
            Ok(PersistentDecodePoll {
                status: status.into(),
                command_id,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(PersistentDecodePoll {
                status: PersistentDecodeStatus::Idle,
                command_id: 0,
            })
        }
    }

    pub fn stop_persistent_decode(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = unsafe { ffi::m40llm_stop_persistent_decode(self.inner.raw.as_ptr()) };
            if rc != 0 {
                return Err(anyhow!("m40llm_stop_persistent_decode failed: {rc}"));
            }
            crate::profile::record_stream_sync("persistent_decode_stop");
        }
        Ok(())
    }

    /// # Safety
    /// d_in_f16 and d_out_f32 must be valid device pointers on this context's device.
    pub unsafe fn f16_to_f32(
        &self,
        d_in_f16: *const c_void,
        d_out_f32: *mut c_void,
        n: usize,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_f16_to_f32(self.inner.raw.as_ptr(), d_in_f16, d_out_f32, n);
            if rc != 0 {
                return Err(anyhow!("m40llm_f16_to_f32 failed: {rc}"));
            }
            record_sync_kernel("f16_to_f32");
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in_f16, d_out_f32, n);
        }
        Ok(())
    }

    /// # Safety
    /// d_in_q80 must point to GGML Q8_0 blocks; d_out_f32 is f32 output.
    pub unsafe fn q80_to_f32(
        &self,
        d_in_q80: *const c_void,
        d_out_f32: *mut c_void,
        n: usize,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = ffi::m40llm_q80_to_f32(self.inner.raw.as_ptr(), d_in_q80, d_out_f32, n);
            if rc != 0 {
                return Err(anyhow!("m40llm_q80_to_f32 failed: {rc}"));
            }
            record_sync_kernel("q80_to_f32");
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in_q80, d_out_f32, n);
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaContextInner {
    fn drop(&mut self) {
        // Free tracked weights if present to avoid device memory leak
        if let Ok(inner) = self.weights_ptr.get_mut() {
            if let Some(ptr) = inner.take() {
                unsafe {
                    let _ = ffi::m40llm_device_free(self.raw.as_ptr(), ptr.as_ptr());
                }
            }
        }
        // SAFETY: raw was constructed from a non-null FFI pointer and is only freed here when Arc count drops to 0
        unsafe { ffi::m40llm_destroy_context(self.raw.as_ptr()) };
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq)]
pub struct CompressedKvDebugSnapshot {
    pub seq_len: u32,
    pub recent_window: u32,
    pub block_size: u32,
    pub max_blocks: u32,
    pub representatives: u32,
    pub block_counts: Vec<u32>,
    pub recent_k_f16: Vec<u16>,
    pub recent_v_f16: Vec<u16>,
    pub summary_k_acc: Vec<f32>,
    pub summary_v_acc: Vec<f32>,
    pub summary_k_f16: Vec<u16>,
    pub summary_v_f16: Vec<u16>,
    pub rep_k_f16: Vec<u16>,
    pub rep_v_f16: Vec<u16>,
    pub rep_positions: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct KVCache {
    inner: Arc<KVCacheInner>,
}

#[cfg(feature = "cuda")]
fn convert_group(raw: ffi::M40llmAttentionGroupStats) -> KvAttentionGroupStats {
    KvAttentionGroupStats {
        prob_mass: raw.prob_mass,
        logit_max: raw.logit_max,
        logit_mean: raw.logit_mean,
        count: raw.count,
    }
}

#[cfg(feature = "cuda")]
fn attention_group_name(group: u32) -> &'static str {
    match group {
        1 => "recent",
        2 => "selected_old_exact",
        3 => "summary",
        4 => "representative",
        _ => "other",
    }
}

impl KVCache {
    pub fn num_heads(&self) -> u32 {
        self.inner.num_heads
    }
    pub fn head_dim(&self) -> u32 {
        self.inner.head_dim
    }
    pub fn max_batch_size(&self) -> u32 {
        self.inner.max_batch_size
    }
    pub fn max_seq_len(&self) -> u32 {
        self.inner.max_seq_len
    }
    pub fn is_compressed(&self) -> bool {
        self.inner.compressed
    }

    #[cfg(feature = "cuda")]
    pub fn debug_compressed_snapshot(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
    ) -> Result<CompressedKvDebugSnapshot> {
        if !self.inner.compressed {
            anyhow::bail!("debug_compressed_snapshot requires compressed KV cache");
        }
        if seq_id >= self.inner.max_batch_size {
            anyhow::bail!(
                "debug_compressed_snapshot seq_id {seq_id} >= max_batch_size {}",
                self.inner.max_batch_size
            );
        }
        let elems_per_token = self.elems_per_token();
        let recent_elems = (self.inner.recent_window as usize) * elems_per_token;
        let summary_elems = (self.inner.max_blocks as usize) * elems_per_token;
        let rep_elems = (self.inner.max_blocks as usize)
            * (self.inner.representatives as usize)
            * elems_per_token;
        let rep_slots = (self.inner.max_blocks as usize) * (self.inner.representatives as usize);
        let mut snapshot = CompressedKvDebugSnapshot {
            seq_len: 0,
            recent_window: self.inner.recent_window,
            block_size: self.inner.block_size,
            max_blocks: self.inner.max_blocks,
            representatives: self.inner.representatives,
            block_counts: vec![0; self.inner.max_blocks as usize],
            recent_k_f16: vec![0; recent_elems],
            recent_v_f16: vec![0; recent_elems],
            summary_k_acc: vec![0.0; summary_elems],
            summary_v_acc: vec![0.0; summary_elems],
            summary_k_f16: vec![0; summary_elems],
            summary_v_f16: vec![0; summary_elems],
            rep_k_f16: vec![0; rep_elems],
            rep_v_f16: vec![0; rep_elems],
            rep_positions: vec![0; rep_slots],
        };
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_kvcache_debug_read_compressed_state(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                &mut snapshot.seq_len as *mut u32,
                snapshot.block_counts.as_mut_ptr(),
                snapshot.recent_k_f16.as_mut_ptr().cast::<c_void>(),
                snapshot.recent_v_f16.as_mut_ptr().cast::<c_void>(),
                snapshot.summary_k_acc.as_mut_ptr(),
                snapshot.summary_v_acc.as_mut_ptr(),
                snapshot.summary_k_f16.as_mut_ptr().cast::<c_void>(),
                snapshot.summary_v_f16.as_mut_ptr().cast::<c_void>(),
                snapshot.rep_k_f16.as_mut_ptr().cast::<c_void>(),
                snapshot.rep_v_f16.as_mut_ptr().cast::<c_void>(),
                snapshot.rep_positions.as_mut_ptr(),
            )
        };
        if rc != 0 {
            anyhow::bail!("m40llm_kvcache_debug_read_compressed_state failed: {rc}");
        }
        Ok(snapshot)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn debug_select_old_blocks(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        recent_window: u32,
        block_size: u32,
        top_blocks: u32,
    ) -> Result<(Vec<crate::kv_selection::KvSelectedBlock>, u32)> {
        const MAX_BLOCKS: u32 = 2048;
        let mut blocks = vec![0u32; MAX_BLOCKS as usize];
        let mut scores = vec![0f32; MAX_BLOCKS as usize];
        let mut starts = vec![0u32; MAX_BLOCKS as usize];
        let mut ends = vec![0u32; MAX_BLOCKS as usize];
        let mut count = 0u32;
        let mut total_old_blocks = 0u32;
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_kvcache_debug_select_old_blocks(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                q_heads,
                seq_len,
                recent_window,
                block_size,
                top_blocks,
                blocks.as_mut_ptr(),
                scores.as_mut_ptr(),
                starts.as_mut_ptr(),
                ends.as_mut_ptr(),
                &mut count as *mut u32,
                MAX_BLOCKS,
                &mut total_old_blocks as *mut u32,
            )
        };
        if rc != 0 {
            anyhow::bail!("m40llm_kvcache_debug_select_old_blocks failed: {rc}");
        }
        let selected = (0..count as usize)
            .map(|idx| crate::kv_selection::KvSelectedBlock {
                rank: idx as u32,
                block_index: blocks[idx],
                score: scores[idx],
                absolute_start: starts[idx],
                absolute_end: ends[idx],
            })
            .collect();
        Ok((selected, total_old_blocks))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn debug_attention_telemetry(
        &self,
        ctx: &CudaContext,
        mode: u32,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        recent_window: u32,
        block_size: u32,
        top_blocks: u32,
        needle_block: Option<u32>,
    ) -> Result<KvAttentionTelemetrySummary> {
        let empty_entry = ffi::M40llmAttentionTopEntry {
            group: 0,
            block_index: u32::MAX,
            token_position: u32::MAX,
            score: 0.0,
            probability: 0.0,
        };
        let empty_block_mass = ffi::M40llmAttentionBlockMass {
            block_index: u32::MAX,
            prob_mass: 0.0,
            logit_max: 0.0,
            logit_mean: 0.0,
            count: 0,
        };
        let mut raw = ffi::M40llmAttentionTelemetry {
            recent: ffi::M40llmAttentionGroupStats::default(),
            selected_old_exact: ffi::M40llmAttentionGroupStats::default(),
            summary: ffi::M40llmAttentionGroupStats::default(),
            representatives: ffi::M40llmAttentionGroupStats::default(),
            other: ffi::M40llmAttentionGroupStats::default(),
            needle_block_mass: -1.0,
            selected_block_masses: [empty_block_mass; 64],
            selected_block_mass_count: 0,
            top_entries: [empty_entry; 8],
            top_entry_count: 0,
        };
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_kvcache_debug_attention_telemetry(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                mode,
                seq_id,
                d_q_f32,
                q_heads,
                seq_len,
                recent_window,
                block_size,
                top_blocks,
                needle_block.unwrap_or(u32::MAX),
                &mut raw as *mut ffi::M40llmAttentionTelemetry,
            )
        };
        if rc != 0 {
            anyhow::bail!("m40llm_kvcache_debug_attention_telemetry failed: {rc}");
        }
        let top_entry_count = raw.top_entry_count.min(raw.top_entries.len() as u32) as usize;
        Ok(KvAttentionTelemetrySummary {
            recent: convert_group(raw.recent),
            selected_old_exact: convert_group(raw.selected_old_exact),
            summary: convert_group(raw.summary),
            representatives: convert_group(raw.representatives),
            other: convert_group(raw.other),
            needle_block_mass: (raw.needle_block_mass >= 0.0).then_some(raw.needle_block_mass),
            selected_block_masses: raw.selected_block_masses
                [..raw.selected_block_mass_count.min(64) as usize]
                .iter()
                .filter(|mass| mass.block_index != u32::MAX)
                .map(|mass| crate::kv_selection::KvAttentionBlockMass {
                    block_index: mass.block_index,
                    prob_mass: mass.prob_mass,
                    logit_max: mass.logit_max,
                    logit_mean: mass.logit_mean,
                    count: mass.count,
                })
                .collect(),
            top_entries: raw.top_entries[..top_entry_count]
                .iter()
                .map(|entry| KvAttentionTopEntry {
                    group: attention_group_name(entry.group).to_string(),
                    block_index: (entry.block_index != u32::MAX).then_some(entry.block_index),
                    token_position: (entry.token_position != u32::MAX)
                        .then_some(entry.token_position),
                    score: entry.score,
                    probability: entry.probability,
                })
                .collect(),
        })
    }

    #[cfg(feature = "cuda")]
    pub fn build_compressed_from_dense(
        &self,
        ctx: &CudaContext,
        dense: &KVCache,
        seq_len: u32,
    ) -> Result<()> {
        if !self.inner.compressed {
            anyhow::bail!("build_compressed_from_dense target must be compressed");
        }
        if dense.inner.compressed {
            anyhow::bail!("build_compressed_from_dense source must be dense");
        }
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_kvcache_build_compressed_from_dense(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                dense.inner.raw.as_ptr(),
                seq_len,
            )
        };
        if rc != 0 {
            anyhow::bail!("m40llm_kvcache_build_compressed_from_dense failed: {rc}");
        }
        record_sync_kernel("kvcache_build_compressed_from_dense");
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn build_q8_old_from_dense(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        seq_len: u32,
        recent_window: u32,
    ) -> Result<()> {
        if self.inner.compressed {
            anyhow::bail!("build_q8_old_from_dense requires a dense KV cache");
        }
        if self.inner.exact_old_backing != "q8" {
            anyhow::bail!("build_q8_old_from_dense requires M40LLM_KV_EXACT_OLD_BACKING=q8");
        }
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_kvcache_build_q8_old_from_dense(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                seq_len,
                recent_window,
            )
        };
        if rc != 0 {
            anyhow::bail!("m40llm_kvcache_build_q8_old_from_dense failed: {rc}");
        }
        record_async_kernel("kvcache_build_q8_old_from_dense");
        Ok(())
    }

    pub fn reset(&self, ctx: &CudaContext) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = ctx.inner.lock.lock().unwrap();
            let rc = unsafe {
                ffi::m40llm_kvcache_reset(ctx.inner.raw.as_ptr(), self.inner.raw.as_ptr())
            };
            if rc != 0 {
                return Err(anyhow!("m40llm_kvcache_reset failed: {rc}"));
            }
            record_sync_kernel("kvcache_reset");
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ctx;
            let mut lens = self.inner.len_by_seq.lock().unwrap();
            for len in lens.iter_mut() {
                *len = 0;
            }
            Ok(())
        }
    }
}

#[derive(Debug)]
struct KVCacheInner {
    // Layout: [seq][token][head][head_dim]
    // - seq in [0, max_batch_size)
    // - token in [0, max_seq_len)
    // - head in [0, num_heads)
    // - head_dim in [0, head_dim)
    // Strides (elements):
    //   elems_per_token = num_heads * head_dim
    //   base(seq, token) = (seq * max_seq_len + token) * elems_per_token
    #[allow(dead_code)]
    //   index(seq, token, head, dim) = base + head * head_dim + dim
    max_seq_len: u32,
    max_batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    compressed: bool,
    recent_window: u32,
    block_size: u32,
    max_blocks: u32,
    representatives: u32,
    exact_old_backing: String,
    q8_old_backing_bytes: usize,
    q8_old_backing_scale_bytes: usize,
    actual_bytes: usize,
    dense_equivalent_bytes: usize,
    #[cfg(feature = "cuda")]
    raw: NonNull<ffi::M40llmKVCache>,
    #[cfg(not(feature = "cuda"))]
    k: Mutex<Vec<half::f16>>, // length = max_seq_len * max_batch_size * elems_per_token
    #[cfg(not(feature = "cuda"))]
    v: Mutex<Vec<half::f16>>, // same length as k
    #[cfg(not(feature = "cuda"))]
    len_by_seq: Mutex<Vec<u32>>, // current length per sequence
}

#[cfg(feature = "cuda")]
unsafe impl Send for KVCacheInner {}
#[cfg(feature = "cuda")]
unsafe impl Sync for KVCacheInner {}

impl KVCache {
    pub fn new_with_context(
        ctx: &CudaContext,
        max_seq_len: u32,
        max_batch_size: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let raw = ctx.create_kvcache(max_seq_len, max_batch_size, num_heads, head_dim)?;
            Ok(KVCache {
                inner: Arc::new(KVCacheInner {
                    max_seq_len,
                    max_batch_size,
                    num_heads,
                    head_dim,
                    compressed: false,
                    recent_window: 0,
                    block_size: 0,
                    max_blocks: 0,
                    representatives: 0,
                    exact_old_backing: ExactOldBacking::Dense.as_str().to_string(),
                    q8_old_backing_bytes: 0,
                    q8_old_backing_scale_bytes: 0,
                    actual_bytes: (max_seq_len as usize)
                        * (max_batch_size as usize)
                        * (num_heads as usize)
                        * (head_dim as usize)
                        * std::mem::size_of::<half::f16>()
                        * 2,
                    dense_equivalent_bytes: (max_seq_len as usize)
                        * (max_batch_size as usize)
                        * (num_heads as usize)
                        * (head_dim as usize)
                        * std::mem::size_of::<half::f16>()
                        * 2,
                    raw: NonNull::new(raw).expect("non-null kv from ffi"),
                    #[cfg(not(feature = "cuda"))]
                    k: Mutex::new(Vec::new()),
                    #[cfg(not(feature = "cuda"))]
                    v: Mutex::new(Vec::new()),
                    #[cfg(not(feature = "cuda"))]
                    len_by_seq: Mutex::new(Vec::new()),
                }),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ctx;
            let elems_per_token = (num_heads as usize) * (head_dim as usize);
            let total_tokens = (max_seq_len as usize) * (max_batch_size as usize);
            let cap = total_tokens * elems_per_token;
            Ok(KVCache {
                inner: Arc::new(KVCacheInner {
                    max_seq_len,
                    max_batch_size,
                    num_heads,
                    head_dim,
                    compressed: false,
                    recent_window: 0,
                    block_size: 0,
                    max_blocks: 0,
                    representatives: 0,
                    exact_old_backing: "dense".to_string(),
                    q8_old_backing_bytes: 0,
                    q8_old_backing_scale_bytes: 0,
                    actual_bytes: cap * std::mem::size_of::<half::f16>() * 2,
                    dense_equivalent_bytes: cap * std::mem::size_of::<half::f16>() * 2,
                    k: Mutex::new(vec![half::f16::from_f32(0.0); cap]),
                    v: Mutex::new(vec![half::f16::from_f32(0.0); cap]),
                    len_by_seq: Mutex::new(vec![0u32; max_batch_size as usize]),
                    #[cfg(feature = "cuda")]
                    raw: NonNull::dangling(),
                }),
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_compressed_with_context(
        ctx: &CudaContext,
        max_seq_len: u32,
        max_batch_size: u32,
        num_heads: u32,
        head_dim: u32,
        recent_window: u32,
        block_size: u32,
        top_blocks: u32,
        representatives: u32,
        representative_policy: crate::kv_compression::KvRepresentativePolicy,
        exact_old_backing: ExactOldBacking,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let raw = ctx.create_compressed_kvcache(
                max_seq_len,
                max_batch_size,
                num_heads,
                head_dim,
                recent_window,
                block_size,
                top_blocks,
                representatives,
                representative_policy,
                exact_old_backing,
            )?;
            let elems_per_token = (num_heads as usize) * (head_dim as usize);
            let old_capacity = max_seq_len.saturating_sub(recent_window);
            let max_blocks = if old_capacity == 0 {
                1usize
            } else {
                (old_capacity as usize).div_ceil(block_size as usize)
            };
            let recent_bytes = (max_batch_size as usize)
                * (recent_window as usize)
                * elems_per_token
                * std::mem::size_of::<half::f16>()
                * 2;
            let summary_f16_bytes = (max_batch_size as usize)
                * max_blocks
                * elems_per_token
                * std::mem::size_of::<half::f16>()
                * 2;
            let summary_acc_bytes = (max_batch_size as usize)
                * max_blocks
                * elems_per_token
                * std::mem::size_of::<f32>()
                * 2;
            let representative_bytes = (max_batch_size as usize)
                * max_blocks
                * (representatives as usize)
                * elems_per_token
                * std::mem::size_of::<half::f16>()
                * 2;
            let representative_position_bytes = (max_batch_size as usize)
                * max_blocks
                * (representatives as usize)
                * std::mem::size_of::<u32>();
            let count_bytes = (max_batch_size as usize) * max_blocks * std::mem::size_of::<u32>();
            let seq_map_bytes = (max_batch_size as usize) * std::mem::size_of::<u32>();
            let q8_old_backing_bytes = if exact_old_backing == ExactOldBacking::Q8 {
                (max_seq_len as usize)
                    * (max_batch_size as usize)
                    * elems_per_token
                    * std::mem::size_of::<i8>()
                    * 2
            } else {
                0
            };
            let q8_old_backing_scale_bytes = if exact_old_backing == ExactOldBacking::Q8 {
                (max_seq_len as usize)
                    * (max_batch_size as usize)
                    * (num_heads as usize)
                    * std::mem::size_of::<f32>()
                    * 2
            } else {
                0
            };
            let actual_bytes = recent_bytes
                + summary_f16_bytes
                + summary_acc_bytes
                + representative_bytes
                + representative_position_bytes
                + count_bytes
                + seq_map_bytes
                + q8_old_backing_bytes
                + q8_old_backing_scale_bytes;
            let dense_equivalent_bytes = (max_seq_len as usize)
                * (max_batch_size as usize)
                * elems_per_token
                * std::mem::size_of::<half::f16>()
                * 2;
            Ok(KVCache {
                inner: Arc::new(KVCacheInner {
                    max_seq_len,
                    max_batch_size,
                    num_heads,
                    head_dim,
                    compressed: true,
                    recent_window,
                    block_size,
                    max_blocks: max_blocks as u32,
                    representatives,
                    exact_old_backing: exact_old_backing.as_str().to_string(),
                    q8_old_backing_bytes,
                    q8_old_backing_scale_bytes,
                    actual_bytes,
                    dense_equivalent_bytes,
                    raw: NonNull::new(raw).expect("non-null compressed kv from ffi"),
                    #[cfg(not(feature = "cuda"))]
                    k: Mutex::new(Vec::new()),
                    #[cfg(not(feature = "cuda"))]
                    v: Mutex::new(Vec::new()),
                    #[cfg(not(feature = "cuda"))]
                    len_by_seq: Mutex::new(Vec::new()),
                }),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                ctx,
                max_seq_len,
                max_batch_size,
                num_heads,
                head_dim,
                recent_window,
                block_size,
                top_blocks,
                representatives,
                representative_policy,
                exact_old_backing,
            );
            anyhow::bail!("compressed KV cache requires cuda")
        }
    }

    pub fn actual_bytes(&self) -> usize {
        self.inner.actual_bytes
    }

    pub fn dense_equivalent_bytes(&self) -> usize {
        self.inner.dense_equivalent_bytes
    }

    pub fn exact_old_backing(&self) -> &str {
        &self.inner.exact_old_backing
    }

    pub fn q8_old_backing_bytes(&self) -> usize {
        self.inner.q8_old_backing_bytes
    }

    pub fn q8_old_backing_scale_bytes(&self) -> usize {
        self.inner.q8_old_backing_scale_bytes
    }

    #[inline]
    pub fn elems_per_token(&self) -> usize {
        (self.inner.num_heads as usize) * (self.inner.head_dim as usize)
    }

    #[inline]
    pub fn base_offset_elems(&self, seq: u32, token: u32) -> usize {
        ((seq as usize) * (self.inner.max_seq_len as usize) + (token as usize))
            * self.elems_per_token()
    }

    #[cfg(not(feature = "cuda"))]
    fn append_token_host(
        &self,
        seq_id: u32,
        k_f32: *const c_void,
        v_f32: *const c_void,
    ) -> Result<()> {
        let elems = self.elems_per_token();
        // Determine token index from len_by_seq
        let mut lens = self.inner.len_by_seq.lock().unwrap();
        let token = lens[seq_id as usize];
        if token >= self.inner.max_seq_len {
            return Err(anyhow::anyhow!("append_token_host: seq {} full", seq_id));
        }
        // Safety: caller promises k_f32/v_f32 are valid pointers to elems f32 entries
        let k_slice = unsafe { std::slice::from_raw_parts(k_f32 as *const f32, elems) };
        let v_slice = unsafe { std::slice::from_raw_parts(v_f32 as *const f32, elems) };
        let base = self.base_offset_elems(seq_id, token);
        let mut k_lock = self.inner.k.lock().unwrap();
        let mut v_lock = self.inner.v.lock().unwrap();
        for i in 0..elems {
            k_lock[base + i] = half::f16::from_f32(k_slice[i]);
            v_lock[base + i] = half::f16::from_f32(v_slice[i]);
        }
        lens[seq_id as usize] += 1;
        Ok(())
    }

    #[inline]
    #[allow(dead_code)]
    pub fn index_elems(&self, seq: u32, token: u32, head: u32, dim: u32) -> usize {
        self.base_offset_elems(seq, token)
            + (head as usize) * (self.inner.head_dim as usize)
            + (dim as usize)
    }
}

#[cfg(feature = "cuda")]
impl KVCache {
    /// # Safety
    /// `k_dev` and `v_dev` must be valid device pointers containing one token's worth of K/V in f16 layout.
    /// `seq_id` must be in range. Context must target same device as this cache.
    pub unsafe fn append_token(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        k_dev: *const c_void,
        v_dev: *const c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_kvcache_append_token(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                k_dev,
                v_dev,
            )
        };
        if rc != 0 {
            return Err(anyhow!("m40llm_kvcache_append_token failed: {rc}"));
        }
        record_sync_kernel("kvcache_append_token");
        Ok(())
    }

    /// # Safety
    /// `k_dev_f32` and `v_dev_f32` must be valid device pointers containing one token's worth of K/V in f32 layout.
    /// They will be converted to f16 in-place in the cache. Context/device must match.
    pub unsafe fn append_token_f32(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        k_dev_f32: *const c_void,
        v_dev_f32: *const c_void,
    ) -> Result<()> {
        self.append_token_f32_async(ctx, seq_id, k_dev_f32, v_dev_f32)?;
        ctx.synchronize_stream_for_op(CudaStream::Decode, "kvcache_append_token_f32")
    }

    /// # Safety
    /// Enqueues f32-to-f16 KV append on the decode stream. Callers must synchronize before reading from the cache.
    pub unsafe fn append_token_f32_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        k_dev_f32: *const c_void,
        v_dev_f32: *const c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_kvcache_append_token_f32_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                k_dev_f32,
                v_dev_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_kvcache_append_token_f32_async failed: {rc}"
            ));
        }
        record_async_kernel("kvcache_append_token_f32");
        Ok(())
    }

    /// # Safety
    /// `k_dev_f32` and `v_dev_f32` must contain one token of K/V in f32 layout.
    /// K is RoPE-rotated for `past_len` while both K and V are converted to f16
    /// and appended to the cache. Context/device must match.
    pub unsafe fn append_token_f32_rope_k(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        k_dev_f32: *const c_void,
        v_dev_f32: *const c_void,
        past_len: u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        self.append_token_f32_rope_k_async(
            ctx, seq_id, k_dev_f32, v_dev_f32, past_len, freq_base, freq_scale,
        )?;
        ctx.synchronize_stream_for_op(CudaStream::Decode, "kvcache_append_token_f32_rope_k")
    }

    /// # Safety
    /// Enqueues fused K RoPE and f32-to-f16 KV append on the decode stream.
    /// Callers must synchronize before reading from the cache.
    pub unsafe fn append_token_f32_rope_k_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        k_dev_f32: *const c_void,
        v_dev_f32: *const c_void,
        past_len: u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_kvcache_append_token_f32_rope_k_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                k_dev_f32,
                v_dev_f32,
                past_len,
                freq_base,
                freq_scale,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_kvcache_append_token_f32_rope_k_async failed: {rc}"
            ));
        }
        record_async_kernel("kvcache_append_token_f32_rope_k");
        Ok(())
    }

    /// # Safety
    /// Enqueues fused K RoPE and f32-to-f16 KV append at an explicit cache
    /// position. Unlike `append_token_f32_rope_k_async`, this does not read the
    /// current KV length back to the host; the launched kernel writes
    /// `seq_map[seq_id] = position + 1` on device.
    pub unsafe fn append_token_f32_rope_k_at_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        k_dev_f32: *const c_void,
        v_dev_f32: *const c_void,
        position: u32,
        past_len: u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_kvcache_append_token_f32_rope_k_at_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                k_dev_f32,
                v_dev_f32,
                position,
                past_len,
                freq_base,
                freq_scale,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_kvcache_append_token_f32_rope_k_at_async failed: {rc}"
            ));
        }
        record_async_kernel("kvcache_append_token_f32_rope_k_at");
        Ok(())
    }

    /// # Safety
    /// Enqueues fused K RoPE and f32-to-f16 KV append using a device-resident
    /// position parameter. The graph can bind `position_dev` once and update the
    /// pointed-to value between launches.
    pub unsafe fn append_token_f32_rope_k_position_dev_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        k_dev_f32: *const c_void,
        v_dev_f32: *const c_void,
        position_dev: *const u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_kvcache_append_token_f32_rope_k_position_dev_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                k_dev_f32,
                v_dev_f32,
                position_dev,
                freq_base,
                freq_scale,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_kvcache_append_token_f32_rope_k_position_dev_async failed: {rc}"
            ));
        }
        record_async_kernel("kvcache_append_token_f32_rope_k_position_dev");
        Ok(())
    }

    /// # Safety
    /// `d_q_f32` and `d_out_f32` must be valid device pointers. `seq_len` must not exceed already appended tokens.
    /// Context/device must match. Shapes must align with KV cache configuration.
    pub unsafe fn attention_last_token_f32(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                seq_len,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!("m40llm_attention_last_token_f32 failed: {}", rc));
        }
        record_sync_kernel("attention_last_token_f32");
        Ok(())
    }

    /// # Safety
    /// `d_q_f32` and `d_out_f32` must be valid device pointers. `q_heads` must be a multiple of
    /// this cache's KV head count.
    pub unsafe fn attention_last_token_f32_gqa(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        self.attention_last_token_f32_gqa_async(ctx, seq_id, d_q_f32, q_heads, seq_len, d_out_f32)?;
        ctx.synchronize_stream_for_op(CudaStream::Decode, "attention_last_token_f32_gqa")
    }

    /// # Safety
    /// Enqueues GQA last-token attention on the decode stream. Callers must synchronize before reading `d_out_f32`.
    pub unsafe fn attention_last_token_f32_gqa_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32_gqa_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                q_heads,
                seq_len,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_attention_last_token_f32_gqa_async failed: {rc}"
            ));
        }
        record_async_kernel("attention_last_token_f32_gqa");
        Ok(())
    }

    /// # Safety
    /// Enqueues GQA last-token attention on the decode stream using a
    /// device-resident `seq_len` parameter. The initial implementation uses the
    /// generic attention kernel so graph replay can vary sequence length without
    /// host-updated kernel parameters.
    pub unsafe fn attention_last_token_f32_gqa_seq_len_dev_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        d_seq_len: *const u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32_gqa_seq_len_dev_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                q_heads,
                d_seq_len,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_attention_last_token_f32_gqa_seq_len_dev_async failed: {rc}"
            ));
        }
        record_async_kernel("attention_last_token_f32_gqa_seq_len_dev");
        Ok(())
    }

    /// # Safety
    /// Enqueues dense GQA last-token attention restricted to the absolute
    /// recent window `[seq_len - recent_window, seq_len)` on the decode stream.
    /// This diagnostic path preserves absolute KV positions and requires
    /// head_dim=64.
    pub unsafe fn attention_last_token_f32_gqa_dense_recent_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        recent_window: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32_gqa_dense_recent_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                q_heads,
                seq_len,
                recent_window,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_attention_last_token_f32_gqa_dense_recent_async failed: {rc}"
            ));
        }
        record_async_kernel("attention_last_token_f32_gqa_dense_recent");
        Ok(())
    }

    /// # Safety
    /// Enqueues experimental exact block-select GQA attention on the decode stream.
    /// Full exact KV remains allocated; only the attention read set is sparse.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_last_token_f32_gqa_block_select_exact_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        recent_window: u32,
        block_size: u32,
        top_blocks: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32_gqa_block_select_exact_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                q_heads,
                seq_len,
                recent_window,
                block_size,
                top_blocks,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_attention_last_token_f32_gqa_block_select_exact_async failed: {rc}"
            ));
        }
        record_async_kernel("attention_last_token_f32_gqa_block_select_exact");
        Ok(())
    }

    /// # Safety
    /// Enqueues diagnostic staged exact block-select GQA attention on the
    /// decode stream. This path first gathers the selected exact old K/V plus
    /// recent K/V into temporary compact buffers, then attends over that
    /// working set.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_last_token_f32_gqa_block_select_exact_staged_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        recent_window: u32,
        block_size: u32,
        top_blocks: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32_gqa_block_select_exact_staged_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                q_heads,
                seq_len,
                recent_window,
                block_size,
                top_blocks,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_attention_last_token_f32_gqa_block_select_exact_staged_async failed: {rc}"
            ));
        }
        record_async_kernel("attention_last_token_f32_gqa_block_select_exact_staged");
        Ok(())
    }

    /// # Safety
    /// Enqueues diagnostic staged exact block-select GQA attention using
    /// caller-owned staging buffers. The buffers must be sized for
    /// `q_heads * staging.capacity_tokens * 64` FP16 K/V entries, positions,
    /// and per-query-head counts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_last_token_f32_gqa_block_select_exact_staged_with_buffers_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        recent_window: u32,
        block_size: u32,
        top_blocks: u32,
        staging: ExactBlockStagingPtrs,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32_gqa_block_select_exact_staged_with_buffers_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                q_heads,
                seq_len,
                recent_window,
                block_size,
                top_blocks,
                staging.staged_k,
                staging.staged_v,
                staging.staged_positions,
                staging.staged_counts,
                staging.capacity_tokens,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_attention_last_token_f32_gqa_block_select_exact_staged_with_buffers_async failed: {rc}"
            ));
        }
        record_async_kernel("attention_last_token_f32_gqa_block_select_exact_staged_reuse");
        Ok(())
    }

    /// # Safety
    /// Enqueues staged exact block-select GQA attention using q8 old-token
    /// backing for selected old blocks and FP16 dense KV for recent tokens.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_last_token_f32_gqa_block_select_exact_staged_q8_old_with_buffers_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        recent_window: u32,
        block_size: u32,
        top_blocks: u32,
        staging: ExactBlockStagingPtrs,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32_gqa_block_select_exact_staged_q8_old_with_buffers_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                q_heads,
                seq_len,
                recent_window,
                block_size,
                top_blocks,
                staging.staged_k,
                staging.staged_v,
                staging.staged_positions,
                staging.staged_counts,
                staging.capacity_tokens,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_attention_last_token_f32_gqa_block_select_exact_staged_q8_old_with_buffers_async failed: {rc}"
            ));
        }
        record_async_kernel("attention_last_token_f32_gqa_block_select_exact_staged_q8_old");
        Ok(())
    }

    /// # Safety
    /// Enqueues direct exact block-select GQA attention over q8 old-token
    /// backing and FP16 recent KV without materializing staged FP16 K/V.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_last_token_f32_gqa_block_select_exact_q8_old_direct_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        recent_window: u32,
        block_size: u32,
        top_blocks: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32_gqa_block_select_exact_q8_old_direct_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                q_heads,
                seq_len,
                recent_window,
                block_size,
                top_blocks,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_attention_last_token_f32_gqa_block_select_exact_q8_old_direct_async failed: {rc}"
            ));
        }
        record_async_kernel("attention_last_token_f32_gqa_block_select_exact_q8_old_direct");
        Ok(())
    }

    /// # Safety
    /// Enqueues experimental lossy block-summary GQA attention on the decode
    /// stream. `top_blocks=0` attends all old block summaries; nonzero selects
    /// the top scoring old summaries before attending.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_last_token_f32_gqa_block_summary_lossy_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        recent_window: u32,
        block_size: u32,
        top_blocks: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32_gqa_block_summary_lossy_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                q_heads,
                seq_len,
                recent_window,
                block_size,
                top_blocks,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_attention_last_token_f32_gqa_block_summary_lossy_async failed: {rc}"
            ));
        }
        record_async_kernel("attention_last_token_f32_gqa_block_summary_lossy");
        Ok(())
    }

    /// # Safety
    /// Enqueues diagnostic compressed attention that attends only the exact recent ring.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_last_token_f32_gqa_compressed_recent_only_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32_gqa_compressed_recent_only_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                q_heads,
                seq_len,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_attention_last_token_f32_gqa_compressed_recent_only_async failed: {rc}"
            ));
        }
        record_async_kernel("attention_last_token_f32_gqa_compressed_recent_only");
        Ok(())
    }

    /// # Safety
    /// `d_seq_ids` and `d_seq_lens` must contain `batch_size` u32 entries.
    /// `d_q_f32` and `d_out_f32` are packed [batch_size, q_heads, head_dim].
    pub unsafe fn attention_last_token_f32_gqa_batched(
        &self,
        ctx: &CudaContext,
        d_seq_ids: *const u32,
        d_seq_lens: *const u32,
        batch_size: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32_gqa_batched(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                d_seq_ids,
                d_seq_lens,
                batch_size,
                d_q_f32,
                q_heads,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_attention_last_token_f32_gqa_batched failed: {rc}"
            ));
        }
        record_sync_kernel("attention_last_token_f32_gqa_batched");
        Ok(())
    }

    /// # Safety
    /// Enqueues batched GQA decode attention on the decode stream and returns
    /// after launch validation. Call `ctx.synchronize_stream(CudaStream::Decode)`
    /// before reading `d_out_f32` or reusing input/output buffers.
    pub unsafe fn attention_last_token_f32_gqa_batched_async(
        &self,
        ctx: &CudaContext,
        d_seq_ids: *const u32,
        d_seq_lens: *const u32,
        batch_size: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32_gqa_batched_async(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                d_seq_ids,
                d_seq_lens,
                batch_size,
                d_q_f32,
                q_heads,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "m40llm_attention_last_token_f32_gqa_batched_async failed: {rc}"
            ));
        }
        record_async_kernel("attention_last_token_f32_gqa_batched");
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
impl KVCache {
    pub fn append_token(
        &self,
        _ctx: &CudaContext,
        seq_id: u32,
        k_dev: *const c_void,
        v_dev: *const c_void,
    ) -> Result<()> {
        self.append_token_host(seq_id, k_dev, v_dev)
    }
    pub fn append_token_f32(
        &self,
        _ctx: &CudaContext,
        seq_id: u32,
        k_dev_f32: *const c_void,
        v_dev_f32: *const c_void,
    ) -> Result<()> {
        self.append_token_host(seq_id, k_dev_f32, v_dev_f32)
    }
    pub fn attention_last_token_f32(
        &self,
        _ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        // Pure CPU reference implementation operating on host f16 K/V, f32 compute
        let elems = self.elems_per_token();
        let num_heads = self.inner.num_heads as usize;
        let head_dim = self.inner.head_dim as usize;
        let q = unsafe { std::slice::from_raw_parts(d_q_f32 as *const f32, elems) };
        let k_lock = self.inner.k.lock().unwrap();
        let v_lock = self.inner.v.lock().unwrap();
        let mut out = vec![0f32; elems];
        let inv_sqrt = 1.0f32 / (head_dim as f32).sqrt();
        for h in 0..num_heads {
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut max_s = f32::NEG_INFINITY;
            for t in 0..(seq_len as usize) {
                let base = self.base_offset_elems(seq_id, t as u32) + h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let kf = k_lock[base + d].to_f32();
                    dot += qh[d] * kf;
                }
                let s = dot * inv_sqrt;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut denom = 0.0f32;
            let mut scores = vec![0.0f32; seq_len as usize];
            for (t, score) in scores.iter_mut().enumerate().take(seq_len as usize) {
                let base = self.base_offset_elems(seq_id, t as u32) + h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let kf = k_lock[base + d].to_f32();
                    dot += qh[d] * kf;
                }
                let s = dot * inv_sqrt;
                let e = (s - max_s).exp();
                *score = e;
                denom += e;
            }
            if denom == 0.0 {
                denom = 1.0;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, prob) in scores
                    .iter()
                    .map(|s| s / denom)
                    .enumerate()
                    .take(seq_len as usize)
                {
                    let vbase = self.base_offset_elems(seq_id, t as u32) + h * head_dim;
                    acc += prob * v_lock[vbase + d].to_f32();
                }
                out[h * head_dim + d] = acc;
            }
        }
        // Write back to out_dev
        let out_slice = unsafe { std::slice::from_raw_parts_mut(d_out_f32 as *mut f32, elems) };
        out_slice.copy_from_slice(&out);
        Ok(())
    }

    pub fn attention_last_token_f32_gqa(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        q_heads: u32,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        if q_heads != self.inner.num_heads {
            anyhow::bail!(
                "host GQA attention is not implemented: q_heads={} kv_heads={}",
                q_heads,
                self.inner.num_heads
            );
        }
        self.attention_last_token_f32(ctx, seq_id, d_q_f32, seq_len, d_out_f32)
    }
}

#[cfg(feature = "cuda")]
impl Drop for KVCacheInner {
    fn drop(&mut self) {
        unsafe { ffi::m40llm_kvcache_destroy(self.raw.as_ptr()) };
    }
}

// Public test/debug helper to read back one KV token (FP16) via FFI.
// Only available when the CUDA feature is enabled.
#[cfg(feature = "cuda")]
/// # Safety
/// `out_k_f16` and `out_v_f16` must be valid pointers to write one token's K and V (num_heads*head_dim f16 each).
/// `seq_id`/`token` must be within appended ranges; the context and cache must be on the same device.
pub unsafe fn ffi_debug_read_kv_token(
    ctx: &CudaContext,
    kv: &KVCache,
    seq_id: u32,
    token: u32,
    out_k_f16: *mut u8,
    out_v_f16: *mut u8,
) -> i32 {
    let _g = ctx.inner.lock.lock().unwrap();
    ffi::m40llm_kvcache_debug_read_token(
        ctx.inner.raw.as_ptr(),
        kv.inner.raw.as_ptr(),
        seq_id,
        token,
        out_k_f16 as *mut c_void,
        out_v_f16 as *mut c_void,
    )
}

// Host-pinned ring buffer stub. In non-CUDA environments, we just heap-allocate.
pub struct SharedRing<T> {
    pub ptr: *mut T,
    pub len: usize,
}

impl<T> SharedRing<T> {
    #[allow(dead_code)]
    pub fn new(count: usize) -> Result<Self> {
        let mut v: Vec<T> = Vec::with_capacity(count);
        let ptr = v.as_mut_ptr();
        std::mem::forget(v); // leak capacity; fine for test stubs
        Ok(Self { ptr, len: count })
    }
}
