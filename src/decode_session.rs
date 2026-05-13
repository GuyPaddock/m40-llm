#[cfg(feature = "cuda")]
use crate::cuda::{CudaContext, CudaGraphExec, CudaStream, DeviceBuffer};
#[cfg(feature = "cuda")]
use crate::infer::LoadedModel;
#[cfg(feature = "cuda")]
use crate::timing;
#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use std::ffi::c_void;

#[cfg(feature = "cuda")]
pub struct DecodeSession {
    model: *const LoadedModel,
    sequence_id: u32,
    processed_len: usize,
    can_forward: bool,
    d_x: DeviceBuffer,
    d_out: Option<DeviceBuffer>,
    d_logits: Option<DeviceBuffer>,
    d_norm_hidden: Option<DeviceBuffer>,
    d_graph_position: Option<DeviceBuffer>,
    d_graph_seq_len: Option<DeviceBuffer>,
    decode_graph: Option<CudaGraphExec>,
    graph_disabled: bool,
    last_forward_used_graph: bool,
    log_prefix: &'static str,
    step: usize,
    logged_full_forward: bool,
}

#[cfg(feature = "cuda")]
unsafe impl Send for DecodeSession {}

#[cfg(feature = "cuda")]
impl DecodeSession {
    pub fn new(
        model: &LoadedModel,
        d_model: usize,
        can_forward: bool,
        log_prefix: &'static str,
        d_x_tag: &'static str,
        d_out_tag: &'static str,
    ) -> Result<Self> {
        Self::new_for_sequence(
            model,
            0,
            d_model,
            can_forward,
            log_prefix,
            d_x_tag,
            d_out_tag,
        )
    }

    pub fn new_for_sequence(
        model: &LoadedModel,
        sequence_id: u32,
        d_model: usize,
        can_forward: bool,
        log_prefix: &'static str,
        d_x_tag: &'static str,
        d_out_tag: &'static str,
    ) -> Result<Self> {
        if can_forward && !model.kv_cache_can_address_layer_sequence(0, sequence_id) {
            anyhow::bail!(
                "decode session sequence_id {sequence_id} is not addressable by KV cache"
            );
        }
        let bytes = d_model
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("decode session d_model byte size overflow"))?;
        let d_x = DeviceBuffer::new_tagged(&model.cuda, bytes, d_x_tag)?;
        let d_out = if can_forward {
            Some(DeviceBuffer::new_tagged(&model.cuda, bytes, d_out_tag)?)
        } else {
            None
        };
        let (d_logits, d_norm_hidden) = match model.map_lm_head() {
            Ok((_name, _lm, lm_d_model, vocab, _tied)) => {
                if lm_d_model != d_model {
                    anyhow::bail!(
                        "decode session d_model {} != lm_head d_model {}",
                        d_model,
                        lm_d_model
                    );
                }
                let logits_bytes = vocab
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| anyhow::anyhow!("decode session logits byte size overflow"))?;
                let d_logits = DeviceBuffer::new_tagged(
                    &model.cuda,
                    logits_bytes,
                    "decode_session:logits_f32",
                )?;
                let d_norm_hidden = if model.map_output_norm()?.is_some() {
                    Some(DeviceBuffer::new_tagged(
                        &model.cuda,
                        bytes,
                        "decode_session:logits_norm_hidden_f32",
                    )?)
                } else {
                    None
                };
                (Some(d_logits), d_norm_hidden)
            }
            Err(_) => (None, None),
        };
        let graph_params = if can_forward && decode_graph_enabled() {
            Some((
                DeviceBuffer::new_tagged(
                    &model.cuda,
                    std::mem::size_of::<u32>(),
                    "decode_graph:position_u32",
                )?,
                DeviceBuffer::new_tagged(
                    &model.cuda,
                    std::mem::size_of::<u32>(),
                    "decode_graph:seq_len_u32",
                )?,
            ))
        } else {
            None
        };
        let (d_graph_position, d_graph_seq_len) = match graph_params {
            Some((position, seq_len)) => (Some(position), Some(seq_len)),
            None => (None, None),
        };
        Ok(Self {
            model: model as *const LoadedModel,
            sequence_id,
            processed_len: 0,
            can_forward,
            d_x,
            d_out,
            d_logits,
            d_norm_hidden,
            d_graph_position,
            d_graph_seq_len,
            decode_graph: None,
            graph_disabled: false,
            last_forward_used_graph: false,
            log_prefix,
            step: 0,
            logged_full_forward: false,
        })
    }

    fn model(&self) -> &LoadedModel {
        // DecodeSession does not own the model. CLI/server callers keep the
        // model alive for the whole session; the raw pointer avoids infecting
        // streaming response futures with non-'static borrows.
        unsafe { &*self.model }
    }

    pub fn processed_len(&self) -> usize {
        self.processed_len
    }

    pub fn sequence_id(&self) -> u32 {
        self.sequence_id
    }

    pub fn can_forward(&self) -> bool {
        self.can_forward
    }

    pub fn forward_batch_item(&self, seq_len: u32) -> Result<crate::infer::ForwardBatchItem> {
        let d_out = self
            .d_out
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("d_out is not allocated for this decode session"))?;
        Ok(crate::infer::ForwardBatchItem {
            d_x_f32: self.d_x.as_mut_ptr(),
            sequence_id: self.sequence_id,
            seq_len,
            d_out_f32: d_out.as_mut_ptr(),
        })
    }

    pub fn d_out_ptr(&self) -> Result<*mut std::ffi::c_void> {
        self.d_out
            .as_ref()
            .map(DeviceBuffer::as_mut_ptr)
            .ok_or_else(|| anyhow::anyhow!("d_out is not allocated for this decode session"))
    }

    pub unsafe fn load_next_unprocessed_token(&mut self, ids: &[u32]) -> Result<Option<usize>> {
        if self.processed_len > ids.len() {
            self.processed_len = 0;
            if self.model().kv_cache.is_some() {
                self.model().reset_kv_cache()?;
            }
        }
        if self.processed_len >= ids.len() {
            return Ok(None);
        }
        let token_idx = self.processed_len;
        let tok_id = ids[token_idx] as u64;
        let embed_start = std::time::Instant::now();
        (*self.model).load_token_embedding_to_f32(tok_id, self.d_x.as_mut_ptr())?;
        timing::timing_log!(
            embed_start.elapsed(),
            "{}.token.{token_idx}.embedding_load",
            self.log_prefix
        );
        Ok(Some(token_idx))
    }

    pub unsafe fn logits_after_batched_forward(&mut self, token_idx: usize) -> Result<Vec<f32>> {
        let d_out = self
            .d_out
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("d_out is not allocated for this decode session"))?
            .as_mut_ptr();
        let logits_start = std::time::Instant::now();
        let logits = match self.d_logits.as_ref() {
            Some(d_logits) => (*self.model).logits_from_hidden_into(
                d_out as *const _,
                d_logits.as_mut_ptr(),
                self.d_norm_hidden.as_ref().map(DeviceBuffer::as_mut_ptr),
            ),
            None => (*self.model).logits_from_hidden(d_out as *const _),
        };
        timing::timing_log!(
            logits_start.elapsed(),
            "{}.token.{token_idx}.logits",
            self.log_prefix
        );
        logits
    }

    pub fn mark_processed_through(&mut self, processed_len: usize) {
        self.processed_len = processed_len;
    }

    pub fn logits_for_ids(
        &mut self,
        ids: &[u32],
        mut on_token_logits: impl FnMut(&[f32]),
    ) -> Result<Vec<f32>> {
        let logits_fn_start = std::time::Instant::now();
        eprintln!(
            "[{}] logits_fn called with {} tokens",
            self.log_prefix,
            ids.len()
        );
        self.step += 1;
        eprintln!(
            "[mem] (token) step={} pid={} device_id={} TOTAL_DEVICE_BYTES={}",
            self.step,
            std::process::id(),
            self.model().cuda.device_id(),
            CudaContext::total_device_bytes()
        );

        if ids.is_empty() {
            anyhow::bail!("empty ids");
        }
        if self.processed_len > ids.len() {
            self.processed_len = 0;
            if self.model().kv_cache.is_some() {
                self.model().reset_kv_cache()?;
            }
        }

        let start = if self.can_forward {
            self.processed_len
        } else {
            ids.len().saturating_sub(1)
        };
        let mut logits: Option<Vec<f32>> = None;
        for (token_idx, &tok_id_u32) in ids.iter().enumerate().skip(start) {
            let token_start = std::time::Instant::now();
            let tok_id = tok_id_u32 as u64;
            eprintln!("[{}] token id {}", self.log_prefix, tok_id);

            let token_logits = unsafe { self.logits_for_token(tok_id, token_idx, start)? };
            on_token_logits(&token_logits);
            timing::timing_log!(
                token_start.elapsed(),
                "{}.token.{token_idx}.total",
                self.log_prefix
            );
            logits = Some(token_logits);
        }
        if self.can_forward {
            self.processed_len = ids.len();
        }

        let result = logits.ok_or_else(|| anyhow::anyhow!("no token processed for logits"));
        timing::timing_log!(
            logits_fn_start.elapsed(),
            "{}.logits_fn.ids_len_{}",
            self.log_prefix,
            ids.len()
        );
        result
    }

    unsafe fn logits_for_token(
        &mut self,
        tok_id: u64,
        token_idx: usize,
        start: usize,
    ) -> Result<Vec<f32>> {
        let embed_start = std::time::Instant::now();
        (*self.model).load_token_embedding_to_f32(tok_id, self.d_x.as_mut_ptr())?;
        timing::timing_log!(
            embed_start.elapsed(),
            "{}.token.{token_idx}.embedding_load",
            self.log_prefix
        );

        if self.can_forward {
            let forward_start = std::time::Instant::now();
            let d_out = self
                .d_out
                .as_ref()
                .expect("d_out allocated when full forward is enabled")
                .as_mut_ptr();
            let layers = self.forward_with_optional_graph(token_idx, d_out)?;
            timing::timing_log!(
                forward_start.elapsed(),
                "{}.token.{token_idx}.forward_all_layers",
                self.log_prefix
            );
            if token_idx == start && !self.logged_full_forward {
                eprintln!(
                    "[{}] full-layer forward enabled layers={layers}",
                    self.log_prefix
                );
                self.logged_full_forward = true;
            }
            let logits_start = std::time::Instant::now();
            if self.last_forward_used_graph {
                self.model().cuda.stream_wait_for_stream(
                    CudaStream::Prefill,
                    CudaStream::Decode,
                    "logits_wait_graph_replay",
                )?;
            }
            let logits = match self.d_logits.as_ref() {
                Some(d_logits) => (*self.model).logits_from_hidden_into(
                    d_out as *const _,
                    d_logits.as_mut_ptr(),
                    self.d_norm_hidden.as_ref().map(DeviceBuffer::as_mut_ptr),
                ),
                None => (*self.model).logits_from_hidden(d_out as *const _),
            };
            timing::timing_log!(
                logits_start.elapsed(),
                "{}.token.{token_idx}.logits",
                self.log_prefix
            );
            logits
        } else {
            let logits_start = std::time::Instant::now();
            let logits = match self.d_logits.as_ref() {
                Some(d_logits) => (*self.model).logits_from_hidden_into(
                    self.d_x.as_ptr(),
                    d_logits.as_mut_ptr(),
                    self.d_norm_hidden.as_ref().map(DeviceBuffer::as_mut_ptr),
                ),
                None => (*self.model).logits_from_hidden(self.d_x.as_ptr()),
            };
            timing::timing_log!(
                logits_start.elapsed(),
                "{}.token.{token_idx}.logits",
                self.log_prefix
            );
            logits
        }
    }

    unsafe fn forward_with_optional_graph(
        &mut self,
        token_idx: usize,
        d_out: *mut c_void,
    ) -> Result<usize> {
        let seq_len = (token_idx + 1) as u32;
        self.last_forward_used_graph = false;
        if self.graph_disabled
            || !decode_graph_enabled()
            || crate::kv_compression::runtime_config().mode.is_enabled()
        {
            return (*self.model).forward_one_token_all_layers_for_sequence(
                self.d_x.as_ptr(),
                self.sequence_id,
                seq_len,
                d_out,
            );
        }
        let position = token_idx as u32;
        let d_position = self
            .d_graph_position
            .as_ref()
            .expect("graph position allocated when graph is enabled");
        let d_seq_len = self
            .d_graph_seq_len
            .as_ref()
            .expect("graph seq_len allocated when graph is enabled");
        self.upload_graph_params(position, seq_len)?;

        if self.decode_graph.is_none() {
            // Warm once through the normal path so lazy workspace and FP32 weight
            // materialization do not occur inside CUDA stream capture.
            let warmed_layers = (*self.model).forward_one_token_all_layers_for_sequence(
                self.d_x.as_ptr(),
                self.sequence_id,
                seq_len,
                d_out,
            )?;
            let captured = (*self.model).cuda.capture_graph(CudaStream::Decode, || {
                (*self.model)
                    .forward_one_token_all_layers_for_sequence_graph_params(
                        self.d_x.as_ptr(),
                        self.sequence_id,
                        d_position.as_ptr() as *const u32,
                        d_seq_len.as_ptr() as *const u32,
                        d_out,
                    )
                    .map(|_| ())
            });
            match captured {
                Ok(graph) => {
                    eprintln!(
                        "[{}] captured full-token decode CUDA graph layers={warmed_layers}",
                        self.log_prefix
                    );
                    self.decode_graph = Some(graph);
                }
                Err(err) => {
                    eprintln!(
                        "[{}] disabling full-token decode CUDA graph after capture failure: {err:#}",
                        self.log_prefix
                    );
                    self.graph_disabled = true;
                    return Ok(warmed_layers);
                }
            }
        }

        if let Some(graph) = &self.decode_graph {
            let launch_result = if decode_graph_diagnostic_sync_enabled() {
                graph
                    .launch_timed_sync(CudaStream::Decode)
                    .map(|elapsed_ms| {
                        eprintln!(
                            "[{}] decode graph replay gpu_elapsed_ms={elapsed_ms:.3}",
                            self.log_prefix
                        );
                        if let Some(max_ms) = decode_graph_diag_max_ms() {
                            if elapsed_ms > max_ms {
                                eprintln!(
                                    "[{}] decode graph replay {:.3}ms exceeds diagnostic max {:.3}ms; disabling graph",
                                    self.log_prefix, elapsed_ms, max_ms
                                );
                                self.graph_disabled = true;
                            }
                        }
                    })
            } else {
                graph.launch(CudaStream::Decode)
            };
            if let Err(err) = launch_result {
                eprintln!(
                    "[{}] disabling full-token decode CUDA graph after launch failure: {err:#}",
                    self.log_prefix
                );
                self.graph_disabled = true;
                return (*self.model).forward_one_token_all_layers_for_sequence(
                    self.d_x.as_ptr(),
                    self.sequence_id,
                    seq_len,
                    d_out,
                );
            }
            self.last_forward_used_graph = true;
            Ok((*self.model).model_config.block_count as usize)
        } else {
            (*self.model).forward_one_token_all_layers_for_sequence(
                self.d_x.as_ptr(),
                self.sequence_id,
                seq_len,
                d_out,
            )
        }
    }

    fn upload_graph_params(&self, position: u32, seq_len: u32) -> Result<()> {
        let d_position = self
            .d_graph_position
            .as_ref()
            .expect("graph position allocated when graph is enabled");
        let d_seq_len = self
            .d_graph_seq_len
            .as_ref()
            .expect("graph seq_len allocated when graph is enabled");
        unsafe {
            self.model().cuda.memcpy_h2d(
                d_position.as_mut_ptr(),
                (&position as *const u32).cast::<c_void>(),
                std::mem::size_of::<u32>(),
            )?;
            self.model().cuda.memcpy_h2d(
                d_seq_len.as_mut_ptr(),
                (&seq_len as *const u32).cast::<c_void>(),
                std::mem::size_of::<u32>(),
            )?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn decode_graph_enabled() -> bool {
    std::env::var("M40LLM_DECODE_GRAPH")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn decode_graph_diagnostic_sync_enabled() -> bool {
    std::env::var("M40LLM_DECODE_GRAPH_DIAG_SYNC")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn decode_graph_diag_max_ms() -> Option<f32> {
    std::env::var("M40LLM_DECODE_GRAPH_DIAG_MAX_MS")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| *value > 0.0)
}
