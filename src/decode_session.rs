#[cfg(feature = "cuda")]
use crate::cuda::{CudaContext, DeviceBuffer};
#[cfg(feature = "cuda")]
use crate::infer::LoadedModel;
#[cfg(feature = "cuda")]
use crate::timing;
#[cfg(feature = "cuda")]
use anyhow::Result;

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
        Ok(Self {
            model: model as *const LoadedModel,
            sequence_id,
            processed_len: 0,
            can_forward,
            d_x,
            d_out,
            d_logits,
            d_norm_hidden,
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
        for token_idx in start..ids.len() {
            let token_start = std::time::Instant::now();
            let tok_id = ids[token_idx] as u64;
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
            let layers = (*self.model).forward_one_token_all_layers_for_sequence(
                self.d_x.as_ptr(),
                self.sequence_id,
                (token_idx + 1) as u32,
                d_out,
            )?;
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
}
