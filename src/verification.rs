//! Dense-only research target verification. Not a generation/scheduler policy.
use crate::{
    cuda::{CudaStream, DeviceBuffer},
    decode_session::DecodeSession,
    infer::LoadedModel,
    kv_compression::{KvCompressionConfig, ScopedRuntimeConfig},
};
use anyhow::{ensure, Context, Result};

/// Exclusive mutable model ownership prevents workspace/cache interleaving.
pub struct VerificationSession<'m> {
    model: &'m mut LoadedModel,
    input: DeviceBuffer,
    hidden: DeviceBuffer,
    logits: DeviceBuffer,
    norm: Option<DeviceBuffer>,
    token: DeviceBuffer,
    width: usize,
    vocab: usize,
    committed_len: u32,
    next_token: Option<u32>,
    stop_tokens: Vec<u32>,
    is_poisoned: bool,
    direct_workspace: Option<crate::infer::DirectVerificationWorkspace>,
}

#[derive(Debug)]
pub struct VerificationResult {
    /// Accepted candidates, including a matched stop token if present.
    pub accepted_tokens: Vec<u32>,
    pub next_token: u32,
    pub has_stopped: bool,
}

/// An uncommitted attempt owns its session borrow and rolls back on drop.
pub struct VerificationAttempt<'s, 'm> {
    session: &'s mut VerificationSession<'m>,
    candidates: Vec<u32>,
    predictions: Vec<u32>,
    captured_logits: Option<Vec<Vec<f32>>>,
    start: u32,
    original_next: u32,
    is_finished: bool,
}

impl<'m> VerificationSession<'m> {
    /// Requires caller-selected dense, batch-one KV with no cloned storage.
    /// Does not allocate a replacement cache or change generation defaults.
    pub fn new(model: &'m mut LoadedModel, stop_tokens: &[u32]) -> Result<Self> {
        let kv = model
            .kv_cache
            .as_ref()
            .context("verification requires explicit dense KV; allocate it first")?;
        ensure!(!kv.is_compressed() && kv.has_exclusive_storage()
            && kv.max_batch_size()==model.model_config.block_count,
            "verification requires exclusive dense batch-one KV; compressed/aliased/multi-sequence caches are unsupported");
        ensure!(
            model.active_layer_count()? == model.model_config.block_count as usize,
            "verification requires every model layer"
        );
        let (_, _, width, vocab, _) = model.map_lm_head()?;
        let bytes = width
            .checked_mul(4)
            .context("verification hidden bytes overflow")?;
        let input = DeviceBuffer::new_tagged(&model.cuda, bytes, "verification:input")?;
        let hidden = DeviceBuffer::new_tagged(
            &model.cuda,
            bytes.checked_mul(8).context("verification rows overflow")?,
            "verification:hidden",
        )?;
        let logits = DeviceBuffer::new_tagged(
            &model.cuda,
            vocab
                .checked_mul(4)
                .context("verification logits overflow")?,
            "verification:logits",
        )?;
        let norm = if model.map_output_norm()?.is_some() {
            Some(DeviceBuffer::new_tagged(
                &model.cuda,
                bytes,
                "verification:norm",
            )?)
        } else {
            None
        };
        let token = DeviceBuffer::new_tagged(&model.cuda, 4, "verification:token")?;
        Ok(Self {
            model,
            input,
            hidden,
            logits,
            norm,
            token,
            width,
            vocab,
            committed_len: 0,
            next_token: None,
            stop_tokens: stop_tokens.to_vec(),
            is_poisoned: false,
            direct_workspace: None,
        })
    }

    /// Bounded research prototype, never selected by generation defaults.
    /// Keeps the original SGEMM constructor available only as its old control.
    pub fn new_direct_four_rows(model: &'m mut LoadedModel, stop_tokens: &[u32]) -> Result<Self> {
        let mut session = Self::new(model, stop_tokens)?;
        session.direct_workspace = Some(crate::infer::DirectVerificationWorkspace::new(
            session.model,
        )?);
        Ok(session)
    }

    pub fn committed_len(&self) -> u32 {
        self.committed_len
    }
    pub fn next_token(&self) -> Option<u32> {
        self.next_token
    }
    pub fn scratch_bytes(&self) -> usize {
        self.input.bytes()
            + self.hidden.bytes()
            + self.logits.bytes()
            + self.token.bytes()
            + self.norm.as_ref().map_or(0, DeviceBuffer::bytes)
            + self.direct_workspace.as_ref().map_or(0, |w| w.bytes())
    }

    /// Same packed-prefix/final-token preparation for serial and verified runs.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        ensure!(!self.is_poisoned, "verification session is poisoned");
        let capacity = self.model.kv_cache.as_ref().unwrap().max_seq_len();
        ensure!(
            !tokens.is_empty() && tokens.len() <= capacity as usize,
            "invalid verification prompt length"
        );
        let _scope = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
        self.model.reset_kv_cache()?;
        self.committed_len = 0;
        self.next_token = None;
        let result = (|| {
            let mut decode = DecodeSession::new(
                self.model,
                self.width,
                true,
                "verification",
                "verification:prefill-input",
                "verification:prefill-output",
            )?;
            decode.prefill_packed_prefix_ids(tokens)?;
            decode.greedy_token_for_ids(tokens)
        })();
        match result {
            Ok(next) => {
                self.committed_len = tokens.len() as u32;
                self.next_token = Some(next);
                Ok(next)
            }
            Err(error) => {
                self.is_poisoned = true;
                Err(error)
            }
        }
    }

    /// One ordinary qualified target decode step, including logits/greedy.
    pub fn serial_step(&mut self, token: u32) -> Result<u32> {
        ensure!(
            !self.is_poisoned && self.next_token.is_some(),
            "prefill a healthy session first"
        );
        ensure!(
            self.committed_len < self.model.kv_cache.as_ref().unwrap().max_seq_len(),
            "verification capacity exceeded"
        );
        let _scope = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
        let result = unsafe {
            (|| {
                self.model
                    .load_token_embedding_to_f32(token as u64, self.input.as_mut_ptr())?;
                self.model.forward_one_token_all_layers_for_sequence(
                    self.input.as_ptr(),
                    0,
                    self.committed_len + 1,
                    self.hidden.as_mut_ptr(),
                )?;
                self.read_prediction(0, false).map(|(next, _)| next)
            })()
        };
        match result {
            Ok(next) => {
                self.committed_len += 1;
                self.next_token = Some(next);
                Ok(next)
            }
            Err(error) => {
                self.is_poisoned = true;
                Err(error)
            }
        }
    }

    /// Diagnostic full logits; primary timing uses `serial_step` instead.
    pub fn serial_step_with_logits(&mut self, token: u32) -> Result<(u32, Vec<f32>)> {
        let next = self.serial_step(token)?;
        Ok((next, self.copy_logits()?))
    }

    fn copy_logits(&self) -> Result<Vec<f32>> {
        let mut values = vec![0f32; self.vocab];
        unsafe {
            self.model.cuda.memcpy_d2h(
                values.as_mut_ptr().cast(),
                self.logits.as_ptr(),
                self.vocab * 4,
            )?;
        }
        Ok(values)
    }

    /// Batch real candidate rows, leaving their KV uncommitted until consumed.
    /// Optional full logits are diagnostics, never enabled in primary timing.
    pub fn begin<'s>(
        &'s mut self,
        candidates: &[u32],
        capture_logits: bool,
    ) -> Result<VerificationAttempt<'s, 'm>> {
        ensure!(!self.is_poisoned, "verification session is poisoned");
        let original_next = self.next_token.context("prefill before verification")?;
        ensure!(
            self.direct_workspace.is_none() || candidates.len() == 4,
            "direct verification requires exactly four candidates"
        );
        ensure!(
            matches!(candidates.len(), 2 | 4 | 8),
            "verification requires 2, 4 or 8 candidates"
        );
        ensure!(
            candidates.iter().all(|&id| (id as usize) < self.vocab),
            "verification candidate outside vocabulary"
        );
        let start = self.committed_len;
        ensure!(
            start
                .checked_add(candidates.len() as u32)
                .is_some_and(|end| end <= self.model.kv_cache.as_ref().unwrap().max_seq_len()),
            "verification exceeds KV capacity"
        );
        let mut attempt = VerificationAttempt {
            session: self,
            candidates: candidates.to_vec(),
            predictions: Vec::with_capacity(candidates.len()),
            captured_logits: capture_logits.then(Vec::new),
            start,
            original_next,
            is_finished: false,
        };
        let _scope = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
        unsafe {
            if let Some(workspace) = &attempt.session.direct_workspace {
                attempt.session.model.forward_direct_verification_rows(
                    candidates,
                    start,
                    attempt.session.hidden.as_mut_ptr(),
                    workspace,
                )?;
            } else {
                attempt.session.model.forward_verification_rows(
                    candidates,
                    start,
                    attempt.session.hidden.as_mut_ptr(),
                )?;
            }
        }
        for row in 0..candidates.len() {
            let (prediction, logits) = attempt.session.read_prediction(row, capture_logits)?;
            attempt.predictions.push(prediction);
            if let (Some(capture), Some(logits)) = (&mut attempt.captured_logits, logits) {
                capture.push(logits);
            }
        }
        Ok(attempt)
    }

    fn read_prediction(&self, row: usize, capture_logits: bool) -> Result<(u32, Option<Vec<f32>>)> {
        let hidden = unsafe {
            self.hidden
                .as_ptr()
                .cast::<u8>()
                .add(row * self.width * 4)
                .cast()
        };
        let next = unsafe {
            self.model.greedy_token_from_hidden_into(
                hidden,
                self.logits.as_mut_ptr(),
                self.norm.as_ref().map(DeviceBuffer::as_mut_ptr),
                self.token.as_mut_ptr(),
            )?
        };
        let logits = if capture_logits {
            Some(self.copy_logits()?)
        } else {
            None
        };
        Ok((next, logits))
    }

    fn drain(&self) -> Result<()> {
        self.model.cuda.synchronize_stream(CudaStream::Decode)?;
        self.model.cuda.synchronize_stream(CudaStream::Prefill)
    }

    fn truncate(&mut self, new_len: u32, old_len: u32) -> Result<()> {
        self.drain()?;
        unsafe {
            self.model.kv_cache.as_ref().unwrap().truncate_dense_async(
                &self.model.cuda,
                0,
                self.model.model_config.block_count,
                new_len,
                old_len,
            )?;
        }
        self.model.cuda.synchronize_stream(CudaStream::Decode)
    }

    /// Owned copies for rollback/parity diagnostics, never raw cache handles.
    pub fn debug_lengths(&self) -> Result<Vec<u32>> {
        let kv = self.model.kv_cache.as_ref().unwrap();
        (0..self.model.model_config.block_count)
            .map(|slot| kv.debug_sequence_length(&self.model.cuda, slot))
            .collect()
    }

    /// Inspect allocated dense slots, including a cleared rejected suffix.
    pub fn debug_kv_token(&self, layer: u32, position: u32) -> Result<(Vec<u16>, Vec<u16>)> {
        let kv = self.model.kv_cache.as_ref().unwrap();
        ensure!(
            layer < self.model.model_config.block_count && position < kv.max_seq_len(),
            "invalid verification debug position"
        );
        self.drain()?;
        let mut k = vec![0u16; kv.elems_per_token()];
        let mut v = k.clone();
        let rc = unsafe {
            crate::cuda::ffi_debug_read_kv_token(
                &self.model.cuda,
                kv,
                layer,
                position,
                k.as_mut_ptr().cast(),
                v.as_mut_ptr().cast(),
            )
        };
        ensure!(rc == 0, "verification KV debug copy failed: {rc}");
        Ok((k, v))
    }
}

impl VerificationAttempt<'_, '_> {
    pub fn predictions(&self) -> &[u32] {
        &self.predictions
    }
    pub fn captured_logits(&self) -> Option<&[Vec<f32>]> {
        self.captured_logits.as_deref()
    }

    pub fn accepted_len(&self) -> usize {
        let mut count = 0;
        for (index, &token) in self.candidates.iter().enumerate() {
            let expected = if index == 0 {
                self.original_next
            } else {
                self.predictions[index - 1]
            };
            if token != expected {
                break;
            }
            count += 1;
            if self.session.stop_tokens.contains(&token) {
                break;
            }
        }
        count
    }

    pub fn commit(self) -> Result<VerificationResult> {
        let accepted = self.accepted_len();
        self.commit_prefix(accepted)
    }

    /// A shorter verified prefix may be committed; unverified tokens cannot.
    pub fn commit_prefix(mut self, accepted: usize) -> Result<VerificationResult> {
        ensure!(
            accepted <= self.accepted_len(),
            "cannot commit an unverified candidate"
        );
        let next = if accepted == 0 {
            self.original_next
        } else {
            self.predictions[accepted - 1]
        };
        if accepted < self.candidates.len() {
            self.session.truncate(
                self.start + accepted as u32,
                self.start + self.candidates.len() as u32,
            )?;
        }
        self.session.committed_len = self.start + accepted as u32;
        self.session.next_token = Some(next);
        self.is_finished = true;
        Ok(VerificationResult {
            accepted_tokens: self.candidates[..accepted].to_vec(),
            next_token: next,
            has_stopped: accepted > 0
                && self
                    .session
                    .stop_tokens
                    .contains(&self.candidates[accepted - 1]),
        })
    }
}

impl Drop for VerificationAttempt<'_, '_> {
    fn drop(&mut self) {
        if !self.is_finished {
            if let Err(error) = self
                .session
                .truncate(self.start, self.start + self.candidates.len() as u32)
            {
                self.session.is_poisoned = true;
                eprintln!("[verification] rollback failed; session poisoned: {error:#}");
            }
        }
    }
}

impl Drop for VerificationSession<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.drain() {
            eprintln!("[verification] final drain failed: {error:#}");
        }
    }
}
