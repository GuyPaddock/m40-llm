use super::super::fixture::source_bytes;
use super::reference::{decode_q8, fit_error};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, DeviceBuffer},
    evaluation::{
        ffn_capture::FfnTensorSource,
        identity::sha256,
        quantized_fixture::{decode_q4_values, requantize_q8_to_q4},
    },
    infer::{
        execution_weights::{
            DeviceQ8ExecutionWeight, HostQ8ExecutionWeight, Q8SourceIdentity, Q8SourceView,
        },
        LoadedModel,
    },
    shiftadd::ShiftAddMatrix,
};
use serde_json::{json, Value};
use std::time::Instant;

pub(super) struct Weights {
    pub sa: ShiftAddMatrix,
    pub sa_values: Vec<f64>,
    pub q4_values: Vec<f64>,
    pub q8_values: Vec<f64>,
    pub sa_device: DeviceBuffer,
    pub q4_device: DeviceBuffer,
    pub q8_device: DeviceBuffer,
    pub prepared: DeviceQ8ExecutionWeight,
    q4_bytes: Vec<u8>,
    q8_bytes: Vec<u8>,
    prepared_host: HostQ8ExecutionWeight,
    pub report: Value,
}

fn upload(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer> {
    let device = DeviceBuffer::new_tagged(ctx, bytes.len(), "research:shiftadd-weights")?;
    unsafe {
        ctx.memcpy_h2d(device.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(device)
}

impl Weights {
    pub fn new(model: &LoadedModel, source: &FfnTensorSource) -> Result<Self> {
        let [k, n] = source.shape;
        ensure!(
            k.is_multiple_of(256) && k * n <= 2048 * 11008,
            "outside frozen full-projection fit budget"
        );
        let began = Instant::now();
        let q8_bytes = source_bytes(model, source)?.to_vec();
        ensure!(
            sha256(&q8_bytes)? == source.source_sha256,
            "source identity mismatch"
        );
        let original = decode_q8(&q8_bytes)?;
        let decoded_ms = began.elapsed().as_secs_f64() * 1000.0;
        let began = Instant::now();
        let sa = ShiftAddMatrix::fit(k, n, &original)?;
        let sa_fit_ms = began.elapsed().as_secs_f64() * 1000.0;
        let began = Instant::now();
        let (q4_bytes, q4_report) = requantize_q8_to_q4(&q8_bytes)?;
        let q4_fit_ms = began.elapsed().as_secs_f64() * 1000.0;
        let q8_values: Vec<_> = original.into_iter().map(f64::from).collect();
        let sa_values: Vec<_> = sa.decode().into_iter().map(f64::from).collect();
        let q4_values = decode_q4_values(&q4_bytes)?;
        let sa_error = fit_error(&q8_values, &sa_values)?;
        let q4_error = fit_error(&q8_values, &q4_values)?;
        let began = Instant::now();
        let tensor = model
            .gguf
            .tensors
            .iter()
            .find(|v| v.name == source.name)
            .context("missing source tensor")?;
        let stats = model.q8_execution_stats();
        let digest = *stats
            .tensors
            .first()
            .context("missing prepared model identity")?
            .key
            .source
            .model_sha256();
        let identity = Q8SourceIdentity::from_gguf_tensor(digest, tensor)?;
        let prepared_host = HostQ8ExecutionWeight::separate_scales_values(
            Q8SourceView::from_tensor_bytes(&identity, &q8_bytes)?,
        )?;
        let prepared_ms = began.elapsed().as_secs_f64() * 1000.0;
        let before = CudaContext::total_device_bytes();
        let began = Instant::now();
        let sa_device = upload(&model.cuda, sa.get_bytes())?;
        let q4_device = upload(&model.cuda, &q4_bytes)?;
        let q8_device = upload(&model.cuda, &q8_bytes)?;
        let prepared = DeviceQ8ExecutionWeight::upload(&model.cuda, &prepared_host)?;
        let upload_ms = began.elapsed().as_secs_f64() * 1000.0;
        let bytes = sa.get_bytes().len()
            + q4_bytes.len()
            + q8_bytes.len()
            + prepared_host.plan().resident_bytes();
        ensure!(
            CudaContext::total_device_bytes() == before + bytes,
            "weight allocation accounting mismatch"
        );
        let report = json!({"source":source,"sa_sha256":sha256(sa.get_bytes())?,"q4":q4_report,
            "fit_error":{"shiftadd":sa_error,"q4":q4_error},
            "preparation_ms":{"source_copy_hash_decode":decoded_ms,"shiftadd_fit":sa_fit_ms,"q4_fit":q4_fit_ms,"q8_split":prepared_ms,"allocation_upload":upload_ms},
            "bytes":{"sa":sa.get_bytes().len(),"sa_signs":k*n/256*128,"sa_amplitudes":k*n/256*8,"sa_scales":k*n/256*2,
                "q4":q4_bytes.len(),"q8":q8_bytes.len(),"prepared_q8":prepared_host.plan().resident_bytes(),"all_device_weights":bytes,
                "retained_cpu_decoded":(sa_values.len()+q4_values.len()+q8_values.len())*8,
                "retained_cpu_encoded":bytes},
            "estimated_unique_weight_sectors_32B":{"shiftadd":sa.get_bytes().len().div_ceil(32),"q4":q4_bytes.len().div_ceil(32),"q8":q8_bytes.len().div_ceil(32)},
            "sector_scope":"unique aligned backing sectors for one complete weight sweep; excludes repeated lane/warp requests, activation reads and cache effects; not measured transactions",
            "dynamic_counters":"unavailable","prepared_key":prepared_host.key()});
        Ok(Self {
            sa,
            sa_values,
            q4_values,
            q8_values,
            sa_device,
            q4_device,
            q8_device,
            prepared,
            q4_bytes,
            q8_bytes,
            prepared_host,
            report,
        })
    }

    pub fn verify(&self, ctx: &CudaContext) -> Result<()> {
        for (device, expected) in [
            (&self.sa_device, self.sa.get_bytes()),
            (&self.q4_device, &self.q4_bytes),
            (&self.q8_device, &self.q8_bytes),
        ] {
            let mut actual = vec![0; expected.len()];
            unsafe {
                ctx.memcpy_d2h(actual.as_mut_ptr().cast(), device.as_ptr(), actual.len())?;
            }
            ensure!(actual == expected, "immutable encoded weights changed");
        }
        let planes = unsafe { self.prepared.read_planes()? };
        ensure!(
            planes == self.prepared_host.planes(),
            "immutable prepared weights changed"
        );
        Ok(())
    }
}
