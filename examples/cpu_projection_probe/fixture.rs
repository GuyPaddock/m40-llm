use crate::kernel::Q8Matrix;
use anyhow::{ensure, Context, Result};
use m40_llm::{
    evaluation::{
        ffn_capture::{FfnCaptureArtifact, FfnTensorSource},
        identity::{file_sha256, sha256},
    },
    gguf::{load_gguf, GgufModel},
};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub const MODEL_SHA: &str = "4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d";
pub const INPUT_SHA: &str = "eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687";
pub const REPORT_SHA: &str = "d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801";

pub struct Fixture {
    pub artifact: FfnCaptureArtifact,
    pub gguf: GgufModel,
    file: File,
}

impl Fixture {
    pub fn load(model: &Path, input: &Path, report: &Path) -> Result<Self> {
        // This bounded study admits exactly the retained capture, not new data.
        ensure!(file_sha256(model)? == MODEL_SHA, "model SHA differs");
        ensure!(
            file_sha256(report)? == REPORT_SHA,
            "capture report SHA differs"
        );
        ensure!(
            fs::metadata(input)?.len() <= 64 * 1024 * 1024,
            "input artifact too large"
        );
        let text = fs::read_to_string(input)?;
        ensure!(
            sha256(text.as_bytes())? == INPUT_SHA,
            "capture input SHA differs"
        );
        let artifact = FfnCaptureArtifact::parse(&text)?;
        ensure!(
            artifact.plan.model_sha256 == MODEL_SHA
                && artifact.plan.selected_layer_ids == [0, 17, 35]
                && artifact.plan.d_model == 2048
                && artifact.plan.hidden_dim == 11008,
            "capture dimensions differ"
        );
        for layer in &artifact.layers {
            ensure!(layer.rows.len() >= 8, "capture needs eight distinct rows");
            for (values, width) in [
                (&layer.normalized_inputs, 2048),
                (&layer.swiglu_inputs, 11008),
            ] {
                for a in 0..8 {
                    for b in a + 1..8 {
                        ensure!(
                            values[a * width..(a + 1) * width]
                                .iter()
                                .zip(&values[b * width..(b + 1) * width])
                                .any(|(x, y)| x.to_bits() != y.to_bits()),
                            "duplicated activation rows"
                        );
                    }
                }
            }
        }
        Ok(Self {
            artifact,
            gguf: load_gguf(model)?,
            file: File::open(model)?,
        })
    }

    pub fn read_matrix(&mut self, source: &FfnTensorSource) -> Result<Q8Matrix> {
        let tensor = self
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == source.name)
            .context("tensor missing")?;
        ensure!(
            tensor.dtype == source.dtype
                && tensor.offset == source.byte_offset
                && tensor.shape == source.shape.map(|v| v as u64),
            "tensor identity differs"
        );
        let mut bytes = vec![0u8; source.byte_len()?];
        self.file.seek(SeekFrom::Start(
            self.gguf
                .data_offset
                .checked_add(source.byte_offset)
                .context("file offset overflow")?,
        ))?;
        self.file.read_exact(&mut bytes)?;
        ensure!(
            sha256(&bytes)? == source.source_sha256,
            "tensor payload SHA differs"
        );
        Q8Matrix::new(source.shape[0], source.shape[1], bytes)
    }
}
