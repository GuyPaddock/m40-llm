//! Head-major, bit-preserving fixtures and independent full-context reference.
// The standalone CPU admission tool precedes the dependent GPU runner.
#![allow(dead_code)]

use super::capture::{LayerSample, KV_HEADS, QUERY_HEADS};
use anyhow::{ensure, Result};
use half::f16;
use m40_llm::lossless_kv::LosslessKvBlocks;
use serde_json::{json, Value};

pub const DIMENSION: usize = 128;

pub struct Fixture {
    pub length: usize,
    pub sequences: usize,
    pub keys: Vec<Vec<u16>>,
    pub values: Vec<Vec<u16>>,
    pub queries: Vec<f32>,
}

pub struct Encoded {
    pub keys: Vec<LosslessKvBlocks>,
    pub values: Vec<LosslessKvBlocks>,
}

impl Encoded {
    pub fn encode(keys: &[Vec<u16>], values: &[Vec<u16>]) -> Result<Self> {
        ensure!(
            !keys.is_empty() && keys.len() == values.len(),
            "invalid stream count"
        );
        let encode = |streams: &[Vec<u16>]| {
            streams
                .iter()
                .map(|words| LosslessKvBlocks::encode(words))
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            keys: encode(keys)?,
            values: encode(values)?,
        })
    }
    pub fn verify(&self, fixture: &Fixture) -> Result<()> {
        ensure!(
            self.keys.len() == fixture.keys.len() && self.values.len() == fixture.values.len(),
            "encoded stream count differs"
        );
        for (encoded, source) in self
            .keys
            .iter()
            .chain(&self.values)
            .zip(fixture.keys.iter().chain(&fixture.values))
        {
            ensure!(
                encoded.decode()? == *source,
                "CPU reconstruction changed original FP16 bits"
            );
        }
        Ok(())
    }
    pub fn describe(&self) -> Value {
        let streams: Vec<_> = self
            .keys
            .iter()
            .chain(&self.values)
            .map(|value| value.get_stats())
            .collect();
        let original: usize = streams.iter().map(|v| v.original_bytes).sum();
        let encoded: usize = streams.iter().map(|v| v.encoded_bytes).sum();
        let directory = streams.len() * std::mem::size_of::<*const u8>();
        json!({"streams":streams,"original_bytes":original,"encoded_stream_bytes":encoded,
            "device_pointer_directory_bytes":directory,"device_representation_bytes":encoded+directory,
            "net_device_saved_bytes":original as i64-encoded as i64-directory as i64})
    }
}

impl Fixture {
    pub fn from_sample(sample: &LayerSample, length: usize) -> Result<Self> {
        let sequences = sample.keys.len();
        ensure!(
            (1..=8).contains(&sequences)
                && length > 0
                && length <= 8192
                && sample.values.len() == sequences
                && sample.queries.len() == sequences,
            "invalid fixture shape"
        );
        for sequence in 0..sequences {
            ensure!(
                sample.keys[sequence].len() == length * KV_HEADS * DIMENSION
                    && sample.values[sequence].len() == sample.keys[sequence].len()
                    && sample.queries[sequence].len() == QUERY_HEADS * DIMENSION,
                "fixture row shape differs"
            );
        }
        let split = |rows: &[Vec<f32>]| -> Result<Vec<Vec<u16>>> {
            let mut streams = Vec::new();
            for row in rows {
                ensure!(
                    row.iter().all(|value| value.is_finite()
                        && f16::from_f32(*value).to_f32().to_bits() == value.to_bits()),
                    "source must retain finite FP16 bits"
                );
                for head in 0..KV_HEADS {
                    let mut words = Vec::with_capacity(length * DIMENSION);
                    for token in 0..length {
                        words.extend(
                            row[token * KV_HEADS * DIMENSION + head * DIMENSION
                                ..token * KV_HEADS * DIMENSION + (head + 1) * DIMENSION]
                                .iter()
                                .map(|value| f16::from_f32(*value).to_bits()),
                        );
                    }
                    streams.push(words);
                }
            }
            Ok(streams)
        };
        ensure!(
            sample.queries.iter().flatten().all(|v| v.is_finite()),
            "non-finite query"
        );
        Ok(Self {
            length,
            sequences,
            keys: split(&sample.keys)?,
            values: split(&sample.values)?,
            queries: sample.queries.iter().flatten().copied().collect(),
        })
    }
    pub fn encode(&self) -> Result<Encoded> {
        let result = Encoded::encode(&self.keys, &self.values)?;
        result.verify(self)?;
        Ok(result)
    }
    pub fn get_last_tokens(&self, is_key: bool) -> Vec<f32> {
        let streams = if is_key { &self.keys } else { &self.values };
        streams
            .iter()
            .flat_map(|stream| {
                stream[(self.length - 1) * DIMENSION..]
                    .iter()
                    .map(|word| f16::from_bits(*word).to_f32())
            })
            .collect()
    }
    pub fn get_prefix(&self, batch: usize, is_key: bool) -> Result<Vec<Vec<u16>>> {
        ensure!(
            batch > 0 && batch <= self.sequences && self.length > 1,
            "invalid append prefix"
        );
        let streams = if is_key { &self.keys } else { &self.values };
        Ok(streams[..batch * KV_HEADS]
            .iter()
            .map(|v| v[..(self.length - 1) * DIMENSION].to_vec())
            .collect())
    }
    pub fn append_tokens(streams: &mut [Vec<u16>], tokens: &[f32]) -> Result<()> {
        ensure!(
            tokens.len() == streams.len() * DIMENSION
                && tokens
                    .iter()
                    .all(|v| v.is_finite() && f16::from_f32(*v).to_f32().to_bits() == v.to_bits()),
            "invalid append token values"
        );
        for (stream, token) in streams.iter_mut().zip(tokens.chunks_exact(DIMENSION)) {
            stream.extend(token.iter().map(|v| f16::from_f32(*v).to_bits()));
        }
        Ok(())
    }
    pub fn calculate_reference(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.queries.len());
        for sequence in 0..self.sequences {
            for head in 0..QUERY_HEADS {
                let stream = sequence * KV_HEADS + head / (QUERY_HEADS / KV_HEADS);
                let query =
                    &self.queries[(sequence * QUERY_HEADS + head) * DIMENSION..][..DIMENSION];
                let scores: Vec<_> = self.keys[stream]
                    .chunks_exact(DIMENSION)
                    .map(|row| {
                        row.iter()
                            .zip(query)
                            .map(|(word, q)| {
                                f64::from(f16::from_bits(*word).to_f32()) * f64::from(*q)
                            })
                            .sum::<f64>()
                            / (DIMENSION as f64).sqrt()
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|v| (v - maximum).exp()).collect();
                let total: f64 = weights.iter().sum();
                for channel in 0..DIMENSION {
                    result.push(
                        weights
                            .iter()
                            .enumerate()
                            .map(|(token, w)| {
                                w * f64::from(
                                    f16::from_bits(
                                        self.values[stream][token * DIMENSION + channel],
                                    )
                                    .to_f32(),
                                )
                            })
                            .sum::<f64>()
                            / total,
                    );
                }
            }
        }
        result
    }
}

pub fn encode_words(words: &[Vec<u16>]) -> Vec<u8> {
    words
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> LayerSample {
        let keys = (0..2)
            .map(|sequence| {
                (0..3 * KV_HEADS * DIMENSION)
                    .map(|i| {
                        if i == 0 {
                            -0.0
                        } else {
                            (sequence + i / (KV_HEADS * DIMENSION)) as f32
                        }
                    })
                    .collect()
            })
            .collect();
        let values = (0..2)
            .map(|sequence| {
                (0..3 * KV_HEADS * DIMENSION)
                    .map(|i| {
                        (sequence * 16
                            + (i / DIMENSION) % KV_HEADS * 4
                            + i / (KV_HEADS * DIMENSION)) as f32
                    })
                    .collect()
            })
            .collect();
        LayerSample {
            keys,
            values,
            queries: vec![vec![0.0; QUERY_HEADS * DIMENSION]; 2],
        }
    }

    /// Converting token-major capture data to head streams retains signed zero,
    /// sequence/head identity and a hand-computable full-context GQA mean.
    #[test]
    fn test_stream_layout_bits_and_full_attention() -> Result<()> {
        let fixture = Fixture::from_sample(&sample(), 3)?;
        assert_eq!(fixture.keys[0][0], 0x8000);
        let reference = fixture.calculate_reference();
        for sequence in 0..2 {
            for head in 0..QUERY_HEADS {
                let expected = (sequence * 16 + head / (QUERY_HEADS / KV_HEADS) * 4 + 1) as f64;
                assert!(
                    reference[(sequence * QUERY_HEADS + head) * DIMENSION..][..DIMENSION]
                        .iter()
                        .all(|v| (*v - expected).abs() < 1e-12)
                );
            }
        }
        fixture.encode()?.verify(&fixture)
    }

    /// An actual final token rebuilds exactly the captured prefix, while bad
    /// token lengths and non-FP16 input fail without mutating any stream.
    #[test]
    fn test_append_rebuilds_original_words() -> Result<()> {
        let fixture = Fixture::from_sample(&sample(), 3)?;
        for is_key in [false, true] {
            let mut prefix = fixture.get_prefix(2, is_key)?;
            let before = prefix.clone();
            assert!(Fixture::append_tokens(&mut prefix, &[0.0]).is_err());
            assert_eq!(prefix, before);
            Fixture::append_tokens(&mut prefix, &fixture.get_last_tokens(is_key))?;
            assert_eq!(
                prefix,
                if is_key {
                    fixture.keys.clone()
                } else {
                    fixture.values.clone()
                }
            );
        }
        let mut invalid = sample();
        invalid.keys[0][1] = 0.1;
        assert!(Fixture::from_sample(&invalid, 3).is_err());
        Ok(())
    }
}
