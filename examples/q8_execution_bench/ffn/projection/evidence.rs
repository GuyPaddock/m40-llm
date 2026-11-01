use super::super::fixture::FfnInputs;
use anyhow::{ensure, Context, Result};
use m40_llm::evaluation::identity::sha256;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

const QKV_REPORT_SHA256: &str = "9c1468a93abc8b4828e04d5880f1d1fbb70c36fe4790c73c27087a0098217b73";

pub(super) struct QkvEvidence {
    pub binding: Value,
    gains: BTreeMap<(usize, usize), f64>,
}

impl QkvEvidence {
    pub fn load(path: &Path, inputs: &FfnInputs) -> Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)?
            .take(2 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        let mut evidence = Self::parse(
            &bytes,
            &inputs.artifact.plan.model_sha256,
            &inputs.artifact_sha256,
            &inputs.report_sha256,
        )?;
        evidence.binding["path"] = json!(path);
        Ok(evidence)
    }

    fn parse(
        bytes: &[u8],
        model_sha256: &str,
        fixture_sha256: &str,
        capture_sha256: &str,
    ) -> Result<Self> {
        ensure!(
            bytes.len() <= 2 * 1024 * 1024,
            "QKV peer report exceeds bound"
        );
        ensure!(
            sha256(bytes)? == QKV_REPORT_SHA256,
            "QKV peer differs from prespecified evidence"
        );
        let records = std::str::from_utf8(bytes)?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<Vec<Value>, _>>()?;
        let manifest = records.first().context("empty QKV report")?;
        ensure!(
            manifest["kind"] == "qkv_warp_manifest"
                && manifest["model_sha256"] == model_sha256
                && manifest["fixture_sha256"] == fixture_sha256
                && manifest["capture_report_sha256"] == capture_sha256,
            "QKV peer does not match frozen model/input provenance"
        );
        let mut gains = BTreeMap::new();
        for record in records.iter().filter(|r| r["kind"] == "qkv_warp_screen") {
            ensure!(
                record["result"]["passes_micro_screen"] == true,
                "QKV peer stage did not qualify"
            );
            let key = (
                record["layer_id"].as_u64().context("missing peer layer")? as usize,
                record["rows"].as_u64().context("missing peer rows")? as usize,
            );
            ensure!(
                gains
                    .insert(
                        key,
                        record["result"]["throughput_gain"]
                            .as_f64()
                            .context("missing peer gain")?
                    )
                    .is_none(),
                "duplicate peer case"
            );
        }
        ensure!(
            gains.len() == 6
                && records
                    .last()
                    .is_some_and(|r| r["kind"] == "qkv_warp_complete" && r["is_complete"] == true),
            "incomplete QKV peer"
        );
        Ok(Self {
            binding: json!({"sha256":QKV_REPORT_SHA256,"revision":manifest["revision"],"executable_sha256":manifest["executable_sha256"]}),
            gains,
        })
    }

    pub fn combined_gain(&self, layer: usize, rows: usize, output_gain: f64) -> Result<f64> {
        let qkv_gain = self
            .gains
            .get(&(layer, rows))
            .context("missing matching QKV peer case")?;
        Ok(
            1.0 / (1.0 - 0.1172 - 0.0825
                + 0.1172 / (1.0 + qkv_gain)
                + 0.0825 / (1.0 + output_gain))
                - 1.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the prespecified completed peer can qualify the combination;
    /// changed bytes or a different model/input provenance are rejected.
    #[test]
    fn test_peer_requires_frozen_report_and_matching_inputs() -> Result<()> {
        let bytes =
            include_bytes!("../../../../docs/experiments/2026-10-25-maxwell-qkv-warp-screen.jsonl");
        let model = "4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d";
        let fixture = "eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687";
        let capture = "d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801";
        let peer = QkvEvidence::parse(bytes, model, fixture, capture)?;
        for layer in [0, 17, 35] {
            for rows in [1, 4] {
                assert!(peer.combined_gain(layer, rows, 0.1)? > 0.03);
            }
        }
        for (m, f, c) in [
            ("other", fixture, capture),
            (model, "other", capture),
            (model, fixture, "other"),
        ] {
            assert!(QkvEvidence::parse(bytes, m, f, c).is_err());
        }
        let mut changed = bytes.to_vec();
        changed.push(b'\n');
        assert!(QkvEvidence::parse(&changed, model, fixture, capture).is_err());
        assert!(QkvEvidence::parse(&[], model, fixture, capture).is_err());
        assert!(
            QkvEvidence::parse(&vec![0; 2 * 1024 * 1024 + 1], model, fixture, capture).is_err()
        );
        Ok(())
    }

    /// The combined ceiling uses separate measured stage gains and cannot
    /// treat projection-only gains as whole-token improvements.
    #[test]
    fn test_combined_gain_weights_independent_stages() -> Result<()> {
        let evidence = QkvEvidence {
            binding: Value::Null,
            gains: BTreeMap::from([((17, 1), 0.275)]),
        };
        let gain = evidence.combined_gain(17, 1, 0.1)?;
        assert!(
            (gain - (1.0 / (1.0 - 0.1172 - 0.0825 + 0.1172 / 1.275 + 0.0825 / 1.1) - 1.0)).abs()
                < 1e-12
        );
        assert!(gain > 0.03 && gain < 0.04);
        assert!(evidence.combined_gain(35, 1, 0.1).is_err());
        Ok(())
    }
}
