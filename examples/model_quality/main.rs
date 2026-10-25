#[cfg(all(feature = "cuda", nvcc))]
mod activations;
#[cfg(all(feature = "cuda", nvcc))]
mod compact;
#[cfg(all(test, feature = "cuda", nvcc))]
#[path = "../../tests/common/compact_gguf.rs"]
mod compact_fixture;
#[cfg(all(test, feature = "cuda", nvcc))]
#[path = "../../tests/common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[cfg(all(test, feature = "cuda", nvcc))]
#[path = "../../tests/common/env_restore.rs"]
mod env_restore;
#[cfg(all(feature = "cuda", nvcc))]
mod ffn_capture;
#[cfg(all(test, feature = "cuda", nvcc))]
#[path = "../../tests/common/tiny_gguf.rs"]
mod fixture;
#[cfg(all(feature = "cuda", nvcc))]
mod layers;
#[cfg(all(feature = "cuda", nvcc))]
mod prefill_drift;
#[cfg(all(feature = "cuda", nvcc))]
mod prepared_drift;
#[cfg(all(test, feature = "cuda", nvcc))]
#[path = "../../tests/common/q8_model.rs"]
#[allow(dead_code)]
mod q8_fixture;
#[path = "../common/q8_runtime.rs"]
mod q8_runtime;
#[cfg(all(feature = "cuda", nvcc))]
#[path = "../common/qwen_workload.rs"]
mod qwen_workload;
#[cfg(any(all(feature = "cuda", nvcc), test))]
mod records;
#[cfg(all(feature = "cuda", nvcc))]
mod research;
#[cfg(all(feature = "cuda", nvcc))]
mod run;
#[cfg(all(feature = "cuda", nvcc))]
mod runtime_quality;
#[cfg(all(feature = "cuda", nvcc))]
mod step_timing;
#[cfg(all(feature = "cuda", nvcc))]
mod tasks;
#[cfg(all(test, feature = "cuda", nvcc))]
mod test_files;

use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use m40_llm::evaluation::corpus::Split;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SplitChoice {
    Calibration,
    HeldOut,
    All,
}

impl SplitChoice {
    fn selected(self) -> Vec<Split> {
        match self {
            Self::Calibration => vec![Split::Calibration],
            Self::HeldOut => vec![Split::HeldOut],
            Self::All => vec![Split::Calibration, Split::HeldOut],
        }
    }
}

#[derive(Parser)]
#[command(
    about = "Explicit model-variant quality, loss, and resident generation timing; not HTTP throughput"
)]
struct Args {
    #[arg(long)]
    reference_model: PathBuf,
    #[arg(long)]
    model: PathBuf,
    /// Descriptive label; actual model hashes determine source identity.
    #[arg(long)]
    variant_id: String,
    #[arg(long, default_value = "data/research_quality/v1.json")]
    corpus: PathBuf,
    #[arg(long, value_enum, default_value = "all")]
    split: SplitChoice,
    #[arg(long, default_value_t = 1024)]
    context_tokens: u32,
    /// M08 runtime comparison; both models keep prepared down/vocabulary.
    #[arg(long, value_enum, conflicts_with_all=["capture_ffn_inputs","q8_prefix_drift","compact_export_report","skip_layers","rank_layers","layer_ranking","remove_layer_count","capture_activations","activation_calibration","activation_target","step_timing_trials"])]
    q8_runtime_candidate: Option<q8_runtime::FfnChoice>,
    #[arg(long, value_enum, requires = "q8_runtime_candidate")]
    q8_runtime_prefill: Option<q8_runtime::PrefillChoice>,
    /// Untimed comparison of sequential admission, explicit stream drains and
    /// packed prefixes on identical prompt IDs, hidden states and physical KV.
    #[arg(long="q8-prefill-drift", conflicts_with_all=["q8_runtime_candidate","capture_ffn_inputs","q8_prefix_drift","compact_export_report","skip_layers","rank_layers","layer_ranking","remove_layer_count","capture_activations","activation_calibration","activation_target","step_timing_trials"])]
    should_diagnose_prefill: bool,
    /// Untimed M02 input capture using the qualified prepared down/vocabulary
    /// control. Requires calibration split and creates a new paired artifact.
    #[arg(long, conflicts_with_all=["q8_prefix_drift","compact_export_report","skip_layers","rank_layers","layer_ranking","remove_layer_count","capture_activations","activation_calibration","activation_target","step_timing_trials"])]
    capture_ffn_inputs: Option<PathBuf>,
    /// Diagnose the recorded native/prepared down+vocabulary branch on identical
    /// prefixes. No fitting, free generation or performance promotion occurs.
    #[arg(long, conflicts_with_all=["compact_export_report","skip_layers","rank_layers","layer_ranking","remove_layer_count","capture_activations","activation_calibration","activation_target"])]
    q8_prefix_drift: Option<PathBuf>,
    /// Compare the masked source against an unmasked compact candidate using
    /// this completed export record. Other candidate transforms are forbidden.
    #[arg(long, conflicts_with_all=["skip_layers","rank_layers","layer_ranking","remove_layer_count","capture_activations","activation_calibration","activation_target"])]
    compact_export_report: Option<PathBuf>,
    /// Candidate-only diagnostic, zero-based interior IDs; reference stays intact.
    #[arg(long, value_delimiter = ',', conflicts_with_all=["rank_layers","layer_ranking"])]
    skip_layers: Vec<usize>,
    /// Calibration-only BI and iterative loss study; create a new frozen artifact.
    #[arg(long, conflicts_with_all=["layer_ranking","capture_activations","activation_calibration","skip_layers"])]
    rank_layers: Option<PathBuf>,
    #[arg(long, default_value_t=3, value_parser=clap::value_parser!(u32).range(1..=3))]
    layer_ranking_removals: u32,
    #[arg(long, default_value_t = 64)]
    layer_ranking_budget_mb: usize,
    /// Apply a mask from this already frozen calibration-only artifact.
    #[arg(long, requires="remove_layer_count", conflicts_with_all=["rank_layers","capture_activations","skip_layers"])]
    layer_ranking: Option<PathBuf>,
    #[arg(long, requires="layer_ranking", value_parser=clap::value_parser!(u32).range(1..=3))]
    remove_layer_count: Option<u32>,
    /// Capture calibration-only generated SwiGLU inputs and freeze this new artifact.
    #[arg(long, conflicts_with_all=["activation_calibration","activation_target","skip_layers"])]
    capture_activations: Option<PathBuf>,
    #[arg(long, default_value_t = 32)]
    activation_capture_rows: usize,
    #[arg(long, default_value_t = 512)]
    activation_observation_budget_mb: usize,
    /// Apply a checked frozen artifact only to the candidate. A simultaneous
    /// frozen layer ranking composes both changes on original source layer IDs.
    #[arg(long, requires = "activation_target", conflicts_with = "skip_layers")]
    activation_calibration: Option<PathBuf>,
    #[arg(long, requires = "activation_calibration")]
    activation_target: Option<f64>,
    #[arg(long, default_value_t = 1024)]
    activation_budget_mb: usize,
    /// Optional paired forced-prefix greedy step timings, not generated HTTP TPS.
    #[arg(long, default_value_t = 0)]
    step_timing_trials: usize,
    #[arg(long, default_value_t = 8)]
    step_timing_warm_prefix_tokens: usize,
    /// Must not exist. Flush each completed row; never overwrite prior evidence.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!cfg!(debug_assertions), "use --release for evaluation");
    ensure!(
        !args.variant_id.trim().is_empty() && args.context_tokens > 1,
        "provide a nonempty variant ID and at least two context slots"
    );
    ensure!(
        args.step_timing_trials <= 32 && args.step_timing_warm_prefix_tokens > 0,
        "step timing accepts 0..32 trials and a nonempty warm prefix"
    );
    #[cfg(all(feature = "cuda", nvcc))]
    {
        run::evaluate(args)
    }
    #[cfg(not(all(feature = "cuda", nvcc)))]
    {
        // Keep the CLI/error available in non-CUDA builds, not a successful skip.
        let _ = args.split.selected();
        anyhow::bail!("evaluation requires --features cuda and an NVCC build")
    }
}

#[cfg(test)]
mod argument_tests {
    use super::*;

    fn parse(extra: &[&str]) -> Result<Args, clap::Error> {
        let mut values = vec![
            "model_quality",
            "--reference-model",
            "reference.gguf",
            "--model",
            "candidate.gguf",
            "--variant-id",
            "test",
            "--output",
            "new.jsonl",
        ];
        values.extend_from_slice(extra);
        Args::try_parse_from(values)
    }

    /// Prefill attribution cannot silently change the model or reuse a different
    /// quality/capture mode. It is a separate diagnostic, not an optimization.
    #[test]
    fn test_prefill_drift_arguments_exclude_other_experiments() {
        assert!(
            parse(&["--q8-prefill-drift"])
                .unwrap()
                .should_diagnose_prefill
        );
        for extra in [
            vec!["--q8-runtime-candidate", "native"],
            vec!["--skip-layers", "1"],
            vec!["--capture-ffn-inputs", "new.json"],
            vec!["--q8-prefix-drift", "trace.jsonl"],
            vec!["--step-timing-trials", "1"],
        ] {
            let values = [vec!["--q8-prefill-drift"], extra].concat();
            assert!(parse(&values).is_err());
        }
    }

    /// Runtime comparisons require explicit candidates and cannot mutate the
    /// frozen model with pruning, capture or unrelated timing experiments.
    #[test]
    fn test_runtime_candidate_arguments_are_explicit_and_exclusive() {
        for choice in [
            "native",
            "packed-f32-256",
            "packed-f32-128",
            "q8-1-128",
            "q8-1-256",
        ] {
            assert!(parse(&["--q8-runtime-candidate", choice]).is_ok());
        }
        assert!(parse(&["--q8-runtime-prefill", "dequant-sgemm"]).is_err());
        for extra in [
            vec!["--skip-layers", "1"],
            vec!["--capture-ffn-inputs", "new.json"],
            vec!["--q8-prefix-drift", "trace.jsonl"],
            vec!["--step-timing-trials", "1"],
            vec![
                "--activation-calibration",
                "frozen.json",
                "--activation-target",
                "0.5",
            ],
        ] {
            let values = [vec!["--q8-runtime-candidate", "native"], extra].concat();
            assert!(parse(&values).is_err());
        }
    }

    /// Frozen masks require an explicit bounded count and cannot be combined
    /// with ad-hoc masks or simultaneous fitting. Frozen activation thresholds
    /// may compose with a frozen ranking, but not with capture or compact export.
    #[test]
    fn test_layer_ranking_arguments_are_explicit_and_exclusive() {
        for count in ["1", "2", "3"] {
            assert_eq!(
                parse(&[
                    "--layer-ranking",
                    "frozen.json",
                    "--remove-layer-count",
                    count
                ])
                .unwrap()
                .remove_layer_count,
                Some(count.parse::<u32>().unwrap())
            );
        }
        for extra in [
            vec!["--layer-ranking", "frozen.json"],
            vec!["--remove-layer-count", "1"],
            vec![
                "--layer-ranking",
                "frozen.json",
                "--remove-layer-count",
                "0",
            ],
            vec![
                "--layer-ranking",
                "frozen.json",
                "--remove-layer-count",
                "4",
            ],
            vec![
                "--layer-ranking",
                "frozen.json",
                "--remove-layer-count",
                "1",
                "--skip-layers",
                "2",
            ],
            vec![
                "--rank-layers",
                "new.json",
                "--layer-ranking",
                "frozen.json",
                "--remove-layer-count",
                "1",
            ],
            vec![
                "--rank-layers",
                "new.json",
                "--capture-activations",
                "other.json",
            ],
        ] {
            assert!(parse(&extra).is_err());
        }
    }

    /// Combinations require both independently frozen artifacts and explicit
    /// parameters; no fitting, ad-hoc mask or compact model can be substituted.
    #[test]
    fn test_frozen_layer_activation_combination_arguments() {
        let combined = [
            "--layer-ranking",
            "frozen-ranking.json",
            "--remove-layer-count",
            "1",
            "--activation-calibration",
            "frozen-activations.json",
            "--activation-target",
            "0.5",
        ];
        let args = parse(&combined).unwrap();
        assert_eq!(args.remove_layer_count, Some(1));
        assert_eq!(args.activation_target, Some(0.5));
        for extra in [
            ["--skip-layers", "1"],
            ["--rank-layers", "new.json"],
            ["--capture-activations", "new.json"],
            ["--compact-export-report", "export.json"],
        ] {
            let values = [combined.as_slice(), extra.as_slice()].concat();
            assert!(parse(&values).is_err());
        }
        assert!(parse(&combined[..6]).is_err());
    }

    /// Prefix diagnostics cannot silently add model changes to the recorded
    /// native/prepared comparison or start a simultaneous fitting operation.
    #[test]
    fn test_prefix_drift_arguments_exclude_other_experiments() {
        assert!(parse(&["--q8-prefix-drift", "trace.jsonl"])
            .unwrap()
            .q8_prefix_drift
            .is_some());
        for extra in [
            vec!["--skip-layers", "1"],
            vec!["--rank-layers", "new.json"],
            vec!["--compact-export-report", "export.json"],
            vec!["--capture-activations", "new.json"],
            vec!["--layer-ranking", "rank.json", "--remove-layer-count", "1"],
            vec![
                "--activation-calibration",
                "act.json",
                "--activation-target",
                "0.5",
            ],
        ] {
            let values = [vec!["--q8-prefix-drift", "trace.jsonl"], extra].concat();
            assert!(parse(&values).is_err());
        }
    }

    /// Paired FFN capture is read-only and must not silently compose a model
    /// transform, another capture, a prefix probe or timed evaluation.
    #[test]
    fn test_ffn_capture_arguments_exclude_other_experiments() {
        let capture = ["--capture-ffn-inputs", "new-ffn.json"];
        assert_eq!(
            parse(&capture).unwrap().capture_ffn_inputs,
            Some(PathBuf::from("new-ffn.json"))
        );
        for extra in [
            vec!["--skip-layers", "1"],
            vec!["--rank-layers", "new.json"],
            vec!["--capture-activations", "new.json"],
            vec!["--q8-prefix-drift", "old.jsonl"],
            vec!["--compact-export-report", "export.json"],
            vec!["--layer-ranking", "rank.json", "--remove-layer-count", "1"],
            vec![
                "--activation-calibration",
                "act.json",
                "--activation-target",
                "0.5",
            ],
            vec!["--step-timing-trials", "5"],
        ] {
            assert!(parse(&[capture.as_slice(), extra.as_slice()].concat()).is_err());
        }
    }

    /// Compact equivalence has its own checked source mask. Ad-hoc candidate
    /// masks or simultaneous fitting/transforms cannot silently change the pair.
    #[test]
    fn test_compact_comparison_arguments_reject_other_transforms() {
        assert!(parse(&["--compact-export-report", "export.json"])
            .unwrap()
            .compact_export_report
            .is_some());
        for extra in [
            vec!["--skip-layers", "1"],
            vec!["--rank-layers", "new.json"],
            vec![
                "--layer-ranking",
                "frozen.json",
                "--remove-layer-count",
                "1",
            ],
            vec!["--capture-activations", "new.json"],
            vec![
                "--activation-calibration",
                "frozen.json",
                "--activation-target",
                "0.5",
            ],
        ] {
            let mut values = vec!["--compact-export-report", "export.json"];
            values.extend(extra);
            assert!(parse(&values).is_err());
        }
    }
}
