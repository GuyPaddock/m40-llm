# Native versus split down/vocabulary, November 21

**PASS for the historical short dense HTTP workload.** Five alternating warm pairs gave 31.7152 native and 35.4355 prepared median HTTP tokens/s, +11.7304%, reported paired bootstrap interval +10.4420% to +12.0913%. Both arms returned 512 IDs on the 65-ID prompt and those IDs matched. All ten timed outputs matched within pairs. Three dense-valid quality smoke cases were preserved. The added prepared representation occupied 1,192,935,424 bytes. Cache allocation was 37,748,736 bytes per request in both arms. The process contained two model arms, so its resident snapshot cannot establish a single-model peak VRAM delta.

The release binary was built in the independent consolidation target directory while its tree was at d9af4f4 (documentation-only versus Phase 4), and executed from clean Phase 4 48c6333. Compare prod/test/build sources with git diff 48c6333 d9af4f4 -- src cuda examples build.rs Cargo.toml Cargo.lock. preflight.txt records its hash and sm_52 ELF. Official Ollama 0.40.11 supplied the tokenizer/backend oracle. No Ollama requests overlapped paired timing. Other host CPU-only research work was in progress during part of the cohort. The observed MAD/median was 0.14% control and 0.32% prepared. Stronger portfolio controls, longer contexts, batch economics and peak single-process residency remain separate comparisons.

Replay with the controlled benchmark instructions, the flags in preflight.txt and the checksummed model and oracle. Start the isolated official server as described in docs/controlled_benchmark.md. Invoke compare_ollama --model MODEL --ollama-pid PID --ollama-log LOG --compare-q8-execution split --q8-families down,vocabulary --output NEW.jsonl. Recompute the recorded median and gain directly from client elapsed time with jq -es -f audit.jq report.jsonl. Prefill rate includes request admission/setup through the first sample. Decode rate counts the remaining 511 tokens. The raw report retains dispatch, traces, outputs, memory, run order and GPU boundaries. The confidence interval is copied from the original paired harness.

The operator subsequently reported a Gentoo system upgrade in the background. Its per-trial CPU and I/O load was not recorded. This run establishes the stated paired observation on that host. The interval and low dispersion do not exclude a systematic host effect on both arms. Carry this limitation into any use of the result for integration.

The operator subsequently paused Portage. Repeat the original Qwen cohort
independently with the identical checksummed executable, oracle and flags.
The [repeat runner](repeat.sh) retains the second trace and host scheduler
state. A sleeping or stopped worker still appears in a process census.
Compare throughput, time to first token, complete request time and the
relative gain separately across cohorts. Keep startup/preparation figures
conditional on OS cache state. Command from the consolidation worktree under
the exclusive M40 lock: `bash docs/experiments/2026-11-21-native-split/repeat.sh NEW_OUTPUT_DIRECTORY`.

## Per-arm inference residency

The timing process retained both models. Measure the native and selective
down/vocabulary configurations in two fresh normal server processes,
one model resident at a time. Use the existing isolated-server probe,
the archived flags and the original 65-ID prompt with a 512-token cap.
Require all 36 layers, sequential prefill, both cold and warm traces
matching the original 512 IDs, and 37 split tensors totaling 1,192,935,424
prepared bytes in the candidate. Reject a mismatched dispatch or output.
The native process must have preparation disabled.

Run `bash docs/experiments/2026-11-22-q8-ownership/server-peak.sh NEW_DIRECTORY native-split`
under the same M40 lock. Retain each server PID, 250 ms process samples,
whole-device snapshots, load-through-health and cold/warm request windows,
and host RSS high-water mark. Use these two runs to price the peak memory
increase and its ratio to the retained weight bytes. The paired timing
cohorts supply the speed verdict. No new timing or kernel search follows
from a memory observation.

## Repeat after the Gentoo upgrade pause

**PASS on the frozen short Qwen Q8_0 HTTP workload.** Across five new
alternating pairs, native weights delivered 31.922 HTTP tokens/s and
prepared down/vocabulary weights delivered 35.681. The median gain was
11.776%, with a paired bootstrap interval of 11.579–11.828%. The earlier
cohort gave 11.730%. Both arms generated the same 512 token IDs in the
repeat. Median complete request time fell from 16.039 s to 14.349 s.
The prepared weights still required 1,192,935,424 additional bytes.

Host samples recorded one stopped emerge process, one sleeping emerge
process and nine sleeping make processes at the sampling instants. The
repeat supports this workload-specific gain with the upgrade paused.
It does not quantify the effect of emerge on individual trials.
The build, flags, outputs, host samples and timings are in the
[raw archive](post-pause/measurements.tar.gz). Processing the archived
report with the same jq audit reproduces the [summary](post-pause/audit.json).
