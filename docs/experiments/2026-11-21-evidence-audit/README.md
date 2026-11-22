# Independent replay of November evidence

The first branch commits recorded graph and verifier dispositions and
preflight metadata. Their plain-text CUDA test logs remained in the local
worktrees because the repository ignores files ending in .log. This archive
retains those original outputs, the oracle transcript for the initial
Q8 comparison, and the supporting CPU checks. Each compressed payload
round-tripped to the original bytes. [Archive paths, exact report/source
commits, uncompressed sizes and SHA-256 hashes](files.tsv).

Run `bash docs/experiments/2026-11-21-evidence-audit/check.sh` in a checkout
that has the research Git objects. The [replay](check.sh) verifies all 20
payloads and reads the actual hardware observations from both cuBLAS builds.

- The graph grid contains 84 boundary and 144 short comparisons per build.
  Each recorded zero unequal elements. Four repaired session paths each
  matched all logits and FP16 cache comparisons across 32 positions.
  Before the repair, all four paths reached the failing assertion in
  separate processes. Ordinary replay had 83 differing comparisons.
  Cancellation and two launch-rejection paths had 20 each.
- Five speculative verification tests per build ran on the M40. Each of
  four positive near-tie margins lay between 8e-9 and 1.2e-8 and selected
  token 1. The actual test-source diff from the dirty measured checkout
  matches `git diff a21f192 fc487e8 -- tests/exact_verification.rs`
  byte for byte. This establishes provenance for the measured test code.
- The grouped negative graph test stopped at a poisoned test mutex after
  its first expected failure. Its output is preserved separately from
  the four individual controls. All of these observations are correctness
  evidence. Timing qualification uses separate full-request trials.

The [build compatibility audit](../2026-11-21-build-compatibility/README.md)
records a different validation issue involving a shared Cargo target.
The hardware transcripts here identify independent per-worktree release
builds and their actual device observations.

## Branch and decision provenance

The [Git replay](provenance.py) resolves every October merge base, hashes
the original report bytes, reconstructs the full modified-path inventory
against Phase 4 and main, and checks the chronological non-documentation
commits. On the November 22 matrix at 47cf0f5 it verified 13 topics,
793 changed-path records and 48 implementation commits. Its 27 decision
rows referred to 26 distinct source reports with 121 local Markdown
links. All source commits were reachable from their named local branches.
Run from the project root with
`python3 docs/experiments/2026-11-21-evidence-audit/provenance.py`.
The same replay resolves every short Git ID in the rebuild plan and
matches each ledger row's full ID, subject and original author against
the Git object. Review an individual commit's original message and
trailers with `git show -s --format=fuller FULL_SHA`.
The counts change as new evidence is indexed. The replay checks object
identity and local link targets. Each research report supplies the
hardware, workload and qualification boundaries for its conclusions.

## Archive replays by decision

The replay at `8e851d671fbc1b8927159f972adf85d8b0cf055a` checked all 13
October topics, 793 changed paths, 48 implementation commits, 50
decision rows and 45 distinct source reports. It resolved 210 local
Markdown links and checked 101 immutable rebuild sources. That
consolidation snapshot and each cited topic tip were reachable through
the 24 research branches published to `origin`.

| Study | Check the evidence for the stated scope |
|---|---|
| A2 serving controls | Extract each cohort and rerun its request-ledger, token clock and process-peak replay. Medium Qwen has five reproducible trials per arm and a separate unequal-output verdict. The dense four-client report rechecks all output IDs and actual width-four dispatch. [Serving reproductions](../2026-11-21-serving-baseline/README.md) · [Dense four-client replay](https://github.com/GuyPaddock/m40-llm/blob/3d10ccaa44be00fa7fd3d66ea7b8b836836ad45e/docs/experiments/2026-11-21-serving-concurrency/dense-four/README.md). |
| B1 graph | Inspect actual replay count, full logits, exact FP16 cache observations and teardown per CUDA build. The unsupported F16 capture and ordinary fallback are retained beside the successful cuBLAS replay. Read HTTP throughput and capture measurements in their separate run records. [Main parity archive](https://github.com/GuyPaddock/m40-llm/blob/beb1fafd746bab4d8d88f9f9f692b5da6d4e1337/docs/experiments/2026-11-21-main-graph-parity/result/README.md) · [TinyLlama graph timing](https://github.com/GuyPaddock/m40-llm/blob/a43a9a5199d81eae82cdd5798c2a049775c52026/docs/experiments/2026-11-21-graph-http/results/tiny.md). |
| B2 mixed KV | Recompute all aligned comparisons from the raw FP32 vectors, and score the threshold boundary separate from the broad grid. The staged/direct consumer comparison verifies its dispatch and matched work with both cache modes holding the same representation. [Quality replay](https://github.com/GuyPaddock/m40-llm/blob/d6cabf1cd0e0bd97df030eb47fbf96f613bb0722/docs/experiments/2026-11-21-kv-quality-result/full-replay/README.md) · [Top-64 boundary](https://github.com/GuyPaddock/m40-llm/blob/efaf5e8d84b74c84dad9ff0de3693fb8bd031cec/docs/experiments/2026-11-21-kv-history/result/README.md) · [Matched consumer replay](../2026-11-21-serving-baseline/tiny-kv-consumer/result/README.md). |
| B3 prepared Q8 | Replay the same-executable post-upgrade comparison, then check cold-start and live-process costs from the separate isolated-server records. Check packet/argmax timing against its already prepared control. [Selective preparation](../2026-11-21-native-split/README.md) · [Isolated server peaks](../2026-11-22-q8-ownership/server-peaks/README.md) · [Portfolio replay](https://github.com/GuyPaddock/m40-llm/blob/7ff9d33f48c831c4218b5027d6a1cbf01d6c0ad3/docs/experiments/2026-11-21-q8-economics/portfolio/README.md). |
| C1, C2 and D1 verification | Recompute paired target and complete-request gates. Inspect full logit and KV witnesses, the forced-rejection rows, and natural copy rejection in its separate source-bound run. [Complete target and forced rejection](https://github.com/GuyPaddock/m40-llm/blob/837e0708edda8ca4d74aabcf6436e18dfffad6bc/docs/experiments/2026-11-21-small-n/target-result/README.md) · [Natural rejection and live copy](https://github.com/GuyPaddock/m40-llm/blob/e1fcebb8f7bf274465f2d4493e0423ba0268c1ff/docs/experiments/2026-11-22-copy-verifier/result/README.md). |
| C4 sparse Q4 | Match each dense generation against the independent capture, recompute repetition, and apply the frozen 3% likelihood rule. The operator evidence admits a quality test and the original quality report records the zigzag stop. The 48-case packed quality recheck has a separate runner. [Initial quality replay](https://github.com/GuyPaddock/m40-llm/blob/d64a2edc93ee60c5f148874b51b6e2eac3174ce5/docs/experiments/2026-11-22-sparse-q4-live/quality-result/README.md) · [Expanded runner](https://github.com/GuyPaddock/m40-llm/blob/aef2d6278a44d63069636d1eca3749583d8c2868/docs/experiments/2026-11-22-sparse-q4-live/extended.md). |
| D3 prefill fairness | Replay the mixed-request admission and state guards. The first candidate archive preserves the interrupted diagnostic outside the complete performance gate. [Measured stall](https://github.com/GuyPaddock/m40-llm/blob/a37d0bc8b9d9537f626b696868857f88dd80ada2/docs/experiments/2026-11-21-prefill-fairness/result/README.md) · [Exact state and cancellation](https://github.com/GuyPaddock/m40-llm/blob/c0cdbcb373b80248a551d87e6a942330cd5c7161/docs/experiments/2026-11-21-prefill-fairness/chunking-guards/README.md) · [First instrumented pair and runner failure](https://github.com/GuyPaddock/m40-llm/blob/f2333456ec479e4bd5fb68ed8a4bf7090bf341d7/docs/experiments/2026-11-21-prefill-fairness/chunking-attempt-01/README.md). |
| F5 build compatibility | Compare the exact source paths, target directories, test counts, flags and exit codes. Track each M40 guard in its topic result. [Source-specific build archive index](../2026-11-21-build-compatibility/README.md). |
