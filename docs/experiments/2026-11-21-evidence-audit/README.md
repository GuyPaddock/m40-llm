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
The counts change as new evidence is indexed. The replay checks object
identity and local link targets. Each research report supplies the
hardware, workload and qualification boundaries for its conclusions.
