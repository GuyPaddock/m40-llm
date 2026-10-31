# Real-Qwen CPU projection validation

All 36 retained projection cases pass the fixed FP64 numerical gate for both
the scalar and AVX2 implementations at `bf8dc96`. This admits the next cost
measurement; it does not establish CPU decoding speed or model quality.

| Check | Result |
| --- | --- |
| Grid | Layers 0/17/35, gate/up/down, batches 1/2/4/8 |
| Outputs checked per implementation | 1,082,880 |
| Scalar maximum absolute error | 0.000996087425 |
| AVX2 maximum absolute error | 0.000148677535 |
| Scalar maximum error / allowed tolerance | 0.2499331672 |
| AVX2 maximum error / allowed tolerance | 0.0374773183 |
| Fixed tolerance | `1e-3 + 2e-5 * abs(FP64 reference)` |
| Placement | One thread, CPU 15, NUMA memory binding 0 |
| Timed projection pairs | 0 |
| Generated model tokens | 0 |

Maximum absolute errors and tolerance ratios can occur at different output
elements. The implementation checks every element; a ratio at most one passes.
The AVX2 reduction order differs from the scalar loop, so bit identity is not
required by this diagnostic. No complete-layer or full-model CPU path ran.

The run used the [probe's pinned model and capture identities](2026-10-31-dualpace-projection-probe.md).
Tensor names, dimensions, offsets and source hashes all matched. Every batch
used the corresponding prefix of eight distinct captured activation rows.
The clean release executable SHA-256 is
`07f47ffd5009beeb58ed67c91e2a2ade558902dc07334ff3982f10093f88459d`.

The process exited zero. Its 48 JSONL records contain one manifest, one eviction
allocation record, nine matrix setups, 36 numerical rows and one completion.
Setup/validation elapsed times are retained for accounting; they are not paired
projection measurements. The lossless capture continued independently on the
GPU, and this CPU-only run made no CUDA calls.

Reproduce using the probe's documented CPU 15/node 0 command with
`--validate-only` and a new output. The raw result and execution/build logs are
compressed beside this report. Recompute the [summary](2026-10-31-dualpace-cpu-validation.json)
with the [audit program](2026-10-31-dualpace-cpu-validation.jq):

```bash
gzip -dc docs/experiments/2026-10-31-dualpace-cpu-validation.jsonl.gz | \
  jq -s -f docs/experiments/2026-10-31-dualpace-cpu-validation.jq
sha256sum --check docs/experiments/2026-10-31-dualpace-cpu-validation.sha256
```

The audit verifies source/placement, the complete grid, numerical gates and
absence of timing rows. Its hashes preserve provenance; they do not replace
the numerical checks in the executable.

Next: serial CPU timing on both planned NUMA placements, then H2D/D2H and
synchronization pricing once the M40 capture finishes. The cooperative study
still needs a sourced simulator input and a priced full-CPU-decode prerequisite
report. No scheduler or runtime default changed.
