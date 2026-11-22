# Main-based CUDA Graph dispatch qualification

**PASS for the narrow attention-dispatch repair. PASS for actual
TinyLlama F16 replay with cuBLAS. FAIL for capture eligibility on
the measured F16 build with cuBLAS disabled.** The pinned main
control reproduced bit differences before and across the
8,192/8,193 boundary. The repaired attention path passed with
both CUDA builds on the same Tesla M40.

| Check | Main control, cuBLAS on | Repair, cuBLAS on | Repair, cuBLAS off |
|---|---:|---:|---:|
| Short attention, 96 comparisons | 15,564 unequal FP32 elements | Exact | Exact |
| Capacity 8,193 attention, 56 comparisons | 7,450 unequal FP32 elements | Exact | Exact |
| TinyLlama F16 actual graph replays / 8 | 8 | 8 | 0 |
| TinyLlama unequal FP32 logits | 223,850 | 0 | Fallback |
| TinyLlama unequal valid FP16 KV words | 42,195 | 0 | Fallback |

Each repaired attention grid checked both head widths and both slot
positions, with intact guards, unchanged input and successful
cleanup. The boundary fixture changes the live row count within a
captured graph. The full-model cuBLAS run compared 256,000 FP32
logit values and 405,504 valid FP16 KV words across eight prefixes.
Both sessions released their tracked allocations. The no-cuBLAS
full-model capture failed in the asynchronous F16 Q projection
with FFI status −3. Ordinary fallback matched ordinary state and
completed cleanup. The guard counted zero actual replays and
rejected that configuration. The runtime repair preserves this
fallback behavior.

The executable identities are pinned at
`5877729c9bfa7fdaa4d950729ec1f310936a0dbf` for both repaired
builds and `afaa86e854b67ae459836c07e1bb8929e3c5bd91` for the
unfixed control. The same fixture seeded the required boundary
positions in order in all builds. The [first run](../first-run/README.md)
retains its earlier setup failures.

Carry the narrow correctness change at
`63ea66a2b5f83bf07ac1945b64554796343d6cfd` with these exact
fixtures. Its two production file patches apply directly to the
pinned main without additional Phase 4 projection code. Keep
CUDA Graphs opt-in. A no-cuBLAS F16 capture path would require its
own projection support and parity evidence. The Phase 4 graphs
missed the measured speed gate on Qwen and TinyLlama.

The [archive](measurements.tar.gz), [checksum](archive.sha256) and
[exit ledger](exits.tsv) retain both graphs, ordinary controls,
all build/runtime logs and model/device identity. Under the shared
M40 lock run `bash ../build-retry.sh NEW_BUILD_DIR` from the
research worktree with the script path resolved to this study.
Then invoke [the test runner](../run.sh) with that build directory,
a new raw output directory, the pinned TinyLlama F16 GGUF, and
the build directory's `binaries.sha256`. The runner exits after
the complete matrix, so read the per-fixture exits for outcomes.
