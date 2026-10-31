# Exact Decode Reductions: Stage Gain, Insufficient Runtime Scope

## Decision

Keep the fixed weighted RMSNorm and partitioned argmax probes benchmark-only.
The fresh ten-pair argmax screen is **15.43-16.28x faster** for one request,
including both candidate kernels, but its eligible optimistic E2E scope is
only **1.2533%**, below the prespecified 3% gate. Norm's one-row point gains
are **10.84-22.44%**, but four of six paired intervals cross zero; neither
norm stage passes all three source-layer cases.

The initial five-pair screen and fresh ten-pair run are separate. The latter
addresses excessive initial timing variation, not a candidate/geometry change.
Do not pool them, wire the sampler/norm into generation, or rescue them with
new geometry, cache hints, fusion or register caps. No actual-512 quality or
HTTP qualification is authorized by these results. Both exact goals remain
unmet: **53.06 HTTP tok/s** and **1.30x the fresh official Ollama median**,
currently 56.037 tok/s. Qualified runtime stays 49.146 versus 43.106 HTTP
tok/s; defaults and the preferred compressed-KV hierarchy are unchanged.

## Fresh Ten-Pair Results

Each interval repeats the complete operation ten times, using CUDA events on
the decode stream. Setup, uploads, CPU references, readback, guards and an
explicit pre-drain are outside timing. Event intervals can include host-issued
launch gaps; they are not isolated barrier costs or measured DRAM traffic.

Actual layer 0/17/35 norm weights are FP32, dimension 2048, using the model's
FP32 epsilon near 1e-6. Frozen normalized FFN vectors are **norm-input
surrogates**, not captured pre-norm hidden states.

| One-row norm | Control us | Candidate us | Throughput gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: | ---: |
| Attention, layer 0 | 9.1968 | 7.9888 | 15.12% | -0.76% to 43.68% |
| FFN, layer 0 | 8.4496 | 7.5488 | 11.93% | -3.78% to 28.08% |
| Attention, layer 17 | 8.4928 | 7.6448 | 11.09% | 2.40% to 23.77% |
| FFN, layer 17 | 8.5760 | 7.7376 | 10.84% | -0.31% to 15.13% |
| Attention, layer 35 | 9.2112 | 7.5232 | 22.44% | -3.37% to 48.95% |
| FFN, layer 35 | 8.9264 | 7.7920 | 14.56% | 3.33% to 35.29% |

Norm rows4 is one four-block launch on distinct inputs, not four one-row
launches. Fresh four-row gains are 9.81-13.52%, all with positive intervals.
Those results do not replace the one-request eligibility requirement.

Argmax uses explicitly synthetic finite logits at the original 151936-entry
vocabulary, not captured actual model logits. The fixed seed and byte hashes
bind `random`, `ascending` and `late-tie` inputs across both runs.

| One-request argmax | Control us | Candidate us | Speedup | Paired 95% speedup interval |
| --- | ---: | ---: | ---: | ---: |
| Random | 173.8976 | 11.2720 | 15.43x | 14.69x to 16.54x |
| Ascending | 173.5456 | 10.6624 | 16.28x | 15.89x to 16.62x |
| Late tie | 173.6800 | 11.0192 | 15.76x | 14.56x to 16.35x |

Four independent argmax requests reuse the same partial workspace serially.
Their complete control/candidate medians are 690-694 / 41-43 us across the
two runs. Fresh four-request speedups are 16.29-16.60x. This is not a new
batched sampler or concurrent scratch sharing.

The independent [audit](2026-10-26-maxwell-decode-reductions-audit.json)
recomputes medians, gains, stage gates, interval bounds and profile weighting
from the raw pairs. It does not treat the recorded bootstrap as independent
GPU evidence. Only stages passing all three one-request cases contribute;
the minimum observed gain per stage makes the scope calculation conservative.
The profile fractions come from generated step 255 of the qualified packed
runtime: attention norm 1.9857%, FFN norm 2.0656%, argmax 1.3236%.

## Compiled Implementation And Memory

Clean source `61a48d9147adfaca4252f8d721de1bae7236467a`, release executable
SHA-256 `1e655a9e5fdc2ca3e60092d8ad4602d647c187290717e0c7e524d9b4f9f1eb34`.
NVCC 12.8.93 uses `-O3`, sm_52 SASS plus compute_52 PTX, no fast math;
cuobjdump 12.8.90 inspects the same executable before timing.

| Kernel | Registers | Static shared bytes | Stack/local bytes |
| --- | ---: | ---: | ---: |
| Original weighted norm | 15 | 0 | 0 / 0 |
| Original-tree norm warp finish | 14 | 0 | 0 / 0 |
| Original one-block argmax | 9 | 2048 | 0 / 0 |
| Argmax partial scan | 14 | 2048 | 0 / 0 |
| Argmax final reduction | 13 | 2048 | 0 / 0 |

Both norm variants additionally request 1024 dynamic shared bytes per block.
The old SASS has two barrier instruction sites, including the reduction loop;
source/control flow executes nine block barriers. The candidate has two
barriers and five shuffle steps while preserving the old FP32 sum tree and
output association. Argmax uses at most 48 scan blocks plus one final block,
256 threads each, and one barrier per candidate kernel. No occupancy, stall
or bandwidth attribution follows from registers or source alone.

Both arms retain the same buffers. Additional guarded allocation per norm
case is 16,512 bytes at rows1 and 65,664 at rows4. Argmax uses 608,324 /
2,431,568 guarded bytes at requests1/4, including logits and output. The
actual partial payload is **384 bytes**, with 64 more guard bytes; it is
allocated once per case and reused, not allocated inside enqueue.

The original 3,279,519,744 source bytes and 1,192,935,424 prepared bytes
remain resident, totaling 4,472,455,168 tracked bytes before case scratch.
All 37 prepared keys and payload sizes match the frozen control. No KV or
full FP32 model cache is allocated. End-of-run tracked bytes return 0 to 0.

## Validation And Isolation

All 36 validation rows across the two runs match control output bits or
indices exactly. Maximum norm error against the independent FP64 reference
is 1.597e-5. CPU argmax indices, including lowest-index late ties, match.
Guards, immutable read spans and allocation residency pass. Every measured
enqueue checks the actual selected operation and forbids device alloc/free,
copies, stream sync/wait and cuBLAS calls. Both argmax kernels are counted.

Initial process 2846382 and fresh process 2850083 run serially. All 540
primary intervals are non-overlapping; all 540 pair-boundary GPU samples
show only the benchmark, P0, 1113 MHz SM / 3004 MHz memory. Both processes
exit zero, and a post-process query confirms no GPU compute processes.
No build, test, inference peer or MemPalace operation overlaps timing.

The shared benchmark suite passes 18 CPU/argument/statistical tests, with
12 hardware tests explicitly ignored in that invocation. The new hardware
adapter is then invoked explicitly and passes with cuBLAS on and off.
Non-CUDA warning-as-error tests, CUDA/server all-target Clippy, formatting
and diff checks pass. Prior primitive on/off and affected CUDA regressions
remain recorded at `f8c00bf` / `125d3a3`; this benchmark adds no kernel change.
The artifact audit rejects nine mutations: missing sample, changed prepared
identity, wrong dispatch, timed allocation, changed input identity, incorrect
index, false median, false profile scope and competing GPU process.

## Reproduction And Artifacts

Source the project toolchain and restore the frozen environment before build
and run; do not set new diagnostic environment keys:

```bash
source scripts/dev-env.sh
while IFS=$'\t' read -r key value; do
  export "$key=$value"
done < <(head -n 1 docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl |
  jq -r '.execution_configuration.environment | to_entries[] | [.key,.value] | @tsv')
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --decode-reductions --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-reduction-screen.jsonl
```

Use a new output and `--trials 10` for a fresh confirmation; do not pool.
The bounded 4096-iteration untimed warmups retain the original 1000-ms
control and 100-ms candidate requirements. Primary intervals stay ten
iterations. Request hardware access and inspect competing processes first.

- [Initial five-pair raw report](2026-10-26-maxwell-decode-reductions-screen.jsonl)
- [Fresh ten-pair raw report](2026-10-26-maxwell-decode-reductions-confirmation.jsonl)
- [Independent audit](2026-10-26-maxwell-decode-reductions-audit.jq) and [derived results](2026-10-26-maxwell-decode-reductions-audit.json)
- [Active control/candidate SASS](2026-10-26-maxwell-decode-reductions.sass.gz) and [resources](2026-10-26-maxwell-decode-reductions-resources.txt.gz)
- [Initial log](2026-10-26-maxwell-decode-reductions-screen.log.gz), [confirmation log](2026-10-26-maxwell-decode-reductions-confirmation.log.gz), [release build log](2026-10-26-maxwell-decode-reductions-build.log.gz) and [checksums](2026-10-26-maxwell-decode-reductions.sha256)

Next work needs independent evidence for enough removable qualified-runtime
cost. These smaller reductions do not close the original gap and must not
be combined with rejected FFN/projection mappings to bypass the gates.
