# Direct-Q8 Feasibility Evidence Ledger

The [approved plan](2026-10-27-direct-q8-feasibility-plan.md) permits cost
assessment and one bounded diagnostic session, not a new CUDA kernel.

## Measured Anchors

- The [rejected verifier](2026-10-27-maxwell-exact-verification-screen.md)
  takes 171.687/180.576/206.486 ms per fully accepted batch at widths2/4/8.
  Its serial control is 19.310791 ms per target input row. That control uses
  the older split FFN, not the immutable compound configuration.
- The [compound profile](2026-10-27-maxwell-compound-stage-profile.md)
  measures middle gate/up/SwiGLU7.4215, down4.7614, vocabulary1.5638,
  QKV1.3918, output1.0164 and attention0.7802 ms across all36 layers.
  These are sampled instrumentation spans, not additive unprofiled wall costs.
- The immutable 50.910426 HTTP tok/s request takes 10,056.879 ms for512
  generated IDs. Targets53.06/56.153149 require407.426/938.958 ms savings.
  Keep prefill, HTTP overhead and the511 continuation-input rows explicit.

## Source Audit

The existing verifier executes seven Q8 conversions and SGEMMs per layer
per batch: Q/K/V, output, gate/up and down. Attention is causal per row;
output normalization, vocabulary and greedy selection also run per row.
The original252 prompt cuBLAS calls become64,512/32,256/16,128 whole-request
calls at widths2/4/8, including the common prompt. No exclusive attribution
of conversion versus SGEMM wall time is present in the old report.

Qwen has D2048, H11008,16 query heads,2 KV heads of128,36 layers and151936
vocabulary entries. The seven layer matrices contain2,774,532,096 weights,
or2,947,940,352 packed Q8 bytes. Materialization writes11,098,128,384 FP32
bytes per batch and SGEMM must consume those matrices. These are logical
byte counts, not measured cache misses or DRAM traffic.

The existing compound supports research rows1/4, but launches independent
blocks for each row. Its runtime consumer remains single-row. A separate
prospective four-row consumer would share a joint payload load and scales
across four independent FP32 accumulation/reduction chains. Its pair is
47,906,816 bytes per layer,1,724,645,376 across36 layers. Ideal logical read
reuse is4x; actual physical savings are unknown. Do not assume existing
four-row scheduling gets no cache reuse.

Raw Q8 QKV/output and prepared down/vocabulary have distinct qualified
arithmetic mappings. A future full verifier must preserve those per-row
consumers or independently qualify alternatives. Merely turning off SGEMM
would route some projections through other numerical associations. The
existing scalar shared-activation prefill kernel is not the proposed reader.

## Accounting Rules

For width K, let V be the full target pass, L_a rollback cost, C_a correction
cost, D drafting cost and G_a useful progress at accepted prefix a. The
existing protocol has G_K=K and G_a=a+1 on rejection with correction.
Its first candidate is already cached. For a real drafter, model a>=1 and
acceptance of the remaining K-1 guesses; also retain the old unconditional
curve for comparison and the a=0 error-path diagnostic. Never claim K+1
new IDs from a fully accepted K-row batch.

Expected cost is D + V + sum(P(a)*(L_a+C_a)) for a<K. The drafter budget at
row-cost target S is S*E[G] minus target cost; negative budgets are infeasible.
Compute each HTTP target from per-trial fixed cost,511 rows and serial tail,
not unrelated median differences. Real drafting probabilities and new direct
kernel latency are unknown and must remain scenario inputs.

FP32 work, normalization, causal attention, vocabulary, KV writes and
rollback do not disappear when weights are shared. Provisional dense KV
occupies36,864*K bytes within the existing cache, not a new resident cache.
Per-row query lengths remain prefix+row+1, with unchanged absolute RoPE.
Require exclusive session/model ownership, retained-prefix equality,
cleared rejected suffixes, valid lengths and zero tracked teardown.

The prospective design uses2*K accumulators and shared1024*K+512 bytes:
2,560/4,608/8,704 bytes at K2/4/8. Baseline32 registers/thread is measured;
additional live accumulators, decoded weights and compiler temporaries are
estimates. No achieved occupancy or physical bandwidth is known. Maxwell's
[official tuning guide](https://docs.nvidia.com/cuda/archive/12.8.1/maxwell-tuning-guide/index.html)
documents64K registers per SM,96 KiB shared memory on second-generation
Maxwell, a48 KiB block limit and conventional FP32/vector/shuffle facilities.

## Diagnostic Contract

Use the immutable source/model/environment identities with preferred joint-only
compound ownership and explicit dense KV. Keep the original verifier mode
unchanged; the new diagnostic only observes the existing losing path.
Record five off/on pairs per width at prefix65, then five rejection cycles
per distinct accepted prefix0/floor(K/2)/K-1. Enclose only existing work;
preallocate event pools and serialize reports outside observed intervals.
Reject attribution above3% observer overhead, changed outputs/counters,
incorrect cache/lengths, competing GPU processes or incomplete cleanup.

Atomic sequence: bound/commit this ledger; implement and verify host-only
diagnostics; collect one clean release session; audit and publish the final
feasibility decision. No automatic CUDA implementation follows either result.
