# Decision: Two Structural Routes To 56.153 HTTP tok/s

## Recommendation

**Defer both implementations.** Prefer a concrete compact-drafter artifact lead
over another kernel experiment, but do not commission training or repeat a
generic search without such a lead. The FFN route still lacks a sufficiently
supported new access mechanism. Learned drafting is unresolved, not rejected.

Preserve the immutable **50.910426 HTTP tok/s** checkpoint, joint-only preferred
ownership, defaults and all acceptance gates. Preserve the four-row verifier
and explicit prompt-copy capability independently of this benchmark objective.
The successful negative dataflow audit neither establishes a hardware ceiling
nor licenses a collection of small fusion experiments.

## Comparable Opportunity

For 512 tokens, reference latency is `512000/50.910426 = 10056.879 ms`.
The 56.153149 target requires 9117.921 ms: **938.958 ms saved**, 9.34% lower
latency or 10.30% higher throughput. The additional 53.06 requirement remains.

| Route | Minimum missing evidence | Full-request opportunity | Current admission |
| --- | --- | --- | --- |
| New FFN weight-access architecture | A distinct representation/consumer with demonstrably avoided work and bounded added costs | Conditional 20-30% gate/up latency reduction gives 55.07-57.41 HTTP tok/s; not an evidenced achievable range | No qualifying architecture identified |
| Compact learned drafter | Compatible trained artifact plus joint causal acceptance/coverage and all-in M40 cost | At 95% conditional acceptance, <=3.884 ms/decision reaches the modeled target; free perfect drafting gives a 64.37 HTTP ceiling | Unresolved; no qualified artifact/cost pair |

Neither route has a defensible numerical forecast of its *largest attainable*
gain. The conditional scenarios below quantify what must become credible,
not what implementation will deliver. No new performance measurement is claimed.

## FFN Architecture

The current joint consumer reads each packed gate/up payload once per row,
publishes original scales once, and shares activation loads between projections.
The historical reader audit's repeated scale conversions are already addressed
by the qualified compound. There is no demonstrated duplicate weight stream
to remove. Physical amplification and cache stalls remain unmeasured. [1][2]

The reviewed alternatives do not establish a new candidate: wider packets,
column/CTA reuse, role scheduling and small fusions have closed experiments;
indexed byte and channel-conditioned coding miss their size gates. The latter
saves only 0.41-1.88% with metadata. Those results do not prohibit all lossless
formats, but do not justify another predictor/codebook search. FP32 expansion
increases a gate/up pair from 47,906,816 to 180,355,072 bytes; conversion removal
alone does not establish a net win. No architecture is invented to fill the
comparison table. [2][3]

Keep prepared down/vocabulary fixed. With sampled all-layer gate/up cost
7.417376 ms/token, the sensitivity equation is
`HTTP = 512000 / (10056.879 - 512 * 7.417376 * r + added_ms)`.
At zero added cost, latency reductions of 20%/25%/30% save
759.539/949.424/1139.309 ms, yielding 55.070/56.218/57.415 tok/s.
Closing the gap requires about **24.7% gate/up reduction**. The 25% scenario
leaves only 10.466 ms/request for added work. Eliminating the entire stage
would give 81.80 tok/s, an unattainable zero-work scope bound, not a forecast.
Instrumented middle-stage cost is not an exclusive full-request attribution. [1]

Minimum next evidence: a source-level access/dependency diagram naming the
removed work, original FP32 association, resident/peak bytes, metadata,
preparation and per-token costs. Original GGUF remains for prefill; joint-only
ownership and down/vocabulary cannot be silently changed. Reject any proposal
whose case depends only on presumed DRAM savings, renamed closed variants or
combining optimistic small-kernel bounds.

## Compact Learned Drafter

One bounded source pass revisited EAGLE-3 and frozen Medusa-1 heads, with
Qwen0.5B only as a cost comparator. Official artifact lists and targeted
Qwen2.5-3B searches establish no compatible compact head. This is not proof
of absence. Qwen-family support or equal vocabulary size does not establish
token-ID, feature, precision or target-revision compatibility. [4][5][6]

EAGLE-style committed-feature reuse avoids a separate full transformer, but
requires target-specific training and causal feature/draft-KV maintenance.
Frozen auxiliary heads avoid transformer passes but can be expensive: three
full Qwen vocabulary heads alone require 1,866,989,568 FP16 bytes. The measured
target vocabulary stage suggests a 4.693 ms three-head cost analogy, not a
measurement of a trained compact head. Medusa-2 backbone changes and typical
acceptance cannot satisfy the unchanged-target contract. [7]

The measured verifier costs 57.820 ms and has one known seed plus three guesses.
The existing finite-request model charges correction, rollback, ineligible
decisions and tails. Its historical HTTP scenario reserves 554.538 ms plus
0.370 ms setup. Median perfect/free-draft latency is 7954.435 ms, or 64.367
tok/s: a projected 2102.444 ms saving, **not measured HTTP throughput**. [8]

At full eligibility, D=1 ms needs 92.03% equal conditional acceptance for the
target; D=5 ms needs 96.12%. At p=0.90 even free proposals miss it. At p=0.95,
full-batch acceptance is only 85.74%, and D must fit 3.884 ms. Initialization,
feature capture, vocabulary, transfers, draft repair and same-GPU contention
must fit the accounting, not be assumed hidden. Late rejection costs and
correlated acceptance/coverage remain gaps. No future-token access or fitting
against final held-out prompts is acceptable. [7][8]

## Cost Of Decisive Evidence

Effort ranges are planning estimates for one experienced engineer, not measured
durations or authorizations. Subsequent work requires separate approval.

| Smallest next study | Effort/resource estimate | Pass or stop |
| --- | --- | --- |
| Review a genuinely new FFN design supplied with an avoided-work argument | 1-2 engineer-days, CPU/source analysis; no GPU initially | Stop unless exactness, memory and complete-request scope survive accounting; only then price one fixed operator experiment |
| Validate a named compact artifact | 1-2 engineer-days for provenance, tokenizer/features, license and tensor inventory; metadata only | Stop if incompatible or lacking required trained weights |
| Proposal-only study after artifact validation | 3-5 engineer-days assuming an executable reference exists; reserve 4-8 isolated M40 hours for a bounded pilot, not a guaranteed full study | Measure conditional acceptance, coverage and all-in cost jointly on precommitted prose/code/retrieval/copy and context bands; stop if the unchanged model/gates cannot support worthwhile HTTP savings |
| Obtain a missing trained head | 2-3 engineer-days to specify data, architecture, licensing and pilot budget | Training GPU-hours/cost are not estimable yet: missing parameterization, token count, teacher-feature generation and training throughput. Do not start an open-ended training project |

No GPU probe ran: existing controls cannot reveal an absent architecture or
drafter acceptance, and unavailable counters cannot be replaced by checksum
timing. Calculations were checked against raw model outputs; preserved reference,
dataflow, verifier and draft artifacts retain their hashes. No code/API changed.

**Next investment:** review one named compatible drafter lead if supplied;
otherwise wait for that lead or a genuinely new FFN design. Any later operator
still needs >=5% complete-FFN gain and positive paired bounds, followed by exact
512-token/frozen-quality/ownership checks and unchanged HTTP statistical gates.
Fresh official Ollama comparison is required for target qualification. Stop here.

## Sources

1. [Dataflow measurements and buffer ledger](2026-10-28-dataflow-findings.md).
2. [Historical reader-work audit](2026-10-26-maxwell-reader-work-audit.md); current `cuda/q8_ffn.cuh`, `gate_up_joint_f32x4_shared_scales` inspected at `69a293b`.
3. [Channel-conditioned size result](2026-10-26-maxwell-channel-entropy.md).
4. [EAGLE official artifact list](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/README.md), accessed 2026-10-28.
5. [Medusa official artifact list](https://raw.githubusercontent.com/FasterDecoding/Medusa/main/README.md), accessed 2026-10-28.
6. [Qwen0.5B configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json), accessed 2026-10-28; not a qualified draft artifact.
7. [Learned-drafter assessment and licensing/shape evidence](2026-10-28-learned-draft-feasibility.md).
8. [Measured verifier model](2026-10-28-direct-verifier-model.json) and [accounting assumptions](2026-10-28-direct-verifier-feasibility.md).
