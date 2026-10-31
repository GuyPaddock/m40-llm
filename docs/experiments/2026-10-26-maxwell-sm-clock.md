# Qualified FFN SM-Clock Sensitivity

## Finding

At fixed 3004 MHz memory, reducing observed SM frequency **1113 to 822 MHz**
increases unchanged one-row gate/up latency **6.60-6.79%**, prepared down
**13.13-13.31%** and complete FFN **9.00-9.11%**. The same-plane checksum
reader increases **4.73-4.78%**. All one-row paired intervals are positive.

This is a diagnostic of the qualified work, **not an optimization or a new
throughput baseline**. The application/boost/persistence policy is restored
and independently checked after all ten processes exit. Preserve qualified
**49.146 HTTP tok/s** versus **43.106 official Ollama** and both unmet targets,
**53.06** and **56.037 HTTP tok/s**. No runtime, model, KV, compiler or
preferred clock/default changes follow.

The [contract](2026-10-26-maxwell-sm-clock-hypothesis.md) fixes five outer
process pairs and two supported states, without a new kernel. This follows
the [logical reader audit](2026-10-26-maxwell-reader-work-audit.md), which
cannot identify physical traffic by itself. The new response is independent
evidence of frequency sensitivity, not DRAM, occupancy or stall measurement.

## One-Row Results

GPU-event milliseconds per complete stage call. Each table entry is a
median of five process medians. Each process first performs its existing
five internal rounds, ten calls per stage, with separate bounded warmup.
The outer process pair is the independent unit; internal rounds are not
pooled as 25 independent pairs.

| Layer | Stage | 1113 MHz ms | 822 MHz ms | Latency increase | Paired 95% interval |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Read/checksum | 0.209642 | 0.219629 | +4.76% | +4.66% to +5.47% |
| 17 | Read/checksum | 0.209619 | 0.219542 | +4.73% | +4.57% to +5.07% |
| 35 | Read/checksum | 0.209549 | 0.219574 | +4.78% | +4.69% to +4.82% |
| 0 | Gate/up/SwiGLU | 0.221680 | 0.236730 | +6.79% | +6.55% to +6.95% |
| 17 | Gate/up/SwiGLU | 0.221885 | 0.236525 | +6.60% | +6.42% to +6.74% |
| 35 | Gate/up/SwiGLU | 0.221690 | 0.236646 | +6.75% | +6.70% to +6.87% |
| 0 | Prepared down | 0.132224 | 0.149670 | +13.19% | +12.63% to +13.84% |
| 17 | Prepared down | 0.132496 | 0.150128 | +13.31% | +13.12% to +13.41% |
| 35 | Prepared down | 0.132182 | 0.149542 | +13.13% | +12.58% to +13.83% |
| 0 | Complete FFN | 0.352762 | 0.384906 | +9.11% | +8.86% to +9.22% |
| 17 | Complete FFN | 0.352842 | 0.384912 | +9.09% | +8.99% to +9.49% |
| 35 | Complete FFN | 0.352771 | 0.384518 | +9.00% | +8.90% to +9.20% |

All real one-row FFN stages exceed the prespecified 5% response threshold
with positive intervals. Checksum point estimates are below 5%, but the
layer0/17 intervals include 5%; they do not prove its response is below
that threshold. No additional clock sweep is needed to establish the clear
complete-FFN response. This is not a candidate-promotion decision.

Exhaustive ordered paired bootstrap uses all **5^5 = 3125** resamples of
the five outer pairs. Percentile indices are floor(n*0.025)=78 and
floor(n*0.975)=3046. Each resample computes median(low)/median(high)-1.
This deterministic diagnostic is distinct from the seeded optimization
promotion protocol; no results or trials from previous studies are pooled.

Genuine four-row work is reported separately in the raw/audited results.
Across layers, latency increases are 4.35-4.38% checksum, 4.95-6.85%
gate/up, 5.66-8.39% down and 4.87-5.26% complete FFN. Four-row work has
different stream dependencies and is not single-request/HTTP evidence.

## Interpretation

The observed frequency reduction is 26.15%. Pure inverse-frequency scaling
would imply 35.40% more latency; none of these stages follows that model.
One-row log-frequency elasticity is about 0.15 checksum, 0.21-0.22 gate/up,
0.41 down and 0.28-0.29 complete FFN. Those are response summaries, not
fractions of arithmetic or DRAM time.

Prepared down is more frequency-sensitive than gate/up, and the checksum
is less sensitive than the real stages. That is useful when evaluating a
future avoided-work hypothesis. It does **not** prove that integer issue,
conversion, reductions or another specific instruction class caused the
response. SM/graphics frequency can affect request issue, latency hiding
and other device work as well as arithmetic. This is not an independent
control of every internal GPU domain.

Likewise, weak checksum sensitivity does not establish DRAM saturation.
No DRAM transactions, cache-hit rates, warp stalls or achieved occupancy
are measured. The checksum remains a surrogate, not an inference lower
bound. Do not fit a two-point model and label its remainder pure memory time,
or extrapolate an overclock/new HTTP rate from this underclock diagnostic.

No new implementation is admitted. Closed arithmetic, scale, packet,
geometry and cache probes remain closed. A new organization/reuse proposal
still must identify expensive work avoided, include resident/peak/setup
and per-token costs, and pass complete-FFN, actual512/frozen-quality and
HTTP gates. Frequency response alone is insufficient justification.

## Provenance, Work And Restoration

Clean source `c2db5fed1e1407951e676aa799845640f24a5d25`; release benchmark
SHA-256 `aa218ae8b24422d7ace65c1563fabc3e36ec93b61ae661eb5013d74fc5add579`.
CUDA 12.8.93, `-O3`, sm_52 SASS/compute_52 PTX, no fast math, same frozen
M02 model, capture, fixture, inputs and environment. No CUDA, runtime or
benchmark implementation is changed. Original split-f32x4 gate/up and
prepared down remain the actual dispatch in every process.

All 37 prepared down/vocabulary identities and their 1,192,935,424 bytes
match frozen M02. Source weights retain 3,279,519,744 bytes. Each case
adds the same 47,906,816-byte gate/up pair and existing guarded scratch;
requested totals remain 4,520,651,264 bytes at one row and 4,521,517,568
at four. This is not peak VRAM or full-runtime 109-tensor preparation.
No KV/model-token generation is performed.

All 60 numerical records across ten processes pass source-derived checksums,
existing full-output CPU tolerances, capture/normal parity and guards.
Checksum hashes match across all ten processes; prepared planes/inputs
remain intact. Every process returns tracked bytes zero to zero.

There are **1200 nonoverlapping primary intervals** and **600 primary GPU
boundaries** across ten processes. Each boundary shows only its owned
benchmark PID, P0, SM within 2 MHz of the requested 1114/823 setting and
memory exactly 3004 MHz. Five outer pairs alternate high/low order. Warmup
records remain separate. No inference, builds, tests or MemPalace work
overlaps timing. Session 33239 exits zero without retries/restarts.

The startup script first verifies idle GPU and original defaults, then
temporarily enables persistence, disables default auto boost and selects
only the two supported application-clock pairs. Its EXIT/INT/TERM cleanup
waits/stops only the owned child and restores the original settings.
Final and independent read-only checks confirm **3004/947 application and
default clocks, auto boost/default On, persistence Disabled and no compute
processes**. No power-limit, voltage, ECC or driver setting is changed.
The lower 405/324 memory/SM pair is idle-only, so no independent memory
frequency sweep was attempted.

## Artifacts And Validation

The independent audit replays raw records, numerical checks, dispatch,
memory, warmup, observed clocks, process isolation, intervals and restored
policy before computing process medians and exhaustive paired intervals.
Fourteen corruption controls reject bad clock/work/identity/numerical,
overlap, allocation, leak and restoration evidence. Links, checksums,
shell syntax, diff checks and formatting pass. There is no new CUDA source
change requiring another broad test suite; these are diagnostic benchmark
and artifact-validation results, not a claim that all project tests ran.

```bash
bash docs/experiments/2026-10-26-maxwell-sm-clock-audit.sh
bash docs/experiments/2026-10-26-maxwell-sm-clock-audit.sh --negative-controls
```

- [Raw records, logs, policies and startup script](2026-10-26-maxwell-sm-clock-data.tar.gz)
- [Audit](2026-10-26-maxwell-sm-clock-audit.jq), [derived results and every outer pair](2026-10-26-maxwell-sm-clock-audit.json)
- [Corruption-control log](2026-10-26-maxwell-sm-clock-controls.log.gz), [checksums](2026-10-26-maxwell-sm-clock.sha256)

The archived hardware script is intentionally machine-specific and rejects
other initial policies. Replaying the audit requires no GPU/root and changes
no hardware state. Repeating the hardware diagnostic requires fresh paths,
explicit escalation, exclusivity and the same restoration protections.
