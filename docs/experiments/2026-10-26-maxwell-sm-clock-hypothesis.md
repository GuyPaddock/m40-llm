# Bound SM-Clock Sensitivity Of Qualified FFN Work

## Question And Limits

The reader ledger establishes logical work, not physical traffic. Use a
controlled frequency perturbation to test whether unchanged real FFN and
checksum costs materially depend on the SM clock at fixed memory frequency.
This is diagnostic evidence before another implementation, not clock tuning
for higher throughput or a hardware-ceiling claim.

Read-only device inspection reports one productive memory frequency,
3004 MHz, paired with SM settings including 1114 and 823 MHz. The only
other memory entry is the idle 405/324 pair. There is no independent memory
clock comparison. Original application/default clocks are 3004/947 MHz,
auto boost and its default are On, persistence Disabled, and GPU compute
is empty. Preserve this original state after the diagnostic.

[NVIDIA documents](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
that auto boost can exceed application clocks. Fixed-clock work therefore
requires temporarily disabling it, with persistence enabled. Request
explicit hardware escalation before changing these settings. Use only
supported pairs, no power/voltage/overclock/ECC/driver changes.

## Fixed Measurement

- Reuse release `q8_execution_bench --ffn-weight-read --rows 1,4
  --trials 5 --iterations 10`, frozen M02 fixture/capture, original Qwen
  model and all existing numerical, guard, cache-identity and cleanup checks.
  No CUDA, inference or benchmark implementation changes.
- Two states only, memory 3004 with requested SM 1114 or 823 MHz. Verify
  actual measured clocks within 2 MHz of the requested SM, with memory
  exactly 3004. Do not add states or rescue an unsupported setting.
- Five outer high/low process pairs, alternating order. Each process keeps
  its original internal five rounds and warmup; reduce each stage to that
  process's median before outer-pair comparison. Do not pool internal
  observations into a larger independent sample or mix prior timings.
- Compare original read/checksum, real gate/up, isolated prepared down and
  complete FFN at layers 0/17/35. Report genuine four-row work separately
  from one-row decode. No model-token/HTTP/Ollama qualification follows.
- Save process/clock/application/boost/persistence state, source/executable
  hashes, startup script and all raw records. Every timed boundary must
  show only its benchmark PID. No overlapping inference, builds, tests or
  MemPalace work. Stop on a different process or invalid observed clocks.
- Set cleanup before the first mutation. On success, error or interruption,
  stop/wait only the owned child, reset application clocks to the verified
  original defaults, restore default auto boost On and persistence Disabled,
  and verify the full original policy plus empty GPU compute. If settings
  cannot be changed safely, retain a capability rejection, not timings.

## Interpretation

Prespecify a material latency response as at least 5% with a positive
outer-pair 95% bootstrap interval. Use all 5^5 ordered paired resamples
for this diagnostic, not the separate promotion protocol. Record ratios
and log-frequency elasticity against **observed** SM clocks. The sign and
size of the response are evidence, not proof of a specific bottleneck.

SM frequency also affects request issue, latency hiding and other device
work. Large sensitivity is not pure arithmetic time; small sensitivity is
not measured DRAM saturation. No physical transactions, cache-hit rates,
warp stalls or achieved occupancy are supplied by this test. The checksum
remains a surrogate, not an inference lower bound.

No new kernel follows automatically. A candidate still needs different,
demonstrably avoided work plus memory/representation accounting and the
original complete-FFN, actual512/frozen-quality and HTTP gates. Preserve
qualified 49.146 HTTP tok/s and both unmet 53.06/56.037 targets.

## Atomic Delivery

1. `docs(research): bound SM-clock sensitivity`: this contract and safe
   restore conditions before hardware mutation.
2. `docs(perf): record FFN clock sensitivity`: completed serial evidence or
   explicit capability rejection, raw artifacts, restore verification,
   interpretation, research status and MemPalace checkpoint. No new kernel
   or preferred runtime setting.
