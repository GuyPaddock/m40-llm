# Bounded V3 engineering follow-up

Frozen before measurements on October 31. No full corpus, target outputs,
GPU work, training or selected source manifest is authorized.

One index only: two-pass global shingle-frequency order, deterministic digest
ties, exact Jaccard >=0.5 prefix candidates and safe size filtering. Full
exact/32-token matches remain unfiltered. Preserve all verified raw pairs;
deadlines stop as unknown rather than dropping frequent keys. No exemptions.

One solver only: OR-Tools CP-SAT 9.14.6206, one worker, seed 20261029. Preserve
the whole-component integer formulation. Optimize hash departures first and
then the exact lexicographic order in integer mixed-radix chunks bounded by
2^50. This avoids floating-point epsilon objectives; incomplete chunks do not
constitute a canonical result. Use the original independent witness checker.

Separate commits cover index/tests, solver/tests/environment pins, independent
serialized evidence audits, then measured results and review strategy.

## Fixed CPU probes

Run serially with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Each process
has a 4 GiB address-space limit and 720 CPU-second hard cap. Matching allows
600 CPU/900 wall seconds including both passes; solver allows 90 CPU/120 wall
seconds including construction and up to 128 optimizer calls. Total planned
caps remain below eight CPU-hours. No increasing caps to rescue a failed case.

| Index case | Scope |
| --- | --- |
| retained | 846 retained original-Qwen inputs; exhaustive raw-pair comparison |
| header2k/header10k | 2,000/10,000 synthetic 256-word/ID records with twenty shared header words |
| copied200 | 200 distinct records sharing an exact 64-token passage; all 19,900 pairs required |
| population | 390,220 synthetic 64-word/ID records with twenty shared header words; timed-out population work remains unknown |

| Solver case | Scope |
| --- | --- |
| fixed72 | Full frozen quotas, 72 fixed-role groups; all capacities must be used, giving an independent uniqueness proof |
| small | 2,048/512/128/512 components across prose/code/QA/extraction |
| medium | 8,192/2,048/512/1,024 components |
| large | 65,536/8,192/1,024/5,051 components |
| upper | 230,975/56,025/98,169/5,051 one-row components, matching native-record upper bounds rather than known document counts |

Mixed fixtures retain native role types. Bucket proportions approximate the
retained eligible sample shapes solely as stress inputs, not population
estimates. Code uses one to three rows/component; QA sixteen except in the
one-row upper case. Neither source grouping nor eligible capacity is inferred.
All solver scale probes retain 256/48/96 per family, sixteen held-out groups
per family and global 32/32/32 bucket minima.

No second indexing architecture or optimizer follows a negative decision.
CPU measurements do not replace independent human registry review. Return the
results and a leakage-safe review cost assessment before any census request.
