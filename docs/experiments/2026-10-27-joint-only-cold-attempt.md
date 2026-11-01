# Rejected Cold-Counter Quality Attempt

The first independent quality runner at `eec21c1` omitted the existing
qualified runner's `Say ready.` two-token warmup. All joint payloads, 512
actual IDs, seven full-vocabulary bit comparisons and twelve frozen outputs
matched, but the unchanged external audit rejected ordinary dispatch totals.

It recorded 43 allocations / 26 frees for the first request, versus frozen
42 / 42. Operation launches, copies, stream waits and synchronizations all
matched. Both ownership arms had identical counts. First-use retained
scratch had been included rather than prepared before the comparison.

The [raw invalid attempt](2026-10-27-joint-only-cold-attempt.jsonl) is retained,
not edited or pooled with a rerun. Add exactly the prior runner's warmup,
then rerun quality before any timing. Do not weaken the audit or change
ownership, kernel, numerical, memory or statistical gates. This attempt
does not constitute completed storage/HTTP qualification.
