# Full November quality archive and independent replay

The independent reader reproduced 348 aligned comparisons from 522
full FP32 vocabulary vectors. It recomputed logit errors, target
probability and rank, KL divergence and top-ten overlap. The checks
covered the original dense logits, all-history mixed KV and top-8 mixed
KV at every retained prefix. The [replay receipt](independent-replay.json)
matches a second run from the extracted archive.

The 48 prompts each produced three free generations. All-history
mixed KV preserved 19 of 19 dense-passing short cases and 10 of 10
dense-passing long cases. Top-8 preserved all 19 short cases and seven
of ten long cases. Both candidate denominators exclude the five short
and 14 long dense failures. Within the broad grid, the all-history
arm's top-64 budget exceeded the number of historical blocks. The
separate boundary study tests truncation beyond that budget.
[Generation outcomes and scoring](../continuations.md).

The raw package lives in `research-1121/raw/kv-quality-bundle-01`,
outside Cargo's target directories. The 57 hashed payloads total
288,399,995 bytes, including 48 independently compressed case shards.
The [file inventory](files.tsv) pins sizes. [Transport hashes](bundle.sha256)
cover package files. [Source hashes](raw.sha256) cover decompressed
inputs, original logs, the complete report and every raw FP32 vector.
The full vectors stay in the durable local package.

Run the following from a consolidation checkout containing the
`2026-11-21-kv-audit` tools at
`2db24717ad681d545f127b762c22cd5cd73a3e73`. Set `package` to the
durable bundle and `receipt` to this report's directory.

```bash
audit=docs/experiments/2026-11-21-kv-audit
cmp "$package/bundle.sha256" "$receipt/bundle.sha256"
cmp "$package/raw.sha256" "$receipt/raw.sha256"
bash "$audit/unpack.sh" "$package" NEW_DIRECTORY
python3 "$audit/replay.py" NEW_DIRECTORY/report.jsonl NEW_DIRECTORY/inputs.json |
    diff - "$receipt/independent-replay.json"
```
