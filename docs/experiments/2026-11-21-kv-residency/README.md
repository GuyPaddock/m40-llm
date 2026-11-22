# Why mixed KV is larger at capacity 3,072

Both mixed constructors reported 116,499,600 bytes on the M40 while dense
reported 113,246,208 bytes. The generation rows corroborate these values
on the same long prompt at 2,058 consumed positions across all 36 layers.
The [constructor records](constructor.txt) and [independent reconstruction](reconciled.json)
agree to the byte. Top-8 and top-64 allocate identical backing storage.

| Mixed component | CUDA allocation requests, bytes |
|---|---:|
| Recent FP16 K and V, 1,024 positions | 37,748,736 |
| Historical FP16 K, capacity 3,072 | 56,623,104 |
| Historical packed Q4 V codes, capacity 3,072 | 14,155,776 |
| Historical V scales, one FP32 per head and position | 884,736 |
| Block summaries, FP32 accumulators, block counts, slot lengths | 7,087,248 |
| Total | 116,499,600 |

The historical arrays reserve capacity for all 3,072 positions while the
recent ring holds a further 1,024. Streaming eviction addresses historical
position `position - recent_window`. At full capacity it can populate
2,048 historical positions. The summary arrays already use that smaller
range, in 64 blocks. `KVCache::actual_bytes` counts the 144-byte slot map
in the mixed total. The dense constructor reports only its K/V payload;
its slot map is another 144 bytes. This accounts for the difference
between constructor totals and direct allocation requests.

For one Qwen request, eliminating the entire unused tail from the three
historical arrays would save at most **23,887,872 requested bytes**.
That would reduce this mixed cache's requests to 92,611,728 bytes, before
allocator rounding. The tail is only **0.5341%** of the 4,472,663,048
Rust-tracked device bytes observed at the same boundary. The same tracked
total appears in all three arms because native KV uses its own CUDA
allocations. Trimming the tail cannot by itself reach the single-request
10% peak-VRAM gate on this cohort. It would also require changing strides
in allocation, initialization, eviction, readers, debug access and reset,
with special handling for the zero-history case and independent slots.

**FAIL for a tail-only single-sequence redesign under the memory gate.**
Independent live CUDA and per-process memory observations at 1,024, 3,072
and 8,448 capacity will establish total residency, allocation rounding
and release behavior. The numbers above are logical allocation requests
and a counterfactual bound. Larger sequence pools have different economics.

Source c49e208 on research/1121-quantized-kv:
`src/cuda.rs::KVCache::new_compressed_with_context` and the
`allocate_fp16_old_k_backing`, `allocate_q4_old_v_backing` and streaming
eviction kernels in `cuda/kernels.cu`. The [manifest](manifest.json)
identifies the model and measured executable. Reproduce using the
[original generation rows](../2026-11-21-kv-dense/top8-counterexamples.jsonl.gz).

```sh
study=docs/experiments/2026-11-21-kv-residency
jq -R -s 'split("\n")|map(select(length>0))' "$study/constructor.txt" |
  jq --slurpfile rows <(gzip -dc docs/experiments/2026-11-21-kv-dense/top8-counterexamples.jsonl.gz |
       jq -c 'select(.case=="nov-order-long")') --slurpfile manifest "$study/manifest.json" -f "$study/reconcile.jq"
```
