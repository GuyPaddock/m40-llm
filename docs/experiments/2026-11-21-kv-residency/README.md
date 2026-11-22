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

## Live process residency on the M40

**FAIL for the 10% single-request peak-VRAM promotion gate.** Nine fresh
processes loaded the same Qwen2.5-3B Q8_0 weights. Each retained the
weights while allocating, populating and releasing one 36-layer cache.
The fixed append inputs exercised every position of every layer. This
measures cache residency with real model weights and synthetic KV inputs.
Whole-generation peaks include additional execution workspaces.

| Capacity | Dense requested bytes¹ | Mixed requested bytes¹ | Dense peak MiB² | Mixed peak MiB² | Mixed minus dense MiB |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 37,748,880 | 61,747,488 | 4,413 | 4,435 | +22 |
| 3,072 | 113,246,352 | 116,499,600 | 4,485 | 4,488 | +3 |
| 8,448 | 311,427,216 | 260,514,576 | 4,674 | 4,627 | −47 |

¹ Includes the slot map in both arms. Top-64 and top-8 agreed on every
requested byte and peak. At 8,448, mixed backing saved 16.35% of cache
requests and 1.01% of observed whole-process peak memory.

² Maximum of synchronized process-attributed NVML observations and a
500 ms process sampler. The sampler alone saw only the 4,377 MiB model
baseline in all three short runs. CUDA free-memory snapshots and the
payload breakdown are in the [nine-process summary](live/summary.json).
The Rust device counter stayed at 4,472,456,192 bytes throughout.

After release, CUDA usage returned to baseline in every dense process
and in both recent-only mixed processes. Each mixed process that
populated historical positions retained 1 MiB above baseline at the
final observation. There was no GPU owner after process exit. A single
allocation per process cannot distinguish one-time initialization from
accumulation across cache lifetimes. This limits reuse qualification.

### Capacity at which mixed payload first gets smaller

At this geometry, a dense position uses 36,864 bytes across all layers.
The mixed historical arrays reserve 23,328 bytes per capacity position.
The fixed recent ring costs 37,748,736 bytes and each historical
summary block costs 110,736 bytes across the 36 layers. The 144-byte
length map is common to both arms. The independently reconstructed
allocation requests put mixed 4,464 bytes above dense at capacity
3,402 and 9,072 bytes below at 3,403. Mixed stays smaller at every
subsequent capacity in the 1,024–8,448 range. Those adjacent capacities
were calculated from the allocator formula. The three capacities in the
table were measured on the M40. The crossover does not predict the exact
NVML peak or justify changing the cache policy at a particular length.

```sh
jq -n '
  def difference($n):
    37748736 - 13536*$n + 110736*([1,(($n-1024)/32|ceil)]|max);
  [range(1024;8449) | {capacity:., extra_mixed_bytes:difference(.)}] |
  {last_larger:map(select(.extra_mixed_bytes>=0))[-1],
   first_smaller:map(select(.extra_mixed_bytes<0))[0]}'
```

The [raw archive](live/measurements.tar.gz) includes all nine process
records, model and executable hashes, build and dispatch flags, host
load, sampler observations and timestamps. Archive SHA-256
`8346b95e5a0bb10bce7159304c03fe8d3c542523311b411de63f0555ac84d3f2`.
At source `9d4b78c24741342c81f1f4b61320e89f501e35d6`, use the
release example `kv_quality residency --help` for capture arguments.
To replay with the independent audit from consolidation commit
`6ada455f19bb470f2b610fdb8bb6104e79fe0b52`:

```sh
study=docs/experiments/2026-11-21-kv-residency
audit=docs/experiments/2026-11-21-kv-audit
out=$(mktemp -d)
tar -xzf "$study/live/measurements.tar.gz" -C "$out"
for file in residency.sh residency.jq; do
    git show "6ada455:$audit/$file" > "$out/$file"
done
bash "$out/residency.sh" "$out/kv-residency-01" | diff - "$study/live/summary.json"
```
