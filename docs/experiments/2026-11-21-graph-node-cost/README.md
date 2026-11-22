# Cost of the conditional graph attention fallback

A repaired graph captured for a capacity above 8192 records a specialized
attention node plus a scalar node. At a smaller live length, the scalar
node reads the length and returns. Short-context B1 HTTP runs use a graph
with only the specialized node. Quantify the extra node before considering
a change to long-capacity graph execution.

Compare capacity 8192 against 8193. Both reserve shared memory for 8192
scores, use physical slot zero, and read the same contiguous FP16 keys
and values. Use Qwen's head128, 16 query heads, two KV heads, and fixed
live lengths 65 and 2048. As a placebo, run generic head7 at length 65;
both capacities then capture one scalar node. Inputs are deterministic
synthetic tensors. Verify the device, capture node counts, bitwise output
against ordinary host-length execution, unchanged KV lengths and tracked
teardown. Record actual owned cache bytes and release executable hash.

Compare ordinary and graph dispatch at both capacities. Use five paired
rounds, alternating and rotating order, 100 calls per measured interval
after warmup. Retain ten rounds so the first five and the separate five
can be inspected if timing is noisy. Record CPU enqueue, event span and
enqueue-through-completion separately. Compare paired capacity differences
within each path; report the ordinary and placebo capacity effects. Setup
and instantiation have their own untimed diagnostic records. Attribute an
extra cost to the specialized two-node configuration only if reproducible
above variation while same-node controls remain stable. An inconclusive
result closes this bounded screen. A further prototype needs at least 5%
credible complete-token headroom and a distinct whole-model comparison;
the operator result alone does not qualify pruning of a required fallback.

The benchmark at 783d0e5 uses the production CUDA graph, decode-stream
event and KV-cache APIs. Record [the executable](binary.sha256) using
`M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo build --release --locked --features cuda --example graph_node_cost`.
Run [the serial driver](run.sh) from this worktree. The dimensioned input
generator and live lengths are in examples/graph_node_cost/measure.rs;
no GGUF model is loaded in this operator test.

The [analysis](analyze.py) verifies each actual graph's queried node
count. For each round it subtracts the ordinary capacity difference from
the graph capacity difference. It reports event, CPU and wall means per
call, cohort medians, both five-round subsets and 10,000 paired
resamples of ten rounds. [The accounting check](check-pairs.sh) separated
a common 2 µs capacity effect from an added 3 µs graph effect, recovered
zero on the generic placebo and rejected an incorrect node count.
