# Fixed causal channel-predictability study

This protocol precedes new complete-generation observations. It tests the
user's proposed cheap signals without fitting a neural selector or reopening
the V3 data project. The six retained diagnostic prompts provide a bounded
feasibility result, not representative traffic or a held-out accuracy claim.

## Inputs and observation contract

- Original Qwen2.5-3B Q8_0 model SHA-256:
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
- Existing `data/research_quality/v1.json` SHA-256:
  `2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.
- Exactly its six calibration tasks, in file order, with unchanged token
  limits 48/32/64/16/16/160. No held-out inputs, new corpus, downloads or
  sampled replacement prompts.
- Existing paired FFN capture's native Q8 control: prepared split
  down/vocabulary, sequential prefill, explicit dense KV, greedy top-k one
  and natural stopping. Preserve its explicit qualified environment. This
  is an observation control, not the current compressed runtime default or
  a fresh whole-model speed comparison against qualified compound Q8.
- Capture target layers 0/17/35 and predecessor layers 16/34. At each
  observed position retain normalized input and completed SwiGLU values.
  Cross-layer channel numbers are not interchangeable.
- Capture every consumed generated-token input, with a 160-row per-task
  bound and 256 MiB FP32 observation budget. The prompt-terminal forward
  predicts the first generated token and is outside this existing observer;
  count it as dense bootstrap. The final generated token is not consumed
  after stopping. Require exactly `actual_generated_tokens - 1` rows per
  task/layer, not a capped prefix. Record both omissions explicitly.
- Run each full generation with observation inactive and active. Require
  identical actual IDs, output, count and stop reason. Capture overhead is
  untimed diagnostic work. Release build, source/executable hashes and
  tracked owner cleanup must be recorded. No timed GPU workload overlaps.

The old M02 four-row capture remains valid for its original purpose. Add an
explicit complete-generation option; keep its old defaults and provenance.
Budget and artifact-size limits remain enforced before allocation/parsing.

## Fixed comparisons

For each target layer and token, retain `ceil(H * fraction)` channels at
fractions 0.10 and 0.20. Exact ranks sort descending by absolute completed
SwiGLU activation, breaking ties by ascending channel ID. Also report
down-column-L2-weighted contribution mass at those same selections; do not
silently substitute a different oracle. Original Q8 vectors do not establish
exact-Q4 rankings or Q4 model quality.

Evaluate these fixed controls and signals independently at each task/layer:

1. Current full-vector top-k oracle, an unattainable free-selector bound.
2. Previous-token **oracle** top-k, explicitly diagnostic. Full true hidden
   vectors are available from the recorder, not ordinary sparse inference.
3. A deterministic random selection, independent of observed values.
4. First observed token's top-k, held fixed for the remaining task.
5. Last dense-refresh top-k with refresh intervals 8 and 16. Only refresh
   rows reveal complete oracle labels; charge those rows as full dense FFNs.
6. Cumulative channel frequency from dense refresh rows only, at the same
   intervals. Break frequency ties by channel ID.
7. Nearest dense-refresh state using a fixed 16-dimensional signed
   CountSketch of the current normalized FFN input. Keep at most eight
   refresh records, evicting the oldest. Normalize sketches for cosine
   distance and break equal distances by most recent record. A zero sketch
   falls back to the latest refresh selection.
8. The same causal lookup using the preceding layer's normalized input for
   target layers 17 and 35. Layer 0 has no transformer predecessor and gets
   no row for this signal. Neither target SwiGLU values nor future tokens
   may enter the query feature.

Freeze the hash implementation and its seed in the analyzer before running
on real observations. Projection dimensions, refresh intervals and bank
capacity are not tunable after seeing results. CountSketch needs only
signed additions over 2048 input values; retrieval compares at most eight
16-element vectors. Dense refresh is permitted at observed row zero and
every R rows thereafter. On refresh rows record a dense action and seed the
bank after evaluation; never score it as a successful sparse prediction.

Every predictor resets at each task boundary. No reference is shared across
prompts and no token-random train/test split is used. Online causal history
is the only reference set, so a test token cannot supply its own labels.
Record the exact source row used by each prediction for independent audit.
Even these causal label-access rules remain teacher-forced diagnostics:
features come from the unchanged dense trajectory. Recursive sparse
execution can change those features and still requires generation testing.

## Results, cost bounds and stop rules

Save per-row predictions and summaries by task, layer, retention and sequence
position. Report top-k recall, hidden L1 mass, down-norm-weighted mass,
relative coverage of the current oracle, and worst rows. Zero oracle mass
must be explicit rather than producing NaNs or an automatic win. Include
dense refresh frequency and short generations with no eligible sparse rows.

For a fixed engineering screen, a causal method is promising only if at
least 90% of its eligible sparse rows in every target layer have recall at
least 0.80 and weighted mass at least 0.95 of that row's oracle. Report
each task as well, including failures hidden by pooled averages. These are
triage thresholds, not model-quality guarantees or statistical bounds.

If no causal method passes, stop the fixed screen and report its limits.
Do not tune projections or begin training to improve the same evaluation.
If a method passes, compute masked FFN-output error against dense and oracle
selections, then assess a concrete GPU implementation's complete cost before
considering approximate generation. High overlap alone cannot promote it.

Use the retained per-layer Q8 and selected-Q4 timings only for a labeled
optimistic cost calculation. Charge dense refreshes, maintain the original
5% complete-FFN improvement gate, and report the remaining selector budget.
Exclude no bootstrap or refresh token from amortized counts. Actual scoring,
top-k, index handling, feature computation and memory remain unmeasured until
an M40 implementation is timed; CPU analyzer wall time is not selector cost.

Any successful signal must later survive recursive complete-generation
quality, logit and repetition checks and whole-model latency, including its
bootstrap/refresh policy. A negative result bounds these specific fixed
signals on these six prompts; it cannot rule out dynamic sparsity in general.

## Delivery

After the Q4 carry-forward dependency is verified, commit this protocol before
capture code or analysis of new observations. Then use separate commits for
the explicit complete-generation observer, tested causal analyzer, and raw
evidence with decision. Preserve failed collection attempts and the original
nine experiment branches. Update research status and the agent diary at each
completed unit. No publication or runtime-default change is implied.
