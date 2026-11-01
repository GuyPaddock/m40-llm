# Rejected Primary Projection Loader Attempt

The first fresh Ollama attempt after incremental projection qualification
did **not** execute the requested M40 projection backend. Preserve it as
benchmark-plumbing evidence, not a projection speed or quality result.

Clean report source was `7dd14bb1917517cf9650ff8df83003f7b379d8d6`, using
the previous comparison executable
`7ffa5b7898affb3c67a93fd7040a1b590801c89fa66c6daf6ba3ffdef341a360`.
Its primary configuration label said `warp-256`, but all measured M40
launches used original `qkv_f32xq8_0_gguf_decode` and
`gemm_f32xq8_0_gguf_f32_decode`, each 18,432 times. Neither requested warp
operation executed. Observed M40 rates near 45 tok/s prompted the dispatch
inspection; the rate alone was not the rejection criterion.

The missing wiring was introduced when the third runtime configuration
axis was added in `1349613`: the secondary/candidate factory received it,
but the primary comparison loader still called the older native-projection
constructor. Therefore the completed **incremental** comparison remains
valid: its native primary control was correct and its separately loaded
warp candidate has verified dispatch. The real-model quality factory also
received the correct typed backend; that report remains valid.

The owned benchmark process 1506019 was stopped after dispatch confirmed
the wrong path. All ten primary requests and their five-pair summary had
already been recorded by then. The run was interrupted during untimed
trace work and has no completed selected-runtime qualification record.
Its arithmetic summary is not an admissible result for the labeled
projection configuration. Do not use it as the new Ollama baseline or
promote its printed timing gate.

Only that owned benchmark process was terminated. Official server 1492825
and runner 1494146 were retained for the corrected run; runner logs showed
all slots idle before validation resumed. No competing benchmark was
started. The original data and log remain unchanged:

- [Rejected raw attempt](2026-10-25-maxwell-projection-ollama-loader-rejected.jsonl)
- [Rejected run log](2026-10-25-maxwell-projection-ollama-loader-rejected.log.gz)
- [Audit rejection](2026-10-25-maxwell-projection-ollama-loader-rejected-audit.txt)

The correction routes both primary and candidate construction through
`RuntimeConfig::load_weights`, which forwards every typed choice once and
checks the loaded FFN/prefill/projection backends. Before any timed HTTP
send, the tool validates actual model configuration again. Post-request
validation requires the selected warp launches for every one-row layer
forward and records the actual backend, not only the requested label.
The fixture regression rejects a native-loaded model requested as warp
without any GPU work. No CUDA/model/numerical/default change is involved.

Existing quality evidence can remain the reference because its loader was
already correct and core inference is unchanged. Corrected HTTP requests
must still reproduce its actual 512 IDs and dispatch the selected kernels.
Keep this correction as a follow-up commit rather than rewrite the pinned
quality and valid incremental measurement history.
