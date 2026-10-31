# Fixed-source audit of retained GPU projection costs for scheduling sensitivity.
# Input: the complete, independently hash-checked ShiftAdd JSONL, slurped with -s.
def require($condition; $message):
  if $condition then . else error($message) end;
def key: [.layer_id, .role, .rows];
def is_positive: type == "number" and isfinite and . > 0;
def distribution:
  sort | {minimum:.[0], median:.[length / 2 | floor], maximum:.[-1], samples:length};
def shape($role): if $role == "down" then [11008,2048] else [2048,11008] end;

map(select(.kind == "shiftadd_manifest")) as $manifests |
require(($manifests | length) == 1; "expected one source manifest") |
$manifests[0] as $manifest |
require($manifest.revision == "2c8e98502442dc7a646a5a22d65dff25a83b1d75"
  and $manifest.executable_sha256 == "cf8dba7786991f862e7cf0c707196b8774c06fd9cd2b6636080eb8136352a291"
  and $manifest.model_sha256 == "4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
  and $manifest.fixture_sha256 == "eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
  and $manifest.capture_report_sha256 == "d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
  and $manifest.rows == [1,2,4,8] and $manifest.iterations == 100;
  "source identity or measurement configuration differs") |
map(select(.kind == "shiftadd_complete")) as $complete |
require(($complete | length) == 1 and $complete[0].is_complete == true
  and $complete[0].device_bytes_before == 0 and $complete[0].device_bytes_after == 0;
  "source incomplete or tracked storage not retired") |
map(select(.kind == "shiftadd_weights")) as $weights |
map(select(.kind == "shiftadd_validation")) as $validation |
map(select(.kind == "shiftadd_pair" and (.case.control == "q8" or .case.control == "prepared_q8"))) as $pairs |
[([0,17,35][]) as $layer | (["gate","up","down"][]) as $role |
 ([1,2,4,8][]) as $batch | [$layer,$role,$batch]] | sort as $expected |
require(($weights | length) == 9 and ($validation | length) == 72 and ($pairs | length) == 360;
  "incomplete cost, weight or validation grid") |
require(($validation | map(.case | key) | unique | sort) == $expected;
  "validation grid differs") |
[
  $expected[] as $key |
  ($weights | map(select([.layer_id,.role] == $key[:2]))) as $source |
  ($validation | map(select((.case | key) == $key))) as $checks |
  require(($source | length) == 1 and ($checks | length) == 2
    and ($checks | map(.phase) | sort) == ["after","before"];
    "duplicate or missing source/validation") |
  $source[0].report as $weight |
  ($checks | map(select(.phase == "before")) | .[0]) as $before |
  require($weight.source.shape == shape($key[1]) and $weight.source.dtype == "Q8_0"
    and $weight.bytes.q8 == 23953408 and $weight.bytes.prepared_q8 == 23953408
    and ($weight.source.source_sha256 | test("^[a-f0-9]{64}$"));
    "weight identity differs") |
  require(($checks | all(.case.shape == shape($key[1])
    and .checks.guards == "pass" and .checks.immutable == "pass"))
    and ($before.preparation.input_sha256 | test("^[a-f0-9]{64}$"))
    and ($before.row_positions | length) == $key[2];
    "input identity or guard validation differs") |
  (["q8","prepared_q8"][]) as $arm |
  require(($checks | all((.checks.checks | map(select(.arm == $arm))) as $arithmetic |
    ($arithmetic | length) == 1 and $arithmetic[0].arithmetic.values == ($key[2] * shape($key[1])[1])
    and ($arithmetic[0].arithmetic.maximum_bound_fraction | type == "number" and isfinite and . >= 0 and . <= 1)));
    "arithmetic validation failed") |
  ($pairs | map(select((.case | key) == $key and .case.control == $arm)) | sort_by(.pair)) as $trials |
  require(($trials | map(.pair)) == [0,1,2,3,4]
    and ($trials | all(.phase == "screen" and .case.shape == shape($key[1])
      and .case.stage == "complete_projection"
      and .order == (if .pair % 2 == 0 then [0,1] else [1,0] end)));
    "paired series differs") |
  [$trials[] | .times | map(select(.arm == $arm)) |
    require(length == 1; "missing/duplicate measured arm") | .[0]] as $times |
  require(($times | all(.iterations == 100 and .allocation_free_enqueue == true
    and (.wall_ms | is_positive) and (.gpu_ms | is_positive) and (.submission_ms | is_positive)
    and .completed_unix_us > .started_unix_us
    and .launches == (if $arm == "q8" then 100 * $key[2] else 100 end)
    and .operation == (if $arm == "q8" then "gemm_f32xq8_0_gguf_f32_decode"
      else "gemm_f32xq8_separate_packed_activation" end)));
    "invalid measured timing or dispatch") |
  {layer:$key[0], family:$key[1], batch:$key[2], arm:$arm,
   source:$weight.source, q8_payload_bytes:$weight.bytes.q8,
   input_sha256:$before.preparation.input_sha256,
   row_positions:$before.row_positions,
   operation:$times[0].operation, iterations_per_trial:100,
   wall_ns_per_projection:($times | map(.wall_ms * 1000000) | distribution),
   gpu_ns_per_projection:($times | map(.gpu_ms * 1000000) | distribution),
   submission_ns_per_projection:($times | map(.submission_ms * 1000000) | distribution),
   raw_observations:$times}
] as $costs |
{schema_version:1, source_manifest:$manifest,
 source_compressed_sha256:"e2d107a2457ba4089e0a5e5049fcefad98a68e15d0a99ae767f43a81111ee610",
 source_digest_scope:"expected digest; verify compressed bytes separately before this audit",
 is_new_hardware_measurement:false, is_full_decode_profile:false,
 cost_scope:"isolated completed projections; 100-operation intervals amortize final completion; no transfers or CPU decoding",
 numerical_scope:"each implementation passed its own FP64 arithmetic bound; CPU/GPU bit identity unqualified",
 costs:$costs}
