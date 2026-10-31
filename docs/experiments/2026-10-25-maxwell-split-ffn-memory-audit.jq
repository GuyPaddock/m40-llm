def require($condition;$message): if $condition then . else error($message) end;
def median: sort | .[length/2-1] as $a | .[length/2] as $b | ($a+$b)/2;
map(select(.kind=="ffn_ablation_preparation")) as $setup |
map(select(.kind=="ffn_ablation_pair")) as $pairs |
map(select(.kind=="ffn_ablation_manifest"))[0] as $manifest |
require(($setup|length)==6 and [$setup[]|[.layer_id,.rows]]==[[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]];
  "missing preparation cases") |
require(all($setup[]; . as $record | .preparation as $p |
  .scratch_capacity_bytes==(if .rows==1 then 201152 else 803264 end) and
  $p.layout=="SeparateScalesValues" and $p.additional_prepared_bytes==47906816 and
  $p.retained_host_prepared_bytes==47906816 and $p.estimated_all_layers_extra_bytes==1724645376 and
  $p.packing_ms>=0 and $p.allocation_upload_ms>=0 and
  $p.tracked_bytes_before==4472455168+$record.scratch_capacity_bytes and
  $p.tracked_bytes_after==$p.tracked_bytes_before+47906816 and
  $p.planes==[range(2)|{planes:[{bytes:1409024,kind:"Scales"},{bytes:22544384,kind:"Values"}],
    resident_bytes:23953408,source_bytes:23953408,tile:null}] and
  [$p.keys[].source.tensor_name]==["blk.\(.layer_id).ffn_gate.weight","blk.\(.layer_id).ffn_up.weight"] and
  all($p.keys[]; .layout=="SeparateScalesValues" and .source.dtype=="Q8_0" and
    .source.shape=={input_dim:2048,output_dim:11008} and
    .source.model_sha256==$manifest.prepared_weights.tensors[0].key.source.model_sha256));
  "incorrect preparation layout, identity, byte accounting or setup timing") |
require(($pairs|length)==30 and all($pairs[]; . as $pair |
  ($setup[]|select(.layer_id==$pair.layer_id and .rows==$pair.rows)) as $s |
  .additional_prepared_bytes==47906816 and .tracked_device_bytes==$s.preparation.tracked_bytes_after and
  .scratch_capacity_bytes==$s.scratch_capacity_bytes);"prepared allocations changed during timing") |
{is_complete:true,preparation_cases:6,additional_device_bytes_per_case:47906816,
 retained_host_bytes_per_case:47906816,scale_bytes_per_pair:2818048,value_bytes_per_pair:45088768,
 observed_tracked_bytes_by_rows:($pairs|group_by(.rows)|map({rows:.[0].rows,bytes:map(.tracked_device_bytes)|unique})),
 median_packing_ms:($setup|map(.preparation.packing_ms)|median),
 median_allocation_upload_ms:($setup|map(.preparation.allocation_upload_ms)|median),
 estimated_all_layers_extra_bytes:1724645376,
 estimate_scope:"36 identical gate/up pair shapes; not measured whole-model runtime residency"}
