# Recompute fixed-coordinate information and storage costs from actual counts.
def require($condition;$why): if $condition then . else error($why) end;
def entropy($total): map(select(.>0)|(. / $total) as $p| -$p*($p|log)/(2|log))|add // 0;
. as $input
| [$original[]|select(.kind=="ffn_payload_entropy_tensor" and .layer_id==$input.layer_id and .family==$input.family)] as $matches
| require(($matches|length)==1;"frozen source coverage")
| $matches[0] as $frozen
| .raw.statistics as $s
| require(.source_revision=="7600c7a24b3492e74b7fb4ed24f3a178ac0cd1d3"
    and .source==$frozen.source and .absolute_file_offset==$frozen.absolute_file_offset
    and .actual_tensor_sha256==$frozen.source.source_sha256
    and .actual_model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and .raw.kind=="channel_entropy_tensor" and .raw.loaded_cuda_libraries==[]
    and .resource_exit_code==0 and .peak_rss_kib>0 and .peak_rss_kib<524288;
    "source identities, CPU-only process observation and resource bounds")
| require($s.k==2048 and $s.n==11008 and $s.source_bytes==23953408
    and $s.payload_symbols==22544384 and $s.scale_bytes==1409024
    and $s.output_symbols_per_index_chunk==256
    and $s.code_length_table_bytes==2048*256 and $s.index_bytes==2048*(11008/256|ceil)*4
    and ($s.has_encoder|not) and ($s.has_decoder|not) and ($s.has_materialized_encoded_stream|not)
    and ($s.has_measured_speedup|not) and ($s.includes_alignment_or_decoder_cost|not);
    "fixed optimistic accounting and no codec/performance claim")
| require(($s.channels|length)==2048 and ($s.channels|map(.input_channel))==[range(0;2048)]
    and all($s.channels[];(.histogram|length)==256 and (.histogram|add)==11008
      and all(.histogram[];.>=0 and .==floor))
    and $s.pooled_histogram==$frozen.statistics.payload_histogram
    and ($s.pooled_histogram|add)==22544384
    and ($s.channels|map(.histogram)|transpose|map(add))==$s.pooled_histogram;
    "all channel counts conserve the verified pooled payload")
| require((.independent|map(.input_channel)|sort)==[0,31,32,2047]
    and all(.independent[];. as $check|$s.channels[$check.input_channel].histogram==$check.histogram);
    "independent absolute-channel source recount")
| ($s.channels|map(.histogram|entropy(11008))) as $entropies
| (($entropies|add)/2048) as $conditional
| ($conditional*22544384) as $bits
| ($bits/8|ceil) as $payload_bytes
| ($payload_bytes+1409024+524288+352256) as $stored
| require(all(range(0;2048);. as $i|(($s.channels[$i].entropy_bits-$entropies[$i])|fabs)<1e-9)
    and (($s.pooled_entropy_bits-($s.pooled_histogram|entropy(22544384)))|fabs)<1e-9
    and (($s.pooled_entropy_bits-$frozen.statistics.zero_order_entropy_bits_per_symbol)|fabs)<1e-9
    and (($s.conditional_entropy_bits-$conditional)|fabs)<1e-9
    and $conditional<=$s.pooled_entropy_bits+1e-9
    and (($s.ideal_conditional_payload_bits-$bits)|fabs)<0.001
    and $s.ideal_conditional_payload_bytes==$payload_bytes
    and $s.optimistic_stored_bytes==$stored
    and (($s.optimistic_stored_byte_saving-(1-$stored/23953408))|fabs)<1e-12;
    "independently recomputed entropy, literal scales and metadata")
| {passes_audit:true,layer_id:.layer_id,family:.family,source:.source,
    pooled_entropy_bits:$s.pooled_entropy_bits,conditional_entropy_bits:$conditional,
    coordinate_information_bits:($s.pooled_entropy_bits-$conditional),
    ideal_conditional_payload_bytes:$payload_bytes,literal_scale_bytes:1409024,
    code_length_table_bytes:524288,index_bytes:352256,optimistic_stored_bytes:$stored,
    optimistic_stored_byte_saving:(1-$stored/23953408),
    ideal_saving_without_metadata:(1-($payload_bytes+1409024)/23953408),
    has_size_feasibility:(1-$stored/23953408>=0.10),
    peak_rss_kib:.peak_rss_kib,has_materialized_encoded_stream:false,has_measured_speedup:false}
