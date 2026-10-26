# jq -s --slurpfile fixture M02 --slurpfile capture CAPTURE -f THIS REPORT
def require($ok;$why): if $ok then . else error($why) end;
def close($a;$b): (($a-$b)|fabs)<1e-10;
def entropy: add as $sum|map(select(.>0)|(. / $sum) as $p| -($p*($p|log)/(2|log)))|add;
def prefix_cost:
  map(select(.>0))|
  if length==1 then .[0] else
    {weights:sort,cost:0}|until((.weights|length)==1;
      (.weights[0]+.weights[1]) as $sum|
      .weights=((.weights[2:]+[$sum])|sort)|.cost+=$sum)|.cost
  end;
. as $records
| map(select(.kind=="ffn_payload_entropy_manifest")) as $manifest
| $records|map(select(.kind=="ffn_payload_entropy_complete")) as $complete
| $records|map(select(.kind=="ffn_payload_entropy_tensor")) as $rows
| require(length==8 and ($manifest|length)==1 and ($complete|length)==1 and ($rows|length)==6;"complete coverage")
| $manifest[0] as $m|$complete[0] as $done
| require($m.schema_version==1 and $m.build_profile=="release" and $m.worktree==""
    and $m.revision=="fc7f377b3bf5dd55fddd76d01c157e5f507542a9"
    and $m.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $m.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
    and $m.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
    and $m.execution_configuration==$capture[0].execution_configuration
    and $m.source_layers==[0,17,35] and $m.families==["gate","up"] and $m.chunk_sizes==[32,256,1024]
    and $m.minimum_stored_byte_saving==0.10 and ($m.has_created_cuda_context|not)
    and $m.tracked_device_bytes==0;"clean source, frozen identities and offline scope")
| require(([$rows[]|[.layer_id,.family]]|sort)==[[0,"gate"],[0,"up"],[17,"gate"],[17,"up"],[35,"gate"],[35,"up"]]
    and ([$rows[]|.absolute_file_offset-.source.byte_offset]|unique|length)==1;"tensor multiplicity/absolute offsets")
| require(all($rows[];. as $row|.statistics as $s|
    .source==([$fixture[0].layers[]|select(.source.layer_id==$row.layer_id)|.source[$row.family]][0])
    and .source.dtype=="Q8_0" and .source.shape==[2048,11008]
    and .absolute_file_offset>.source.byte_offset and .raw_tensor_buffer_bytes==23953408
    and .read_and_hash_ms>0 and .analysis_ms>0 and (.has_created_cuda_context|not)
    and $s.source_bytes==23953408 and $s.payload_symbols==22544384 and $s.scale_bytes==1409024
    and ($s.payload_histogram|length)==256 and ($s.huffman_code_lengths|length)==256
    and all($s.payload_histogram[];.>=0 and .==floor) and ($s.payload_histogram|add)==$s.payload_symbols
    and $s.zero_payload_symbols==$s.payload_histogram[0]
    and $s.negative_payload_symbols==($s.payload_histogram[128:]|add)
    and all(range(0;256);$s.huffman_code_lengths[.] as $len|
      $len==($len|floor) and $len>=0 and $len<=255 and (($len>0)==($s.payload_histogram[.]>0)))
    and close($s.zero_order_entropy_bits_per_symbol;($s.payload_histogram|entropy))
    and $s.zero_order_ideal_total_bytes==(($s.zero_order_entropy_bits_per_symbol*$s.payload_symbols/8)|ceil)+$s.scale_bytes
    and $s.maximum_code_length==($s.huffman_code_lengths|max)
    and $s.huffman_payload_bits==([range(0;256)|$s.payload_histogram[.]*$s.huffman_code_lengths[.]]|add)
    and $s.huffman_payload_bits==($s.payload_histogram|prefix_cost)
    and ([$s.huffman_code_lengths[]|select(.>0)|pow(2; -.)]|add)<=1.000000000001
    and ($s.has_measured_speedup|not) and ($s.has_decoder|not)
    and [$s.indexed_size_models[].chunk_payload_symbols]==[32,256,1024]
    and all($s.indexed_size_models[];
      .chunks==(($s.payload_symbols/.chunk_payload_symbols)|ceil)
      and .chunk_alignment_bytes==16 and .offset_bytes_per_chunk==4 and .index_bytes==.chunks*4
      and .code_length_table_bytes==256 and .scale_bytes==$s.scale_bytes
      and .aligned_huffman_payload_bytes%16==0
      and .aligned_huffman_payload_bytes>=(($s.huffman_payload_bits/128)|ceil)*16
      and .aligned_huffman_payload_bytes<=(($s.huffman_payload_bits/128)|ceil)*16+.chunks*16
      and .idealized_stored_bytes==.aligned_huffman_payload_bytes+.index_bytes+.scale_bytes+256
      and close(.stored_byte_saving;1-.idealized_stored_bytes/$s.source_bytes)
      and (.includes_decoder_storage|not) and (.has_materialized_encoded_stream|not)));"histogram, optimal code cost and indexed accounting")
| require($done.is_complete and $done.tensors==6 and $done.minimum_stored_byte_saving==0.10
    and $done.tracked_device_bytes_before==0 and $done.tracked_device_bytes_after==0
    and ($done.runtime_integration_allowed|not) and ($done.is_e2e_claim|not)
    and $done.size_model_eligible_at_all_tensors==[range(0;3)|. as $i|all($rows[];.statistics.indexed_size_models[$i].stored_byte_saving>=0.10)];"eligibility/cleanup/no runtime claim")
| {passes_audit:true,source:$m.revision,tensors:6,has_created_cuda_context:false,
    size_model_eligible_at_all_tensors:$done.size_model_eligible_at_all_tensors,
    tensor_sizes:[$rows[]|{layer_id,family,source_bytes:.statistics.source_bytes,
      entropy_bits_per_payload_symbol:.statistics.zero_order_entropy_bits_per_symbol,
      zero_order_ideal_byte_saving:(1-.statistics.zero_order_ideal_total_bytes/.statistics.source_bytes),
      huffman_payload_bits:.statistics.huffman_payload_bits,indexed_size_models:.statistics.indexed_size_models}],
    runtime_integration_allowed:false,is_e2e_claim:false,
    limitations:"zero-order byte models only; no encoder, decoder cost/storage, GPU bandwidth or throughput measurement"}
