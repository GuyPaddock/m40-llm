#!/usr/bin/env bash
# Reject corrupted observable reports without editing measured evidence.
set -euo pipefail
quality=${1:?quality JSONL path}
http=${2:?HTTP JSONL path}
base=docs/experiments/2026-10-27-joint-only
frozen=docs/experiments/2026-10-27-maxwell-compound-runtime-quality.jsonl
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
quality_sha256=$(sha256sum "$quality" | cut -d ' ' -f 1)

audit_quality() {
    jq -s --slurpfile frozen "$frozen" -f "$base-quality-audit.jq" "$1"
}

audit_http() {
    jq -s --slurpfile quality "$quality" --arg quality_sha256 "$quality_sha256" \
        -f "$base-http-audit.jq" "$1"
}

# Each negative control must change the audit result from success to failure.
reject() {
    local kind=$1 label=$2 mutation=$3 input
    if [[ "$kind" == quality ]]; then input=$quality; else input=$http; fi
    jq -c "$mutation" "$input" > "$temporary/corrupt.jsonl"
    if "audit_$kind" "$temporary/corrupt.jsonl" > "$temporary/result" 2>&1; then
        printf 'unexpected audit success: %s\n' "$label" >&2
        exit 1
    fi
    jq -nc --arg kind "$kind" --arg label "$label" \
        '{kind:$kind,control:$label,is_rejected:true}'
}

audit_quality "$quality" > "$temporary/quality-valid.json"
audit_http "$http" > "$temporary/http-valid.json"
reject quality dirty-worktree 'if .kind=="joint_only_manifest" then .worktree=" M changed" else . end'
reject quality wrong-model 'if .kind=="joint_only_manifest" then .model_sha256="wrong" else . end'
reject quality missing-logical 'if .kind=="joint_only_ownership" then .candidate.logical_sources|=.[1:] else . end'
reject quality duplicate-down 'if .kind=="joint_only_ownership" then .candidate.resident_split.tensors[1]=.candidate.resident_split.tensors[0] else . end'
reject quality offset-mismatch 'if .kind=="joint_only_ownership" then .candidate.resident_split.tensors[0].key.source.byte_offset+=32 else . end'
reject quality duplicate-joint 'if .kind=="joint_only_ownership" then .candidate.joint.pairs[1]=.candidate.joint.pairs[0] else . end'
reject quality reversed-joint 'if .kind=="joint_only_ownership" then .candidate.joint.pairs[0]|={gate:.up,up:.gate} else . end'
reject quality changed-payload 'if .kind=="joint_only_ownership" then .candidate_payload_hashes[0][1]="wrong" else . end'
reject quality understated-bytes 'if .kind=="joint_only_ownership" then .candidate.resident_split.resident_bytes-=1 else . end'
reject quality allocated-split 'if .kind=="joint_only_ownership" then .skipped_device_allocations=0 else . end'
reject quality changed-ids 'if .kind=="joint_only_generation" then .generated_ids[0]+=1 else . end'
reject quality changed-logits 'if .kind=="joint_only_logits" then .candidate_sha256="wrong" else . end'
reject quality changed-task 'if .kind=="joint_only_task" then .candidate_output="wrong" else . end'
reject quality incomplete-cleanup 'select(.kind!="joint_only_quality_complete")'
reject http wrong-executable 'if .kind=="joint_only_manifest" then .executable_sha256="wrong" else . end'
reject http unbound-quality 'if .kind=="joint_only_http_binding" then .quality_sha256="wrong" else . end'
reject http wrong-owner 'if .kind=="joint_only_http_binding" then .candidate.ownership="retain-split" else . end'
reject http missing-trial 'select(.kind!="joint_only_http_trial" or .pair!=0 or .arm!=1)'
reject http changed-output 'if .kind=="joint_only_http_trial" then .row.output="wrong" else . end'
reject http fewer-tokens 'if .kind=="joint_only_http_trial" then .row.metrics.generated_tokens=511 else . end'
reject http natural-stop 'if .kind=="joint_only_http_trial" then .row.metrics.stop_reason="stop" else . end'
reject http changed-dispatch 'if .kind=="joint_only_http_trial" then .row.dispatch={} else . end'
reject http wrong-clocks 'if .kind=="joint_only_http_trial" then .gpu_before.clocks="wrong" else . end'
reject http competing-process 'if .kind=="joint_only_http_trial" then .gpu_before.processes+="\nother GPU user" else . end'
reject http changed-rate 'if .kind=="joint_only_http_trial" then .row.http_tok_per_sec+=1 else . end'
reject http changed-summary 'if .kind=="joint_only_http_summary" then .comparison.candidate.median+=1 else . end'
reject http adverse-confidence 'if .kind=="joint_only_http_summary" then .comparison.bootstrap_gain_ci95[0]= -0.04 else . end'
reject http incomplete-http 'select(.kind!="joint_only_http_complete")'
