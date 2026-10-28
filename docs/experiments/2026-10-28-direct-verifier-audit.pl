#!/usr/bin/env perl
use strict;
use warnings;
use JSON::PP;
use Digest::SHA qw(sha256_hex);

my $json = JSON::PP->new->canonical;
sub read_rows {
    my ($path) = @_;
    open my $fh, '<', $path or die "$path: $!";
    return [map { $json->decode($_) } <$fh>];
}
sub same { return $json->encode($_[0]) eq $json->encode($_[1]); }
sub require_true { die "$_[1]\n" unless $_[0]; }
sub by_kind { my ($rows, $kind) = @_; return grep { $_->{kind} eq $kind } @$rows; }
sub only_kind {
    my ($rows, $kind) = @_;
    my @found = by_kind($rows, $kind);
    require_true(@found == 1, "expected one $kind");
    return $found[0];
}

my $root = 'docs/experiments/';
sub read_json {
    my ($path) = @_;
    open my $fh, '<', $path or die "$path: $!";
    local $/;
    return $json->decode(<$fh>);
}
my $reference = read_json($root . '2026-10-27-compound-reference.json');
my $quality = read_rows($root . '2026-10-27-maxwell-compound-runtime-quality.jsonl');
my $ownership = only_kind(read_rows($root . '2026-10-27-joint-only-final-quality.jsonl'),
    'joint_only_ownership');
my ($browser) = grep { $_->{kind} eq 'q8_runtime_browser_generation'
    && $_->{label} eq 'candidate' } @$quality;
my @tasks = by_kind($quality, 'model_quality_task');
my $expected_ids = $browser->{generation}{generated_token_ids};

sub audit_manifest {
    my ($rows) = @_;
    my $manifest = only_kind($rows, 'direct_verifier_manifest');
    my $complete = only_kind($rows, 'direct_verifier_complete');
    require_true($manifest->{build_profile} eq 'release' && $manifest->{worktree} eq ''
        && $manifest->{revision} =~ /^[0-9a-f]{40}$/
        && $manifest->{executable_sha256} =~ /^[0-9a-f]{64}$/, 'clean release identity');
    require_true(same($manifest->{environment}, $reference->{configuration}{environment}),
        'frozen environment');
    require_true($manifest->{model_sha256} eq $reference->{model}{sha256}
        && $manifest->{quality_sha256} eq $reference->{artifact_sha256}{
            $root . '2026-10-27-maxwell-compound-runtime-quality.jsonl'}, 'frozen artifacts');
    require_true($manifest->{source_bytes} == 3279519744
        && $manifest->{dense_kv_bytes} == 37748736
        && $manifest->{prepared}{resident_bytes} == 1192935424
        && $manifest->{joint}{resident_bytes} == 1724645376, 'resident accounting');
    require_true(same($manifest->{logical_sources}, $ownership->{candidate}{logical_sources})
        && same($manifest->{prepared}{tensors}, $ownership->{candidate}{resident_split}{tensors})
        && same($manifest->{joint}{pairs}, $ownership->{candidate}{joint}{pairs}),
        '109 logical / 37 split / 36 joint identities');
    require_true(same($manifest->{joint_payload_hashes}, $ownership->{candidate_payload_hashes})
        && same($manifest->{joint_payload_hashes}, $complete->{joint_payload_hashes}),
        'actual joint payloads');
    my $payloads = $manifest->{prepared_payload_hashes};
    require_true(@$payloads == 37
        && same($payloads, $complete->{prepared_payload_hashes}), 'actual prepared payloads');
    require_true(same([map { $_->{key} } @$payloads],
        [map { $_->{key} } @{$manifest->{prepared}{tensors}}]), 'prepared hash identities');
    for my $payload (@$payloads) {
        require_true($payload->{sha256} =~ /^[0-9a-f]{64}$/, 'prepared payload digest');
    }
    require_true($complete->{is_complete} && !$complete->{is_runtime_claim}
        && $complete->{device_bytes_after} == 0, 'completion/teardown');
    return $manifest;
}

sub audit_correctness {
    my ($rows) = @_;
    my $manifest = audit_manifest($rows);
    require_true(@$rows == 36 && $manifest->{correctness_only}
        && $rows->[-1]{is_correctness_passed}, 'complete correctness report');
    my $b = only_kind($rows, 'direct_verifier_browser_correctness');
    require_true(same($b->{actual_token_ids}, $expected_ids)
        && $b->{generated_tokens} == 512 && $b->{stop_reason} eq 'length', 'actual browser IDs');
    require_true($b->{has_exact_valid_kv} && $b->{valid_kv_words} == 10616832
        && $b->{valid_kv_sha256} eq
        '63dd40de8aae9f6b8af712d91a6142d4cd3f17b549b8f50d42196aa924e70a36', 'valid dense KV');
    my $comparisons = $b->{full_logit_comparisons};
    require_true(@$comparisons == 511, 'all full-vocabulary comparisons');
    for my $i (0..510) {
        my $c = $comparisons->[$i];
        require_true($c->{input_row} == $i && $c->{expected_next} == $expected_ids->[$i+1]
            && $c->{actual_next} == $expected_ids->[$i+1]
            && $c->{comparison}{values} == 151936
            && $c->{comparison}{has_tolerance_passed}, 'full-logit behavior');
    }
    require_true($b->{target_counters}{cublas_calls} == 0
        && $b->{target_counters}{device_allocs} == 0
        && $b->{target_counters}{device_frees} == 0, 'no SGEMM/target allocations');
    for my $entry (
        ['ffn_q8_joint_four_rows_128',4572],
        ['ffn_q8_joint_f32x4_shared_scales_128',108],
        ['qkv_f32xq8_0_gguf_packed_warp',18396],
        ['gemm_f32xq8_0_gguf_coltile4_packed_warp',18396],
        ['gemm_f32xq8_separate_f32',18907]) {
        require_true($b->{target_dispatch}{$entry->[0]} == $entry->[1], 'qualified dispatch');
    }
    my @transactions = by_kind($rows, 'direct_verifier_transaction');
    require_true(@transactions == 8, 'all transactions');
    for my $partial (0,1) {
        for my $accepted (0..3) {
            my @match = grep { ($_->{is_forced_partial} ? 1 : 0) == $partial
                && $_->{accepted} == $accepted } @transactions;
            require_true(@match == 1 && $match[0]{has_exact_retained_kv}
                && $match[0]{has_cleared_suffix_before_correction}
                && $match[0]{has_exact_correction}, 'transaction retained/suffix/correction');
        }
    }
    my $invalid = only_kind($rows, 'direct_verifier_invalid_cases');
    for my $field (qw(has_drop_rollback has_rejected_alias has_rejected_capacity
        has_rejected_invalid_width_and_id has_rejected_unverified_commit has_stop_cap)) {
        require_true($invalid->{$field}, "invalid case $field");
    }
    my @task_rows = by_kind($rows, 'direct_verifier_quality');
    require_true(@task_rows == 24, 'twelve task pairs');
    for my $task (@tasks) {
        for my $direct (0,1) {
            my @match = grep { $_->{id} eq $task->{id}
                && ($_->{is_direct} ? 1 : 0) == $direct } @task_rows;
            require_true(@match == 1 && $match[0]{has_identical_generation}
                && same($match[0]{actual_token_ids}, $task->{candidate}{generated_token_ids})
                && $match[0]{output} eq $task->{candidate}{output}
                && $match[0]{stop_reason} eq $task->{candidate}{metrics}{stop_reason},
                'frozen task behavior');
        }
    }
    return {is_passed => JSON::PP::true, records => scalar @$rows,
        revision => $manifest->{revision}, executable_sha256 => $manifest->{executable_sha256},
        actual_ids => 512, full_vocabulary_comparisons => 511, valid_kv_words => 10616832,
        transaction_cases => 8, frozen_task_pairs => 12, joint_hashes => 36,
        prepared_hashes => 37, device_bytes_after => 0};
}

my $path = shift @ARGV or die "usage: $0 correctness.jsonl [--self-test]\n";
my $rows = read_rows($path);
my $result = audit_correctness($rows);
if (@ARGV && $ARGV[0] eq '--self-test') {
    # Missing work or altered observable results must fail evidence admission.
    my @mutations = (
        sub { pop @{$_[0]}; },
        sub { $_[0][1]{actual_token_ids}[0] = 0; },
        sub { $_[0][1]{full_logit_comparisons}[0]{comparison}{has_tolerance_passed} = JSON::PP::false; },
        sub { $_[0][1]{has_exact_valid_kv} = JSON::PP::false; },
        sub { $_[0][2]{has_cleared_suffix_before_correction} = JSON::PP::false; },
        sub { $_[0][0]{joint_payload_hashes}[0][1] = '0' x 64; },
        sub { $_[0][0]{prepared_payload_hashes}[0]{sha256} = '0' x 64; },
        sub { $_[0][-1]{device_bytes_after} = 4; },
        sub { $_[0][0]{environment}{M40LLM_DECODE_GRAPH} = '1'; },
        sub { $_[0][1]{target_counters}{cublas_calls} = 1; },
        sub { $_[0][11]{output} = 'wrong'; },
        sub { $_[0][10]{has_rejected_alias} = JSON::PP::false; },
    );
    for my $mutate (@mutations) {
        my $changed = $json->decode($json->encode($rows));
        $mutate->($changed);
        my $has_passed = eval { audit_correctness($changed); 1 };
        require_true(!$has_passed, 'negative control unexpectedly passed');
    }
    $result->{rejected_corruptions} = scalar @mutations;
}
print $json->pretty->encode($result);
