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
sub require_true {
    die "missing audit predicate\n" unless @_ == 2;
    die "$_[1]\n" unless $_[0];
}
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

sub median {
    my @values = sort {$a <=> $b} @_;
    my $n = @values;
    return $n % 2 ? $values[int($n/2)] : ($values[$n/2-1]+$values[$n/2])/2;
}
sub sum { my $v = 0; $v += $_ for @_; return $v; }
sub audit_measurement {
    my ($rows) = @_;
    my $manifest = audit_manifest($rows);
    my $correctness = read_rows($root . '2026-10-28-direct-verifier-correctness.jsonl');
    audit_correctness($correctness);
    require_true(!$manifest->{correctness_only}
        && $manifest->{executable_sha256} eq $correctness->[0]{executable_sha256},
        'timing executable admitted by correctness');
    my @warmups = by_kind($rows, 'direct_verifier_warmup');
    my @trials = by_kind($rows, 'direct_verifier_trial');
    my @rejections = by_kind($rows, 'direct_verifier_rejection');
    require_true(@warmups == 2 && @trials == 10 && @rejections == 20
        && @$rows == 36, 'fixed five-pair complete coverage');
    my $previous_end = 0;
    my (@control, @candidate);
    for my $index (0..$#trials) {
        my $t = $trials[$index];
        my $r = $t->{result};
        my $is_direct = $t->{is_direct} ? 1 : 0;
        my $trial = int($index/2);
        require_true($t->{phase} eq 'screen' && $t->{trial} == $trial
            && $is_direct == (($trial % 2) ? 1-$index%2 : $index%2)
            && ($r->{is_direct} ? 1 : 0) == $is_direct, 'alternating paired order');
        require_true($r->{started_unix_us} >= $previous_end
            && $r->{completed_unix_us} > $r->{started_unix_us}, 'nonoverlapping requests');
        $previous_end = $r->{completed_unix_us};
        for my $gpu ($t->{gpu_before}, $t->{gpu_after}) {
            require_true(scalar($gpu->{device} =~ /^GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB, [^,]+, P0, [^,]+, 1113 MHz, 3004 MHz,/
                && $gpu->{processes} =~ /^\d+, target\/release\/examples\/q8_execution_bench, [^\n]+$/),
                'isolated M40 identity/clocks');
        }
        require_true(same($r->{generated_token_ids}, $expected_ids)
            && $r->{generated_tokens} == 512 && $r->{prompt_tokens} == 65
            && $r->{target_input_rows} == 511 && $r->{stop_reason} eq 'length'
            && $r->{committed_len} == 576 && same($r->{layer_lengths}, [(576)x36]),
            'timed actual output and dense lengths');
        require_true(@{$r->{batch_ms}} == ($is_direct ? 127 : 0)
            && @{$r->{serial_step_ms}} == ($is_direct ? 3 : 511), 'all batches and tails');
        for my $entry (
            ['ffn_q8_joint_four_rows_128',$is_direct ? 4572 : 0],
            ['ffn_q8_joint_f32x4_shared_scales_128',$is_direct ? 108 : 18396],
            ['qkv_f32xq8_0_gguf_packed_warp',18396],
            ['gemm_f32xq8_0_gguf_coltile4_packed_warp',18396],
            ['gemm_f32xq8_separate_f32',18907]) {
            require_true(($r->{target_dispatch}{$entry->[0]} // 0) == $entry->[1],
                'timed qualified target dispatch');
        }
        my $c = $r->{target_counters};
        require_true($c->{cublas_calls} == 0 && $c->{device_allocs} == 0
            && $c->{device_frees} == 0 && $c->{h2d_copies} == 511
            && $c->{d2h_copies} == 1022 && $c->{stream_syncs} == 1022
            && $c->{stream_waits} == 0, 'charged target resources/dependencies');
        require_true($r->{total_counters}{cublas_calls} == 252
            && $r->{total_counters}{device_allocs} == $r->{total_counters}{device_frees}
            && $r->{total_counters}{stream_waits} == 324, 'shared prefill/cleanup costs');
        require_true($r->{session_scratch_bytes} == ($is_direct ? 1521188 : 689668)
            && $r->{tracked_after_setup}-$r->{tracked_before} == $r->{session_scratch_bytes}
            && $r->{tracked_after_teardown} == $r->{tracked_before}, 'session ownership');
        my $phases = sum(map { $r->{$_} } qw(session_setup_ms prefill_ms decode_ms session_teardown_ms));
        require_true($r->{total_ms} >= $phases && $r->{total_ms}-$phases < 1
            && abs($r->{decode_ms}-sum(@{$r->{batch_ms}}, @{$r->{serial_step_ms}})
                -$r->{decode_other_ms}) < 1e-7
            && abs($r->{resident_total_tps}-512000/$r->{total_ms}) < 1e-9,
            'full elapsed intervals and bookkeeping retained');
        push @{$is_direct ? \@candidate : \@control}, $r->{resident_total_tps};
    }
    for my $accepted (0..3) {
        my @cycles = grep { $_->{result}{accepted_prefix} == $accepted } @rejections;
        require_true(@cycles == 5, 'five rejection observations per prefix');
        for my $t (@cycles) {
            my $r = $t->{result};
            require_true($r->{has_exact_correction} && $r->{useful_progress} == $accepted+1
                && $r->{next_token} == $expected_ids->[$accepted+1]
                && $r->{counters}{cublas_calls} == 0
                && $r->{counters}{device_allocs} == 0 && $r->{counters}{device_frees} == 0
                && $r->{elapsed_ms} >= sum(map {$r->{$_}} qw(target_ms rollback_ms correction_ms)),
                'target/rollback/correction charged');
        }
    }
    my $summary = only_kind($rows, 'direct_verifier_summary');
    my $c = $summary->{comparison};
    my @pairs = map { {control_tps=>$control[$_], candidate_tps=>$candidate[$_]} } 0..4;
    require_true(same($summary->{pairs}, \@pairs) && $c->{pairs} == 5
        && abs($c->{control}{median}-median(@control)) < 1e-9
        && abs($c->{candidate}{median}-median(@candidate)) < 1e-9
        && abs($c->{gain}-(median(@candidate)/median(@control)-1)) < 1e-12,
        'summary uses actual paired complete throughput');
    my @variations;
    for my $values (\@control, \@candidate) {
        my $m = median(@$values);
        push @variations, median(map { abs($_-$m) } @$values)/$m;
    }
    my $variation = $variations[0] > $variations[1] ? $variations[0] : $variations[1];
    my $threshold = 2*$variation > .03 ? 2*$variation : .03;
    require_true($c->{promotion_threshold} == $threshold && $c->{bootstrap_resamples} == 10000
        && $c->{bootstrap_seed} == 0 && !$c->{needs_variation_extension}
        && !$c->{needs_fresh_confirmation} && $variation <= .05
        && $c->{gain}-$threshold > .01 && $c->{bootstrap_gain_ci95}[0] > 0
        && $c->{passes_promotion_gate}, 'unchanged statistical gate');
    # Exact empirical paired resampling is an independent check of the recorded
    # seeded Monte Carlo interval, not a replacement for its acceptance rule.
    my @gains;
    for my $sample (0..3124) {
        my (@a,@b);
        my $code = $sample;
        for (1..5) { my $i = $code % 5; $code = int($code/5);
            push @a,$control[$i]; push @b,$candidate[$i]; }
        push @gains,median(@b)/median(@a)-1;
    }
    @gains = sort {$a <=> $b} @gains;
    require_true($gains[78] > 0
        && abs($gains[78]-$c->{bootstrap_gain_ci95}[0]) < .001
        && abs($gains[3045]-$c->{bootstrap_gain_ci95}[1]) < .001,
        'independent exact paired interval');
    require_true(only_kind($rows, 'direct_verifier_gate')->{passes_resident_gate},
        'recorded decision agrees');
    return {is_passed=>JSON::PP::true, pairs=>5, warmups=>2, rejection_cycles=>20,
        revision=>$manifest->{revision}, executable_sha256=>$manifest->{executable_sha256},
        gain=>$c->{gain}, paired_ci95=>$c->{bootstrap_gain_ci95},
        exact_empirical_paired_ci95=>[$gains[78],$gains[3045]],
        complete_request_counters_checked=>JSON::PP::true, device_bytes_after=>0,
        is_http_claim=>JSON::PP::false};
}

my $path = shift @ARGV or die "usage: $0 report.jsonl [--measurement] [--self-test]\n";
my $rows = read_rows($path);
my $is_measurement = grep { $_ eq '--measurement' } @ARGV;
my $audit = $is_measurement ? \&audit_measurement : \&audit_correctness;
my $result = $audit->($rows);
if (grep { $_ eq '--self-test' } @ARGV) {
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
    if ($is_measurement) {
        @mutations = (
            sub { pop @{$_[0]}; },
            sub { $_[0][3]{result}{generated_token_ids}[0] = 0; },
            sub { pop @{$_[0][4]{result}{batch_ms}}; },
            sub { $_[0][3]{result}{target_counters}{cublas_calls} = 1; },
            sub { $_[0][3]{result}{total_ms} = 1; },
            sub { $_[0][3]{result}{tracked_after_teardown} += 4; },
            sub { $_[0][3]{gpu_after}{processes} .= "\n2, other, 1 MiB"; },
            sub { $_[0][0]{executable_sha256} = '0' x 64; },
            sub { $_[0][3]{phase} = 'confirmation'; },
            sub { $_[0][3]{result}{layer_lengths}[0] = 575; },
        );
    }
    my $mutation_index = 0;
    for my $mutate (@mutations) {
        my $changed = $json->decode($json->encode($rows));
        $mutate->($changed);
        my $has_passed = eval { $audit->($changed); 1 };
        require_true(!$has_passed, "negative control $mutation_index unexpectedly passed");
        $mutation_index++;
    }
    $result->{rejected_corruptions} = scalar @mutations;
}
print $json->pretty->encode($result);
