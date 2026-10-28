#!/usr/bin/env perl
# Independent report validation and finite-horizon scenario arithmetic.
use strict;
use warnings;
use JSON::PP;
use Digest::SHA qw(sha256_hex);

my $json = JSON::PP->new->utf8->canonical;
my $root = 'docs/experiments/2026-10-27-';

sub require_value { die "$_[1]\n" unless $_[0]; }
sub read_bytes {
    open my $file, '<:raw', $_[0] or die "$_[0]: $!";
    local $/;
    return <$file>;
}
sub read_rows { return [map {$json->decode($_)} grep {length} split /\n/, read_bytes($_[0])]; }
sub same { return $json->encode($_[0]) eq $json->encode($_[1]); }
sub records { my ($rows,$kind)=@_; return [grep {$_->{kind} eq $kind} @$rows]; }
sub one {
    my ($rows,$kind)=@_;
    my $found=records($rows,$kind);
    require_value(@$found==1,"record multiplicity: $kind");
    return $found->[0];
}
sub median {
    my @values=sort {$a<=>$b} @_;
    require_value(@values>0,'empty median');
    my $m=int(@values/2);
    return @values%2 ? $values[$m] : ($values[$m-1]+$values[$m])/2;
}
sub max_value { my $max=shift; for (@_) {$max=$_ if $_>$max} return $max; }
sub min_value { my $min=shift; for (@_) {$min=$_ if $_<$min} return $min; }

sub validate_gpu {
    my ($state)=@_;
    my @fields=map {s/^\s+|\s+$//gr} split /,/, $state->{device};
    require_value(@fields==9 && $fields[0] eq 'GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852'
        && $fields[3] eq 'P0' && $fields[5] =~ /^1113(?: MHz)?$/
        && $fields[6] =~ /^3004(?: MHz)?$/, 'GPU/clock binding');
    require_value($state->{processes}=~/^([0-9]+), [^\n]+$/, 'competing/missing GPU process');
    return $1;
}

sub validate_report {
    my ($rows)=@_;
    my $ref=$json->decode(read_bytes("${root}compound-reference.json"));
    my $quality=read_rows("${root}maxwell-compound-runtime-quality.jsonl");
    my ($generation)=grep {$_->{kind} eq 'q8_runtime_browser_generation' && $_->{label} eq 'candidate'} @$quality;
    my $ids=$generation->{generation}{generated_token_ids};
    my $m=one($rows,'direct_q8_cost_manifest');
    require_value($m->{reference_id} eq $ref->{reference_id} && $m->{schema_version}==1
        && $m->{worktree} eq '' && $m->{revision}=~/^[a-f0-9]{40}$/
        && $m->{executable_sha256}=~/^[a-f0-9]{64}$/
        && $m->{model_sha256} eq $ref->{model}{sha256}
        && $m->{quality_sha256} eq sha256_hex(read_bytes("${root}maxwell-compound-runtime-quality.jsonl"))
        && same($m->{environment},$ref->{configuration}{environment}), 'source/model/environment binding');
    my $o=one($rows,'direct_q8_cost_ownership');
    my $prior=one(read_rows("${root}joint-only-final-quality.jsonl"),'joint_only_ownership')->{candidate};
    require_value($o->{ownership} eq 'joint-only' && $o->{source_bytes}==3279519744
        && $o->{dense_kv_bytes}==37748736 && @{$o->{logical_sources}}==109
        && same($o->{logical_sources},$prior->{logical_sources})
        && same($o->{split}{tensors},$prior->{resident_split}{tensors})
        && same($o->{joint}{pairs},$prior->{joint}{pairs})
        && $o->{split}{resident_bytes}==1192935424 && @{$o->{split}{tensors}}==37
        && $o->{joint}{resident_bytes}==1724645376 && $o->{joint}{pair_count}==36, 'ownership identities/bytes');
    my $trials=records($rows,'direct_q8_cost_trial');
    my $rejections=records($rows,'direct_q8_cost_rejection');
    require_value(@$trials==30 && @$rejections==40, 'incomplete bounded observation');
    my (%pids,%prefixes);
    for my $record (@$trials,@$rejections) {
        my $r=$record->{row};
        my $k=$r->{width};
        require_value($k==2 || $k==4 || $k==8, 'unexpected width');
        $pids{validate_gpu($record->{gpu_before})}++;
        $pids{validate_gpu($record->{gpu_after})}++;
        $prefixes{$r->{prefix_sha256}}++;
        require_value($r->{has_exact_output} && $r->{has_cleared_suffix}
            && $r->{prefix_tokens}==65 && $r->{prefix_sha256} eq $r->{after_prefix_sha256}
            && $r->{prefix_sha256}=~/^[a-f0-9]{64}$/ && $r->{elapsed_ms}>0
            && $r->{begin_ms}>0 && $r->{commit_ms}>=0 && $r->{correction_ms}>=0
            && $r->{elapsed_ms}+1e-6 >= $r->{begin_ms}+$r->{commit_ms}+$r->{correction_ms}
            && $r->{begin_counters}{cublas_calls}==252 && $r->{counters}{cublas_calls}==252,
            'output/prefix/ordinary target work');
        my $a=$r->{accepted_prefix};
        my $progress=defined($a) ? $a+1 : $k;
        require_value($r->{useful_progress}==$progress && @{$r->{lengths}}==36
            && !(grep {$_!=65+$progress} @{$r->{lengths}}), 'committed lengths/progress');
        for my $counter (keys %{$r->{counters}}) {
            require_value($r->{counters}{$counter} == $r->{begin_counters}{$counter}
                +$r->{commit_counters}{$counter}+$r->{correction_counters}{$counter}, 'phase counter sum');
        }
        if (defined $a) {
            require_value(!$r->{is_profiled} && $r->{next_token}==$ids->[$a]
                && $r->{correction_token}==$ids->[$a+1] && $r->{correction_ms}>0
                && $r->{commit_counters}{launches}==1 && $r->{commit_counters}{stream_syncs}==3,
                'rollback/correction work');
            require_value(@{$r->{predictions}}==$k,'rejection prediction count');
            for my $position (0..$a-1) {
                require_value($r->{predictions}[$position]==$ids->[$position+1], 'accepted greedy prefix');
            }
        } else {
            require_value(same($r->{predictions},[@$ids[1..$k]]) && $r->{next_token}==$ids->[$k]
                && $r->{correction_ms}==0, 'full-acceptance actual IDs');
            if ($r->{is_profiled}) {
                my $stages=$r->{gpu_profile}{stages};
                require_value($r->{gpu_profile}{event_pool_capacity}==4096 && @$stages==829+3*$k,
                    'event pool/span coverage');
                for my $stage ('q8_prefill_dequantize','q8_prefill_sgemm') {
                    my @selected=grep {$_->{stage} eq $stage} @$stages;
                    require_value(@selected==252 && !(grep {$_->{gpu_elapsed_ms}<0} @selected),
                        'projection event coverage');
                }
                for my $layer (0..35) {
                    for my $stage ('q8_prefill_dequantize','q8_prefill_sgemm') {
                        my @selected=grep {$_->{stage} eq $stage && defined($_->{layer_id}) && $_->{layer_id}==$layer} @$stages;
                        require_value(@selected==7, 'layer event coverage');
                    }
                }
            } else {require_value(!defined($r->{gpu_profile}), 'disabled profiler emitted spans');}
        }
    }
    require_value(keys(%pids)==1 && keys(%prefixes)==1, 'process/prefix identity changed');
    for my $k (2,4,8) {
        my @t=grep {$_->{row}{width}==$k} @$trials;
        for my $pair (0..4) {
            my @p=grep {$_->{pair}==$pair} @t;
            require_value(@p==2 && !$p[0]{row}{is_profiled} != !$p[1]{row}{is_profiled}
                && ($p[0]{row}{is_profiled}?1:0)==$pair%2
                && same($p[0]{row}{counters},$p[1]{row}{counters}), 'pair schedule/counters');
        }
        my $off=median(map {$_->{row}{elapsed_ms}} grep {!$_->{row}{is_profiled}} @t);
        my $on=median(map {$_->{row}{elapsed_ms}} grep {$_->{row}{is_profiled}} @t);
        for my $flag (0,1) {
            my @values=map {$_->{row}{elapsed_ms}} grep {($_->{row}{is_profiled}?1:0)==$flag} @t;
            my $middle=median(@values);
            require_value(median(map {abs($_-$middle)} @values)/$middle<=.05,'diagnostic timing unstable');
        }
        my @summaries=grep {$_->{kind} eq 'direct_q8_cost_observer' && $_->{width}==$k} @$rows;
        require_value(@summaries==1 && $summaries[0]{is_attribution_valid}
            && abs($summaries[0]{off_ms}-$off)<1e-9 && abs($summaries[0]{on_ms}-$on)<1e-9
            && abs($summaries[0]{latency_overhead}-($on/$off-1))<1e-12
            && $summaries[0]{limit}==0.03 && $on/$off-1<=0.03, 'observer allowance/summary');
        my %positions=map {$_=>1} (0,int($k/2),$k-1);
        my @r=grep {$_->{row}{width}==$k} @$rejections;
        require_value(@r==5*keys(%positions), 'rejection count');
        for my $accepted (keys %positions) {
            my @r=grep {$_->{row}{accepted_prefix}==$accepted} @r;
            require_value(same([sort {$a<=>$b} map {$_->{trial}} @r],[0..4]), 'rejection repetitions');
        }
    }
    my $done=one($rows,'direct_q8_cost_complete');
    require_value($done->{is_diagnostic_valid} && $done->{device_bytes_after}==0
        && !$done->{is_http_claim} && !$done->{is_new_kernel_measured}, 'completion/cleanup/scope');
    return {is_audit_passed=>JSON::PP::true,revision=>$m->{revision},executable_sha256=>$m->{executable_sha256},
        full_trials=>30,rejection_cycles=>40,profiled_batches=>15,gpu_boundaries=>140,
        actual_ids_and_prefix_bits_match=>JSON::PP::true,device_bytes_after=>0};
}

# Known first candidate is not drafted. A rejection also executes correction;
# full acceptance advances K, never K+1. Preserve the old unconditional model
# separately when reading historical curves.
sub get_distribution {
    my ($k,$p,$known)=@_;
    my @out;
    for my $a ($known..$k-1) {
        push @out,[$a+1,(1-$p)*$p**($a-$known)];
    }
    push @out,[$k,$p**($k-$known)];
    return \@out;
}

# Exact finite horizon for constant-cost sensitivity scenarios. Tail rows run
# serially. Expected cycle count makes drafter budgets linear, without treating
# 511/E[progress] as an exact integer batch count.
sub get_work_counts {
    my ($k,$p,$known,$eligibility)=@_;
    $eligibility=1 unless defined $eligibility;
    my $distribution=get_distribution($k,$p,$known);
    my (@cycles,@tail);
    for my $n (0..511) {
        if ($n<$k) {$cycles[$n]=0; $tail[$n]=$n; next;}
        my ($c,$t)=($eligibility+(1-$eligibility)*$cycles[$n-1],
            (1-$eligibility)*(1+$tail[$n-1]));
        for my $item (@$distribution) {
            my ($progress,$probability)=@$item;
            $c+=$eligibility*$probability*$cycles[$n-$progress];
            $t+=$eligibility*$probability*$tail[$n-$progress];
        }
        ($cycles[$n],$tail[$n])=($c,$t);
    }
    return ($cycles[511],$tail[511]);
}

sub get_projection {
    my ($trial,$k,$p,$beta,$extra,$draft,$reject_extra,$gate,$eligibility)=@_;
    my ($cycles,$tail)=get_work_counts($k,$p,1,$eligibility);
    my $serial=$trial->{metrics}{decode_elapsed_ms}/511;
    my $fixed=$trial->{client_elapsed_ms}-$trial->{metrics}{decode_elapsed_ms};
    my $v=$k*$serial-(1-$beta)*$k*$gate+$extra;
    my $cost=$fixed+$cycles*($v+$draft+(1-$p**($k-1))*$reject_extra)+$tail*$serial;
    return {http_ms=>$cost,http_tok_per_sec=>512000/$cost,gain=>$trial->{client_elapsed_ms}/$cost-1,
        verifier_ms=>$v,expected_cycles=>$cycles,expected_tail_rows=>$tail,
        target_only_http_ms=>$cost-$cycles*$draft};
}

sub self_test {
    require_value(same(get_distribution(4,0,1),[[2,1],[3,0],[4,0],[4,0]]),'known-seed zero acceptance');
    my ($c,$t)=get_work_counts(4,1,1);
    require_value($c==127 && $t==3,'full acceptance finite tail');
    ($c,$t)=get_work_counts(4,0,1);
    require_value($c==254 && $t==3,'rejection progress includes correction, no bonus');
    ($c,$t)=get_work_counts(2,0,1);
    require_value($c==255 && $t==1,'two-row partial case');
    ($c,$t)=get_work_counts(4,.95,1,0);
    require_value($c==0 && $t==511,'ineligible drafter uses ordinary serial steps');
    my $trial={metrics=>{decode_elapsed_ms=>511*20},client_elapsed_ms=>511*20+500};
    my $plain=get_projection($trial,4,1,1,0,0,20,7);
    require_value(abs($plain->{http_ms}-$trial->{client_elapsed_ms})<1e-8,'no reuse equals serial');
    my $draft=get_projection($trial,4,1,1,0,2,20,7);
    require_value(abs($draft->{http_ms}-$plain->{http_ms}-254)<1e-8,'draft charged per actual cycle');
    return {is_self_test_passed=>JSON::PP::true,behavioral_cases=>7};
}

my $path=shift @ARGV // die "usage: $0 RAW [--audit-only] | --self-test\n";
if ($path eq '--self-test') {print $json->pretty->encode(self_test()); exit;}
my $rows=read_rows($path);
my $audit=validate_report($rows);
if (@ARGV && $ARGV[0] eq '--audit-only') {print $json->pretty->encode($audit); exit;}

my @trials=grep {$_->{kind} eq 'trial' && $_->{runtime} eq 'm40-llm' && $_->{phase} eq 'screen'}
    @{read_rows("${root}maxwell-compound-runtime-ollama-http.jsonl")};
require_value(@trials==5,'reference request coverage');
my (@widths,%reject_extra);
for my $k (2,4,8) {
    my @full=map {$_->{row}} grep {$_->{row}{width}==$k} @{records($rows,'direct_q8_cost_trial')};
    my @rejections=map {$_->{row}} grep {$_->{row}{width}==$k} @{records($rows,'direct_q8_cost_rejection')};
    my @positions=sort {$a<=>$b} keys %{{map {$_->{accepted_prefix}=>1} @rejections}};
    my @r=map {my $a=$_;my @v=grep {$_->{accepted_prefix}==$a} @rejections;
        {accepted=>$a,commit_ms=>median(map {$_->{commit_ms}} @v),correction_ms=>median(map {$_->{correction_ms}} @v)}} @positions;
    # Charge the worst sampled median plus the existing early-to-late
    # attention-span increase. This is a scenario allowance, not a measured
    # universal bound on late-context correction.
    $reject_extra{$k}=max_value(map {$_->{commit_ms}+$_->{correction_ms}} @r)+0.3274;
    my @on=grep {$_->{is_profiled}} @full;
    my %stages;
    for my $r (@on) {
        my %totals;
        for my $s (@{$r->{gpu_profile}{stages}}) {$totals{$s->{stage}}+=$s->{gpu_elapsed_ms};}
        for my $s (keys %totals) {push @{$stages{$s}},$totals{$s};}
    }
    push @widths,{width=>$k,off_batch_ms=>median(map {$_->{elapsed_ms}} grep {!$_->{is_profiled}} @full),
        on_batch_ms=>median(map {$_->{elapsed_ms}} @on),stage_interval_medians_ms=>{map {$_=>median(@{$stages{$_}})} keys %stages},
        rejection_components=>\@r,charged_rejection_extra_ms=>$reject_extra{$k},
        shared_bytes=>1024*$k+512,accumulators_per_thread=>2*$k,
        provisional_kv_bytes=>36864*$k,forward_workspace_bytes_without_f32_weights=>207880*$k,
        logical_gateup_bytes_per_group=>1724645376,logical_gateup_bytes_saved=>($k-1)*1724645376};
}

my @curves;
for my $k (2,4,8) {
    for my $p (0,.25,.5,.75,.9,.95,.99,1) {
        for my $beta (.5,.75,1) {
            for my $extra (1,5) {
                for my $draft (0,.25,.5,1,2,5) {
                    my @v=map {get_projection($_,$k,$p,$beta,$extra,$draft,$reject_extra{$k},7.4215)} @trials;
                    push @curves,{width=>$k,guess_acceptance=>$p,gate_time_ratio=>$beta,additional_batch_ms=>$extra,drafter_ms=>$draft,
                        http_tok_per_sec=>median(map {$_->{http_tok_per_sec}} @v),
                        min_gain=>min_value(map {$_->{gain}} @v),max_gain=>max_value(map {$_->{gain}} @v),
                        expected_progress=>do {my $n=0;for (@{get_distribution($k,$p,1)}) {$n+=$_->[0]*$_->[1]} $n},
                        is_hypothetical=>JSON::PP::true};
                }
            }
        }
    }
}

my @budgets;
for my $p (0,.25,.5,.75,.9,.95,.99,1) {
    my @v=map {get_projection($_,4,$p,.5,1,0,$reject_extra{4},7.4215)} @trials;
    for my $target ('break-even','three-percent',53.06,56.153149) {
        my @maximum;
        for my $i (0..$#trials) {
            my $limit=$target eq 'break-even' ? $trials[$i]{client_elapsed_ms}
                : $target eq 'three-percent' ? $trials[$i]{client_elapsed_ms}/1.03 : 512000/$target;
            push @maximum,($limit-$v[$i]{target_only_http_ms})/$v[$i]{expected_cycles};
        }
        push @budgets,{width=>4,guess_acceptance=>$p,gate_time_ratio=>.5,additional_batch_ms=>1,
            target=>$target,max_drafter_ms_min_over_reference_trials=>min_value(@maximum)};
    }
}

my @minimum;
for my $target ('break-even','three-percent',53.06,56.153149) {
    for my $draft (.25,1) {
        for my $extra (1,5) {
            my ($lo,$hi)=(0,1);
            my $is_possible=1;
            for my $iteration (0..35) {
                my $p=$iteration==0 ? 1 : ($lo+$hi)/2;
                my @v=map {get_projection($_,4,$p,.5,$extra,$draft,$reject_extra{4},7.4215)} @trials;
                my $passes=1;
                for my $i (0..$#v) {
                    my $limit=$target eq 'break-even' ? $trials[$i]{client_elapsed_ms}
                        : $target eq 'three-percent' ? $trials[$i]{client_elapsed_ms}/1.03 : 512000/$target;
                    $passes=0 if $v[$i]{http_ms}>=$limit;
                }
                if ($iteration==0 && !$passes) {$is_possible=0;last;}
                if ($passes) {$hi=$p} else {$lo=$p}
            }
            push @minimum,{target=>$target,drafter_ms=>$draft,additional_batch_ms=>$extra,
                gate_time_ratio=>.5,min_guess_acceptance=>$is_possible ? $hi : undef};
        }
    }
}

my @limits;
for my $k (2,4,8) {
    for my $target ('break-even','three-percent',53.06,56.153149) {
        my @v;
        for my $trial (@trials) {
            my $serial=$trial->{metrics}{decode_elapsed_ms}/511;
            my $fixed=$trial->{client_elapsed_ms}-$trial->{metrics}{decode_elapsed_ms};
            my $limit=$target eq 'break-even' ? $trial->{client_elapsed_ms}
                : $target eq 'three-percent' ? $trial->{client_elapsed_ms}/1.03 : 512000/$target;
            push @v,($limit-$fixed-(511%$k)*$serial)/int(511/$k);
        }
        push @limits,{width=>$k,target=>$target,max_perfect_verifier_ms_min_over_reference_trials=>min_value(@v),
            drafter_ms=>0,scope=>'necessary perfect-acceptance bound, not a real-drafter forecast'};
    }
}
my @sensitivity;
for my $p (.9,.95,.99) {
    for my $draft (.25,1) {
        for my $extra (1,5) {
            for my $factor (.9,1) {
                my @v=map {get_projection($_,4,$p,.5,$extra,$draft,$reject_extra{4},7.4215*$factor)} @trials;
                push @sensitivity,{guess_acceptance=>$p,drafter_ms=>$draft,additional_batch_ms=>$extra,
                    gate_time_ratio=>.5,credited_stage_fraction=>$factor,
                    min_gain=>min_value(map {$_->{gain}} @v),http_tok_per_sec=>median(map {$_->{http_tok_per_sec}} @v)};
            }
        }
    }
}
my @eligibility;
for my $availability (.25,.5,1) {
    for my $extra (1,5) {
        my @v=map {get_projection($_,4,.95,.5,$extra,.25,$reject_extra{4},7.4215,$availability)} @trials;
        push @eligibility,{candidate_eligibility=>$availability,guess_acceptance=>.95,gate_time_ratio=>.5,
            additional_batch_ms=>$extra,drafter_ms=>.25,min_gain=>min_value(map {$_->{gain}} @v),
            http_tok_per_sec=>median(map {$_->{http_tok_per_sec}} @v),
            scope=>'optimistic no-match lookup cost zero; no-match rounds remain serial'};
    }
}
print $json->pretty->encode({schema_version=>1,audit=>$audit,
    scope=>'hypothetical direct-Q8 costs, not measured verifier or HTTP throughput; stage-to-wall mapping is an assumption',
    known_first_candidate=>JSON::PP::true,continuation_input_rows=>511,
    main_scenario_candidate_eligibility=>1,
    projection_q8_bytes=>2947940352,temporary_f32_write_bytes=>11098128384,
    logical_materialization_roundtrip_bytes=>22196256768,gate_stage_anchor_ms=>7.4215,
    correction_attention_allowance_ms=>.3274,real_drafter_acceptance_measured=>JSON::PP::false,
    reference_request_inputs=>[map {{pair=>$_->{pair},http_ms=>$_->{client_elapsed_ms},
        decode_ms=>$_->{metrics}{decode_elapsed_ms},fixed_ms=>$_->{client_elapsed_ms}-$_->{metrics}{decode_elapsed_ms}}} @trials],
    widths=>\@widths,curves=>\@curves,four_row_drafter_budgets=>\@budgets,minimum_guess_acceptance=>\@minimum,
    perfect_verifier_limits=>\@limits,stage_mapping_sensitivity=>\@sensitivity,
    candidate_eligibility_sensitivity=>\@eligibility,
    four_row_static_resources=>[map {{assumed_registers_per_thread=>$_,
        registers_per_block=>128*$_,shared_bytes=>4608,
        theoretical_resident_blocks_upper_bound=>int(65536/(128*$_)),
        scope=>'uncompiled resource arithmetic, not measured occupancy'}} (40,48,64)]});
