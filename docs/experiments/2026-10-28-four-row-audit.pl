#!/usr/bin/env perl
# Bind completed observations to the fixed contract and actual timing pairs.
use strict;
use warnings;
use JSON::PP;
my $json = JSON::PP->new->canonical;
sub require_value {die "$_[1]\n" unless $_[0];}
sub read_rows {
    open my $f,'<',$_[0] or die $!;
    return [map {$json->decode($_)} <$f>];
}
sub same {return $json->encode($_[0]) eq $json->encode($_[1]);}
sub median {
    my @v=sort {$a<=>$b} @_;
    require_value(@v>0,'empty median');
    return @v%2 ? $v[int(@v/2)] : ($v[@v/2-1]+$v[@v/2])/2;
}
sub gpu {
    my ($s)=@_;
    my @f=map {s/^\s+|\s+$//gr} split /,/,$s->{device};
    require_value(@f==9 && $f[0] eq 'GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852'
        && $f[3] eq 'P0' && $f[5] eq '1113 MHz' && $f[6] eq '3004 MHz','GPU clocks/identity');
    require_value($s->{processes}=~/^([0-9]+), [^\n]+$/,'GPU process isolation');
    return $1;
}
my $rows=read_rows(shift @ARGV // die 'provide report');
my $m=$rows->[0];
require_value($m->{kind} eq 'r4_manifest' && $m->{schema_version}==1 && $m->{worktree} eq ''
    && $m->{revision}=~/^[a-f0-9]{40}$/ && $m->{executable_sha256}=~/^[a-f0-9]{64}$/
    && $m->{model_sha256} eq '4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d'
    && $m->{fixture_sha256} eq 'eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687'
    && $m->{capture_report_sha256} eq 'd9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801'
    && $m->{rows}==4 && $m->{d}==2048 && $m->{h}==11008 && $m->{threads}==128
    && $m->{shared_bytes}==4608 && $m->{iterations}==10 && $m->{build_profile} eq 'release'
    && same($m->{arms},['serial','independent','shared-read']),'manifest/shape/fixture binding');
my ($capture)=grep {$_->{kind} eq 'ffn_capture_manifest'} @{read_rows('docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl')};
require_value(same($m->{execution_configuration},$capture->{execution_configuration})
    && same($m->{prepared}{tensors},$capture->{prepared_weights}{tensors})
    && $m->{prepared}{resident_bytes}==1192935424 && $m->{source_bytes}==3279519744,'prepared/source identity');
my $done=$rows->[-1];
require_value($done->{kind} eq 'r4_complete' && $done->{is_complete}
    && !$done->{is_runtime_claim} && $done->{device_bytes_after}==0,'completion/teardown');
require_value(@$rows==($m->{correctness_only}?17:29+3*$m->{trials}),'record coverage');
my @conclusions;
my %pids;
my $last_end=0;
for my $layer (0,17,35) {
    my @r=grep {defined($_->{layer_id}) && $_->{layer_id}==$layer} @$rows;
    my @prep=grep {$_->{kind} eq 'r4_preparation'} @r;
    my @cleanup=grep {$_->{kind} eq 'r4_layer_cleanup'} @r;
    require_value(@prep==1 && @cleanup==1 && $prep[0]{tracked_before}==4472455168
        && $prep[0]{tracked_resident}==4520956224 && $prep[0]{scratch_bytes}==594240
        && $prep[0]{joint}{additional_prepared_bytes}==47906816
        && $prep[0]{joint}{control_prepared_bytes}==0
        && $prep[0]{down_payload_sha256}=~/^[a-f0-9]{64}$/
        && $cleanup[0]{tracked_after}==4472455168,'ownership/accounting');
    my @valid=grep {$_->{kind} eq 'r4_validation'} @r;
    require_value(@valid==3 && same([map {$_->{arm}} @valid],$m->{arms}),'validation coverage');
    for my $v (@valid) {
        require_value($v->{has_exact_bits} && $v->{has_ordinary_capture_parity},'exact/ordinary parity');
        for my $stage ('gate','up','hidden','down') {
            my $s=$v->{cpu}{$stage};
            require_value($s->{compared_values}==($stage eq 'down'?8192:44032)
                && $s->{max_abs_diff}>=0 && $s->{max_abs_diff}<1e-3
                && $s->{mean_abs_diff}>=0 && $s->{mean_abs_diff}<1e-3,'CPU error/coverage');
        }
    }
    my @trials=grep {$_->{kind} eq 'r4_trial'} @r;
    if ($m->{correctness_only}) {require_value(@trials==0,'unexpected primary timing');next;}
    require_value(@trials==$m->{trials} && ($m->{trials}==5||$m->{trials}==10||$m->{trials}==20),'trial coverage');
    my @warm=grep {$_->{kind} eq 'r4_warmup'} @r;
    require_value(@warm==3 && same([map {$_->{arm}} @warm],$m->{arms})
        && !(grep {$_->{warmup}{gpu_ms}<1000 || $_->{warmup}{batches}>256} @warm),'warmup');
    my @times;
    for my $trial (0..$#trials) {
        my $t=$trials[$trial];
        require_value($t->{trial}==$trial && $t->{scratch_bytes}==594240
            && $t->{tracked_device_bytes}==4520956224,'trial ordering/residency');
        $pids{gpu($t->{gpu_before})}++;$pids{gpu($t->{gpu_after})}++;
        my @order=(0,1,2);push @order,shift @order for 1..($trial%3);@order=reverse @order if $trial%2;
        require_value(same($t->{order},\@order) && @{$t->{measurements}}==3,'arm schedule');
        my @complete;
        for my $index (0..2) {
            my $a=$order[$index];my $r=$t->{measurements}[$index];
            require_value($r->{arm} eq $m->{arms}[$a] && @{$r->{stages}}==3,'arm coverage');
            my @stages=('gate_up','down','complete');@stages=reverse @stages if $trial%2;
            require_value(same([map {$_->{stage}} @{$r->{stages}}],\@stages),'stage schedule');
            for my $s (@{$r->{stages}}) {
                my $g=$s->{stage} eq 'down'?0:($a==0?40:10);
                my $d=$s->{stage} eq 'gate_up'?0:40;
                my $expected={map {$_=>0} qw(cublas_calls stream_syncs stream_waits device_allocs device_frees h2d_copies d2h_copies h2d_bytes d2h_bytes)};
                $expected->{launches}=$g+$d;
                my %dispatch;
                $dispatch{gemm_f32xq8_separate_f32}=$d if $d;
                $dispatch{$a==2?'ffn_q8_joint_four_rows_128':'ffn_q8_joint_f32x4_shared_scales_128'}=$g if $g;
                require_value(same($s->{enqueue_counters},$expected) && same($s->{dispatch},\%dispatch)
                    && $s->{iterations}==10 && $s->{gpu_ms}>0 && $s->{host_enqueue_and_completion_ms}>0
                    && $s->{started_unix_us}>=$last_end && $s->{completed_unix_us}>$s->{started_unix_us},'dispatch/counters/intervals');
                $last_end=$s->{completed_unix_us};
                $complete[$a]=$s->{gpu_ms} if $s->{stage} eq 'complete';
            }
        }
        push @times,\@complete;
    }
    my @summaries=grep {$_->{kind} eq 'r4_summary'} @r;
    require_value(@summaries==1,'summary coverage');
    my ($passes,$borderline,$max_variation)=(1,0,0);
    for my $control (0,1) {
        my $s=$summaries[0]{$control==0?'versus_serial':'versus_independent'};
        my @pairs=map {[$_->[$control],$_->[2]]} @times;
        my $gain=median(map {$_->[0]} @pairs)/median(map {$_->[1]} @pairs)-1;
        require_value(same($s->{pairs},\@pairs) && abs($s->{throughput_gain}-$gain)<1e-12
            && $s->{bootstrap_resamples}==10000 && $s->{bootstrap_seed}==0,'actual pair/summary binding');
        my ($lo,$hi)=@{$s->{bootstrap_gain_ci95}};
        my @ratios=map {$_->[0]/$_->[1]-1} @pairs;
        my @sorted=sort {$a<=>$b} @ratios;
        require_value($lo<=$hi && $lo>=$sorted[0]-1e-12 && $hi<=$sorted[-1]+1e-12,'paired interval range');
        require_value(($s->{passes_micro_screen}?1:0)==($gain>=.05 && $lo>0?1:0),'gate flag');
        $passes=0 unless $gain>=.05 && $lo>0;
        $borderline=1 if abs($gain-.05)<=.01 || ($gain>=.05 && $lo<=0);
    }
    for my $a (0..2) {
        my @rates=map {1/$_->[$a]} @times;
        my $middle=median(@rates);my $variation=median(map {abs($_-$middle)} @rates)/$middle;
        $max_variation=$variation if $variation>$max_variation;
    }
    push @conclusions,{layer=>$layer,is_micro_passed=>$passes?JSON::PP::true:JSON::PP::false,
        needs_confirmation=>$borderline?JSON::PP::true:JSON::PP::false,max_variation=>$max_variation};
}
require_value(keys(%pids)<=1,'GPU PID changed');
print $json->pretty->encode({is_audit_passed=>JSON::PP::true,revision=>$m->{revision},
    executable_sha256=>$m->{executable_sha256},correctness_only=>$m->{correctness_only},
    layers=>\@conclusions,device_bytes_after=>0,
    confidence_scope=>'checks actual pairs, recorded seed/resample contract and interval range; bootstrap computed by existing Rust summarizer'});
