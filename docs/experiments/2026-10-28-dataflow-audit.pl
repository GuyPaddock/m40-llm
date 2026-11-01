#!/usr/bin/env perl
use strict;
use warnings;
use JSON::PP;
use Digest::SHA qw(sha256_hex);
use List::Util qw(sum min max);
use File::Temp qw(tempdir);

my $prefix = @ARGV && $ARGV[0] ne '--self-test' ? $ARGV[0] : 'docs/experiments/2026-10-28-dataflow';
my $json = JSON::PP->new->canonical;
sub read_bytes { open my $f, '<', $_[0] or die "$!"; binmode $f; local $/; return <$f>; }
sub rows { return [map {decode_json($_)} grep {length} split /\n/, $_[0]]; }
sub same { die "$_[2]\n" unless $json->encode($_[0]) eq $json->encode($_[1]); }
sub median { my @a=sort {$a<=>$b} @_; die 'empty sample' unless @a; return (@a%2)?$a[@a/2]:($a[@a/2-1]+$a[@a/2])/2; }
sub union_ms {
    my @a=sort {$a->[0]<=>$b->[0]} @_;
    return 0 unless @a;
    my ($start,$end)=@{shift @a}; my $total=0;
    for (@a) { if ($_->[0]<=$end) {$end=max($end,$_->[1]);} else {$total+=$end-$start;($start,$end)=@$_;} }
    return $total+$end-$start;
}

my ($h,$e,$q) = map {read_bytes("$prefix-$_")} ('http.jsonl','events.jsonl','quality.jsonl');
if (@ARGV && $ARGV[0] eq '--self-test') {
    my $dir=tempdir(CLEANUP=>1);
    my @mutations=(
        sub {$_[0][0]{executable_sha256}='wrong'},
        sub {$_[0][-1]{device_bytes_after}=4},
        sub {my ($r)=grep {$_->{kind} eq 'dataflow_http_trial'} @{$_[0]};$r->{row}{metrics}{generated_tokens}=511},
        sub {my ($r)=grep {$_->{kind} eq 'dataflow_http_trial'} @{$_[0]};$r->{gpu_before}{processes}.="\nforeign, 1, competitor, 999"},
        sub {my ($r)=grep {$_->{kind} eq 'dataflow_http_trial'} @{$_[0]};$r->{row}{dispatch}{totals}{stream_syncs}++},
        sub {my ($r)=grep {$_->{kind} eq 'dataflow_correctness'} @{$_[0]};$r->{logit_hashes}[0]='wrong'},
        sub {my ($r)=grep {$_->{kind} eq 'dataflow_http_summary'} @{$_[0]};$r->{comparison}{control}{median}=999}
    );
    for my $mutate (@mutations) {
        my $r=rows($h);$mutate->($r);
        for my $suffix (qw(http events quality)) {
            open my $f,'>',"$dir/case-$suffix.jsonl" or die $!;
            print $f $suffix eq 'http' ? join('',map {$json->encode($_)."\n"} @$r) : $suffix eq 'events' ? $e : $q;
            close $f;
        }
        my $pid=fork();die $! unless defined $pid;
        if (!$pid) {open STDOUT,'>','/dev/null';open STDERR,'>','/dev/null';exec $^X,$0,"$dir/case";die $!;}
        waitpid($pid,0);die 'corruption accepted' unless $? != 0;
    }
    print $json->pretty->encode({negative_controls_rejected=>scalar @mutations});exit;
}
my $http=rows($h); my $quality=rows($q);
same($http->[0]{executable_sha256},$quality->[0]{executable_sha256},'executable mismatch');
same($http->[0]{revision},$quality->[0]{revision},'source mismatch');
die 'quality incomplete' unless $quality->[-1]{is_gate_passed} && $quality->[-1]{device_bytes_after}==0;
my ($binding)=grep {$_->{kind} eq 'dataflow_binding'} @$http;
same($binding->{quality_sha256},sha256_hex($q),'quality hash mismatch');
my ($generation)=grep {$_->{kind} eq 'joint_only_generation'} @$quality;
my ($correct)=grep {$_->{kind} eq 'dataflow_correctness'} @$http;
same($correct->{actual_ids},$generation->{generated_ids},'actual IDs mismatch');
die 'incorrect coverage' unless @{$correct->{actual_ids}}==512 && $correct->{is_bitwise_equal}
    && $correct->{valid_kv_words}==10616832 && $correct->{stop_reason} eq 'length';
my @logits=grep {$_->{kind} eq 'joint_only_logits'} @$quality;
same($correct->{logit_hashes},[map {$_->{candidate_sha256}} @logits],'full logits mismatch');
die 'wrong logit coverage' unless @logits==7;
die 'wrong task coverage' unless scalar(grep {$_->{kind} eq 'joint_only_task'} @$quality)==12;
die 'incomplete measurement' unless $http->[-1]{kind} eq 'dataflow_complete'
    && $http->[-1]{device_bytes_after}==0 && $http->[-1]{is_attribution_allowed};
my @trials=grep {$_->{kind} eq 'dataflow_http_trial'} @$http;
die 'wrong bounded trial count' unless @trials==10;
my (%seen,%rates,%metrics,%samples);
for my $trial (@trials) {
    my ($pair,$arm)=@$trial{qw(pair arm)};
    die 'duplicate/invalid trial' if $seen{"$pair/$arm"}++ || $pair<0 || $pair>4 || $arm<0 || $arm>1;
    die 'wrong phase' unless $trial->{phase} eq 'screen';
    for my $boundary (qw(gpu_before gpu_after)) {
        my $gpu=$trial->{$boundary};
        die 'clock mismatch' unless $gpu->{clocks} =~ /GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, P0, 1113, 3004$/;
        my @process=split /\n/,$gpu->{processes};
        die 'GPU isolation' unless @process==1 && $process[0] =~ /, (\d+),/;
        my $pid=$1;
        $seen{pid}//=$pid; die 'PID changed' unless $seen{pid}==$pid;
    }
    my $row=$trial->{row};
    same($row->{output},$generation->{output},'output differs');
    same($row->{dispatch},$generation->{candidate_dispatch},'dispatch differs');
    my $m=$row->{metrics};
    die 'workload differs' unless $m->{generated_tokens}==512 && $m->{prompt_tokens}==65 && $m->{stop_reason} eq 'length';
    die 'rate invalid' unless abs($row->{http_tok_per_sec}-512000/$row->{http_ms})<1e-8;
    push @{$rates{$arm}},$row->{http_tok_per_sec};
    for my $key (qw(prefill_elapsed_ms decode_elapsed_ms total_elapsed_ms)) {push @{$metrics{$arm}{$key}},$m->{$key};}
    push @{$metrics{$arm}{http_ms}},$row->{http_ms};
    push @{$metrics{$arm}{http_minus_generation_ms}},$row->{http_ms}-$m->{total_elapsed_ms};
    my ($begin,$end)=@$trial{qw(event_begin event_end)};
    die 'event bounds' unless $begin>=0 && $end>=$begin && $end<=length($e);
    my $events=rows(substr($e,$begin,$end-$begin));
    if (!$arm) {die 'off emitted events' if @$events; next;}
    same([map {$_->{generated_step}} @$events],[0,255,511],'sample steps');
    for my $r (@$events) {
        my $step=$r->{generated_step};
        die 'position/pool mismatch' unless $r->{absolute_position}==64+$step && $r->{event_pool_capacity}==592
            && !!$r->{event_pool_reused}==!!($step>0) && $r->{process_id}==$seen{pid};
        die 'span count' unless @{$r->{stages}}==400;
        my (%stages,%layers,@cpu,@gpu);
        for my $s (@{$r->{stages}}) {
            die 'unexpected stream' unless $s->{stream} eq 'decode';
            for my $f (qw(cpu_start_ms cpu_call_elapsed_ms gpu_start_ms gpu_end_ms gpu_elapsed_ms)) {
                die "invalid $f" unless defined($s->{$f}) && $s->{$f}>=0 && $s->{$f}<1000;
            }
            push @cpu,[$s->{cpu_start_ms},$s->{cpu_start_ms}+$s->{cpu_call_elapsed_ms}];
            # Serde prints the shortest f32 round-trip decimal, not its f64 expansion.
            push @gpu,[map {unpack('f',pack('f',$_))} @$s{qw(gpu_start_ms gpu_end_ms)}];
            $stages{$s->{stage}}+=$s->{gpu_elapsed_ms};
            $layers{$s->{layer_id}}++ if defined $s->{layer_id};
        }
        die 'layer coverage' unless keys(%layers)==36 && !grep {$_!=11} values %layers;
        my %host;
        for my $s (@{$r->{host_timings}}) {
            die 'host bounds' unless defined($s->{start_ms}) && $s->{start_ms}>=0 && $s->{elapsed_ms}>=0
                && $s->{start_ms}+$s->{elapsed_ms}<=$r->{token_wall_ms};
            push @cpu,[$s->{start_ms},$s->{start_ms}+$s->{elapsed_ms}];
            $host{$s->{operation}}+=$s->{elapsed_ms};
        }
        die 'union mismatch' unless abs(union_ms(@gpu)-$r->{gpu_stage_union_ms})<1e-5;
        die 'steady allocation' if $step && ($r->{counters}{device_allocs} || $r->{counters}{device_frees});
        die 'decode schedule changed' unless $r->{counters}{launches}==435 && $r->{counters}{stream_syncs}==2
            && $r->{counters}{stream_waits}==0 && $r->{counters}{cublas_calls}==0;
        push @{$samples{$step}}, {stages=>\%stages,host=>\%host,counters=>$r->{counters},
            cpu_interval_union_ms=>union_ms(@cpu),cpu_enqueue_sum_ms=>sum(map {$_->{cpu_call_elapsed_ms}} @{$r->{stages}}),
            token_wall_ms=>$r->{token_wall_ms},gpu_union_ms=>$r->{gpu_stage_union_ms},
            gpu_span_ms=>$r->{gpu_timeline_span_ms},wall_minus_gpu_union_ms=>$r->{wall_minus_gpu_union_ms}};
    }
}
my @steps;
for my $step (sort {$a<=>$b} keys %samples) {
    my $s=$samples{$step}; die 'sample count' unless @$s==5;
    my %out=(generated_step=>0+$step,samples=>5,counters=>$s->[0]{counters});
    for my $key (qw(cpu_interval_union_ms cpu_enqueue_sum_ms token_wall_ms gpu_union_ms gpu_span_ms wall_minus_gpu_union_ms)) {
        $out{$key}=median(map {$_->{$key}} @$s);
    }
    for my $category (qw(stages host)) {for my $key (sort keys %{$s->[0]{$category}}) {
        $out{$category}{$key}=median(map {$_->{$category}{$key}} @$s);
    }}
    push @steps,\%out;
}
my %timing;
for my $arm (0,1) {for my $key (keys %{$metrics{$arm}}) {$timing{$arm}{$key}=median(@{$metrics{$arm}{$key}});}}
my $off=median(@{$rates{0}}); my $on=median(@{$rates{1}});
my ($summary)=grep {$_->{kind} eq 'dataflow_http_summary'} @$http;
die 'median mismatch' unless abs($summary->{comparison}{control}{median}-$off)<1e-8 && abs($summary->{comparison}{candidate}{median}-$on)<1e-8;
die 'perturbation gate' unless abs($off/$on-1)<=0.03;
my @gains;
for my $code (0..3124) {
    my (@off,@on); my $index=$code;
    for (1..5) {my $i=$index%5;$index=int($index/5);push @off,$rates{0}[$i];push @on,$rates{1}[$i];}
    push @gains,median(@on)/median(@off)-1;
}
@gains=sort {$a<=>$b} @gains;
die 'paired confidence differs' unless abs($gains[78]-$summary->{comparison}{bootstrap_gain_ci95}[0])<0.001
    && abs($gains[3045]-$summary->{comparison}{bootstrap_gain_ci95}[1])<0.001;
print $json->pretty->encode({is_passed=>JSON::PP::true,primary_trials=>10,primary_samples=>15,
    exhaustive_paired_interval=>[$gains[78],$gains[3045]],
    http_timing_medians=>\%timing,steps=>\@steps,http_off_tok_s=>$off,http_on_tok_s=>$on,
    latency_change=>$off/$on-1,scope=>'Host and device clocks separate; interval costs overlap; not removable work or measured DRAM traffic'});
