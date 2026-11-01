#!/usr/bin/env perl
use strict;
use warnings;
use JSON::PP;
my $json=JSON::PP->new->canonical;
sub read_rows {open my $f,'<',$_[0] or die $!;return [map {$json->decode($_)} <$f>];}
sub check {die 'missing predicate' unless @_==2;die "$_[1]\n" unless $_[0];}
sub same {return $json->encode($_[0]) eq $json->encode($_[1]);}
sub median {my @v=sort {$a<=>$b} @_;my $n=@v;return $n%2 ? $v[int($n/2)] : ($v[$n/2-1]+$v[$n/2])/2;}
my $root='docs/experiments/';
my $heldout=read_rows($root.'2026-10-28-draft-heldout.jsonl');
my $quality=read_rows($root.'2026-10-27-maxwell-compound-runtime-quality.jsonl');
my %traces;
for my $r (@$quality) {
    if ($r->{kind} eq 'q8_runtime_browser_generation' && $r->{label} eq 'candidate') {
        $traces{browser}={ids=>$r->{generation}{generated_token_ids},stop=>$r->{generation}{metrics}{stop_reason},prompt=>$r->{prompt_token_ids}};
    } elsif ($r->{kind} eq 'model_quality_task') {
        $traces{$r->{id}}={ids=>$r->{candidate}{generated_token_ids},stop=>$r->{candidate}{metrics}{stop_reason},prompt=>$r->{prompt_token_ids}};
    }
}
for my $r (grep {$_->{kind} eq 'draft_target_trace'} @$heldout) {
    $traces{$r->{trace}{id}}={ids=>$r->{trace}{generated_token_ids},stop=>$r->{trace}{stop_reason},prompt=>$r->{trace}{prompt_token_ids}};
}
sub audit {
    my ($q,$timing)=@_;
    check(@$q==23 && $q->[-1]{is_correctness_passed} && $q->[-1]{device_bytes_after}==0,'complete correctness');
    check($timing->[0]{executable_sha256} eq $q->[0]{executable_sha256}
        && $timing->[0]{heldout_sha256} eq $q->[0]{heldout_sha256},'executable/corpus binding');
    check($q->[0]{is_correctness_only} && !$timing->[0]{is_correctness_only},'separate quality/timing');
    my %seen;my ($logits,$kv,$cleared)=(0,0,0);
    for my $r (grep {$_->{kind} eq 'live_copy_correctness'} @$q) {
        my $t=$traces{$r->{id}} or die 'unknown quality case';
        check(!$seen{$r->{id}}++ && $r->{has_exact_ids_and_kv},'unique correct cases');
        for my $arm ($r->{serial},$r->{candidate}) {
            check(same($arm->{generated_token_ids},$t->{ids}) && $arm->{stop_reason} eq $t->{stop},'actual quality outputs');
            check($arm->{target_counters}{cublas_calls}==0 && $arm->{target_counters}{device_allocs}==0
                && $arm->{target_counters}{device_frees}==0,'direct target resources');
        }
        check($r->{candidate}{compared_full_logit_rows}==@{$t->{ids}}-1
            && $r->{candidate}{has_full_logit_tolerance_passed},'full-logit gate');
        check($r->{valid_kv_words}==36*(@{$t->{prompt}}+@{$t->{ids}}-1)*512,'valid KV coverage');
        $logits+=$r->{candidate}{compared_full_logit_rows};$kv+=$r->{valid_kv_words};
        $cleared+=$r->{candidate}{cleared_suffix_words};
    }
    check(keys(%seen)==21,'all quality shapes');
    check(@$timing==30 && $timing->[-1]{is_complete} && $timing->[-1]{device_bytes_after}==0
        && !$timing->[-1]{is_http_claim},'complete timing without HTTP claim');
    check((grep {$_->{kind} eq 'live_copy_warmup'} @$timing)==4,'four warmups');
    my @decisions;
    for my $id ('held-copy-short','held-copy-long') {
        my @trials=grep {$_->{kind} eq 'live_copy_trial' && $_->{id} eq $id} @$timing;
        check(@trials==10,'five pairs per workload');
        my (@a,@b);
        for my $index (0..9) {
            my $trial=$trials[$index];my $r=$trial->{result};my $pair=int($index/2);
            my $expected=($pair%2) ? 1-$index%2 : $index%2;
            check($trial->{trial}==$pair && $trial->{phase} eq 'screen'
                && ($r->{is_direct} ? 1 : 0)==$expected,'alternating pair schedule');
            check(same($r->{generated_token_ids},$traces{$id}{ids}) && $r->{stop_reason} eq $traces{$id}{stop}
                && !$r->{includes_debug_readbacks},'timed actual outputs, no diagnostics');
            for my $gpu ($trial->{gpu_before},$trial->{gpu_after}) {
                check($gpu->{device} eq 'GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB, P0, 1113 MHz, 3004 MHz',
                    'GPU identity/clocks');
                check(scalar($gpu->{processes}=~/^\d+, [^\n]+$/),'sole GPU process');
            }
            my $c=$r->{target_counters};my $rows=@{$traces{$id}{ids}}-1;
            check($c->{cublas_calls}==0 && $c->{device_allocs}==0 && $c->{device_frees}==0
                && $c->{h2d_copies}==$rows && $c->{d2h_copies}==2*$rows
                && $r->{rejections}==0,'charged target execution');
            check($r->{attempts}==($r->{is_direct} ? ($id eq 'held-copy-short' ? 12 : 16) : 0),
                'real copy proposal count');
            my $phases=$r->{setup_ms}+$r->{prefill_ms}+$r->{decode_ms}+$r->{teardown_ms};
            check($r->{total_ms}>=$phases && $r->{total_ms}-$phases<1,'whole request phases');
            my $tps=@{$traces{$id}{ids}}*1000/$r->{total_ms};
            check(abs($tps-$r->{resident_tps})<1e-10,'throughput actual counts');
            push @{$r->{is_direct} ? \@b : \@a},$tps;
        }
        my @summary=grep {$_->{kind} eq 'live_copy_summary' && $_->{id} eq $id} @$timing;
        check(@summary==1,'one unextended summary');my $c=$summary[0]{comparison};
        my $gain=median(@b)/median(@a)-1;
        check(abs($gain-$c->{gain})<1e-12 && $c->{pairs}==5,'whole request medians');
        my $variation=0;
        for my $values (\@a,\@b) {my $m=median(@$values);my $v=median(map {abs($_-$m)} @$values)/$m;$variation=$v if $v>$variation;}
        my $threshold=2*$variation>.03 ? 2*$variation : .03;
        check($c->{promotion_threshold}==$threshold && $c->{gain}-$threshold>.01
            && !$c->{needs_variation_extension} && !$c->{needs_fresh_confirmation}
            && $c->{bootstrap_resamples}==10000 && $c->{bootstrap_seed}==0
            && $c->{bootstrap_gain_ci95}[0]>0 && $c->{passes_promotion_gate},'original statistical gate');
        my @gains;
        for my $code (0..3124) {
            my $s=$code;my (@x,@y);
            for (1..5) {my $i=$s%5;$s=int($s/5);push @x,$a[$i];push @y,$b[$i];}
            push @gains,median(@y)/median(@x)-1;
        }
        @gains=sort {$a<=>$b} @gains;
        check($gains[78]>0 && abs($gains[78]-$c->{bootstrap_gain_ci95}[0])<.001
            && abs($gains[3045]-$c->{bootstrap_gain_ci95}[1])<.001,'independent paired confidence');
        my ($gate)=grep {$_->{kind} eq 'live_copy_gate' && $_->{id} eq $id} @$timing;
        check($gate->{passes_resident_gate},'recorded decision');
        push @decisions,{id=>$id,gain=>$gain,paired_ci95=>$c->{bootstrap_gain_ci95},
            exact_empirical_ci95=>[$gains[78],$gains[3045]]};
    }
    return {is_passed=>JSON::PP::true,correctness_cases=>21,full_logit_rows=>$logits,
        valid_kv_words=>$kv,cleared_suffix_words=>$cleared,pairs_per_copy_case=>5,
        decisions=>\@decisions,is_http_claim=>JSON::PP::false,device_bytes_after=>0};
}
my $q=read_rows($root.'2026-10-28-draft-live-correctness.jsonl');
my $timing=read_rows($root.'2026-10-28-draft-live-measurement.jsonl');
my $result=audit($q,$timing);
if (@ARGV && $ARGV[0] eq '--self-test') {
    # Corrupted output, quality coverage, costs or isolation cannot pass admission.
    my @mutations=(
        sub {$_[0][1]{candidate}{generated_token_ids}[0]=0;},
        sub {$_[0][1]{valid_kv_words}=0;},
        sub {$_[0][1]{candidate}{has_full_logit_tolerance_passed}=JSON::PP::false;},
        sub {$_[1][0]{executable_sha256}='wrong';},
        sub {$_[1][3]{gpu_after}{processes}.="\n2, competitor, 1 MiB";},
        sub {$_[1][3]{result}{total_ms}=1;},
        sub {$_[1][-1]{device_bytes_after}=4;},
        sub {pop @{$_[1]};},
    );
    for my $mutate (@mutations) {
        my $a=$json->decode($json->encode($q));my $b=$json->decode($json->encode($timing));
        $mutate->($a,$b);my $has_passed=eval {audit($a,$b);1};check(!$has_passed,'corrupted evidence accepted');
    }
    $result->{rejected_corruptions}=scalar @mutations;
}
print $json->pretty->encode($result);
