#!/usr/bin/env perl
# Measured operator costs; hypothetical full-verifier assembly and drafting.
use strict;
use warnings;
use JSON::PP;
use Digest::SHA qw(sha256_hex);
my $json=JSON::PP->new->canonical;
sub require_value {die "$_[1]\n" unless $_[0];}
sub read_bytes {open my $f,'<',$_[0] or die $!;local $/;return <$f>;}
sub read_rows {return [map {$json->decode($_)} grep {length} split /\n/,read_bytes($_[0])];}
sub median {my @v=sort {$a<=>$b} @_;my $n=@v;require_value($n>0,'empty median');return $n%2?$v[int($n/2)]:($v[$n/2-1]+$v[$n/2])/2;}
sub minimum {my $v=shift;for (@_) {$v=$_ if $_<$v;}return $v;}
sub maximum {my $v=shift;for (@_) {$v=$_ if $_>$v;}return $v;}

sub work_counts {
    my ($p,$eligibility)=@_;
    my (@cycles,@serial);
    for my $n (0..511) {
        if ($n<4) {($cycles[$n],$serial[$n])=(0,$n);next;}
        my ($c,$s)=($eligibility+(1-$eligibility)*$cycles[$n-1],(1-$eligibility)*(1+$serial[$n-1]));
        for my $accepted (1..4) {
            my $probability=$accepted==4?$p**3:$p**($accepted-1)*(1-$p);
            my $progress=$accepted==4?4:$accepted+1;
            $c+=$eligibility*$probability*$cycles[$n-$progress];
            $s+=$eligibility*$probability*$serial[$n-$progress];
        }
        ($cycles[$n],$serial[$n])=($c,$s);
    }
    return ($cycles[511],$serial[511]);
}

sub project {
    my ($trial,$p,$draft,$extra,$credit,$beta,$gate,$reject,$eligibility)=@_;
    my ($cycles,$tail)=work_counts($p,$eligibility);
    my $s=$trial->{decode_ms}/511;
    my $v=4*$s-(1-$beta)*4*$gate*$credit+$extra;
    my $http=$trial->{fixed_ms}+$cycles*($v+$draft+(1-$p**3)*$reject)+$tail*$s;
    return {http_ms=>$http,http_tok_per_sec=>512000/$http,verifier_ms=>$v,
        min_gain=>$trial->{http_ms}/$http-1,cycles=>$cycles,serial_rows=>$tail,
        other_serial_target_ms=>4*($s-$gate),new_gate_model_ms=>4*$gate*(1-(1-$beta)*$credit)};
}

if (@ARGV && $ARGV[0] eq '--self-test') {
    my ($c,$t)=work_counts(1,1);require_value($c==127 && $t==3,'full-acceptance tail');
    ($c,$t)=work_counts(0,1);require_value($c==254 && $t==3,'known seed/rejection progress');
    ($c,$t)=work_counts(.95,0);require_value($c==0 && $t==511,'ineligible serial fallback');
    my $trial={decode_ms=>10220,fixed_ms=>500,http_ms=>10720};
    my $v=project($trial,1,0,0,1,1,7,20,1);
    require_value(abs($v->{http_ms}-10720)<1e-9,'no reuse equals serial');
    my $d=project($trial,1,2,0,1,1,7,20,1);
    require_value(abs($d->{http_ms}-$v->{http_ms}-254)<1e-9,'draft per actual cycle');
    my $r=project($trial,0,0,0,1,1,7,20,1);
    require_value($r->{http_ms}>10720,'rejected extra rows and correction charged');
    print $json->pretty->encode({is_passed=>JSON::PP::true,behavioral_cases=>6});exit;
}

my $root='docs/experiments/';
my $operator="${root}2026-10-28-four-row-screen.jsonl";
my $rows=read_rows($operator);
my $manifest=$rows->[0];
require_value($rows->[-1]{kind} eq 'r4_complete' && $rows->[-1]{is_complete},'operator incomplete');
my $old=$json->decode(read_bytes("${root}2026-10-27-direct-q8-model.json"));
my @reference=@{$old->{reference_request_inputs}};
require_value(@reference==5 && $old->{gate_stage_anchor_ms}==7.4215,'reference calibration');
my @layers;
for my $layer (0,17,35) {
    my @trials=grep {$_->{kind} eq 'r4_trial' && $_->{layer_id}==$layer} @$rows;
    require_value(@trials==5,'fixed screen coverage');
    my %stage;
    for my $t (@trials) {
        for my $a (@{$t->{measurements}}) {
            for my $s (@{$a->{stages}}) {push @{$stage{$a->{arm}}{$s->{stage}}},$s->{gpu_ms};}
        }
    }
    my %medians=map {my $a=$_; $a=>{map {$_=>median(@{$stage{$a}{$_}})} keys %{$stage{$a}}}} keys %stage;
    my ($summary)=grep {$_->{kind} eq 'r4_summary' && $_->{layer_id}==$layer} @$rows;
    require_value($summary->{versus_serial}{passes_micro_screen} && $summary->{versus_independent}{passes_micro_screen},'operator must pass both controls');
    push @layers,{layer=>$layer,stage_medians_ms=>\%medians,
        gate_ratio=>$medians{'shared-read'}{gate_up}/$medians{serial}{gate_up}};
}
my $beta=maximum(map {$_->{gate_ratio}} @layers);
my $all_layer_gate=36*maximum(map {$_->{stage_medians_ms}{'shared-read'}{gate_up}} @layers);
my ($w4)=grep {$_->{width}==4} @{$old->{widths}};
my $reject=$w4->{charged_rejection_extra_ms};
my @diagnostic=grep {$_->{kind} eq 'direct_q8_cost_trial' && $_->{row}{width}==4 && $_->{row}{is_profiled}}
    @{read_rows("${root}2026-10-27-direct-q8-cost.jsonl")};
my @sgemm_patch;
for my $r (@diagnostic) {
    my ($gate_dequant,$gate_sgemm,$other_dequant,$other_sgemm,$swiglu)=(0,0,0,0,0);
    for my $layer (0..35) {
        my @s=grep {defined($_->{layer_id}) && $_->{layer_id}==$layer} @{$r->{row}{gpu_profile}{stages}};
        my @d=grep {$_->{stage} eq 'q8_prefill_dequantize'} @s;
        my @g=grep {$_->{stage} eq 'q8_prefill_sgemm'} @s;
        require_value(@d==7 && @g==7,'source-order attribution requires seven projections');
        # Existing source order: Q,K,V,output,gate,up,down. This is attribution
        # from audited call order, not tensor-named hardware counter evidence.
        for my $index (0..6) {
            if ($index==4 || $index==5) {$gate_dequant+=$d[$index]{gpu_elapsed_ms};$gate_sgemm+=$g[$index]{gpu_elapsed_ms};}
            else {$other_dequant+=$d[$index]{gpu_elapsed_ms};$other_sgemm+=$g[$index]{gpu_elapsed_ms};}
        }
        $swiglu+=$_->{gpu_elapsed_ms} for grep {$_->{stage} eq 'prefill_swiglu'} @s;
    }
    push @sgemm_patch,{pair=>$r->{pair},profiled_batch_ms=>$r->{row}{elapsed_ms},
        removed_gate_dequant_ms=>$gate_dequant,removed_gate_sgemm_ms=>$gate_sgemm,
        removed_swiglu_ms=>$swiglu,remaining_dequant_ms=>$other_dequant,remaining_sgemm_ms=>$other_sgemm,
        hypothetical_patch_batch_ms=>$r->{row}{elapsed_ms}-$gate_dequant-$gate_sgemm-$swiglu+$all_layer_gate};
}
require_value(@sgemm_patch==5,'diagnostic coverage');
my @targets=('break-even','three-percent',53.06,56.153149);
my $limit=sub {my ($t,$target)=@_;return $target eq 'break-even'?$t->{http_ms}:
    $target eq 'three-percent'?$t->{http_ms}/1.03:512000/$target;};
my @curves;
for my $p (0,.25,.5,.75,.9,.95,.99,1) {
    for my $draft (0,.25,.5,1,2,5) {
        for my $extra (1,5) {
            for my $credit (.9,1) {
                my @v=map {project($_,$p,$draft,$extra,$credit,$beta,7.4215,$reject,1)} @reference;
                push @curves,{guess_acceptance=>$p,drafter_ms=>$draft,extra_batch_ms=>$extra,
                    credited_stage_fraction=>$credit,median_http_tok_per_sec=>median(map {$_->{http_tok_per_sec}} @v),
                    min_gain=>minimum(map {$_->{min_gain}} @v),verifier_ms_range=>[minimum(map {$_->{verifier_ms}} @v),maximum(map {$_->{verifier_ms}} @v)]};
            }
        }
    }
}
my (@budgets,@minimum,@batch_limits);
for my $target (@targets) {
    for my $p (0,.25,.5,.75,.9,.95,.99,1) {
        my @b;my @max_v;
        for my $t (@reference) {
            my $v=project($t,$p,0,1,1,$beta,7.4215,$reject,1);
            push @b,($limit->($t,$target)-$v->{http_ms})/$v->{cycles};
            my ($cycles,$tail)=work_counts($p,1);
            push @max_v,($limit->($t,$target)-$t->{fixed_ms}-$tail*$t->{decode_ms}/511)/$cycles-.25-(1-$p**3)*$reject;
        }
        push @budgets,{target=>$target,guess_acceptance=>$p,max_drafter_ms=>minimum(@b),extra_batch_ms=>1,credited_stage_fraction=>1};
        push @batch_limits,{target=>$target,guess_acceptance=>$p,drafter_ms=>.25,max_verifier_ms=>minimum(@max_v)};
    }
    for my $draft (.25,1) {
        for my $extra (1,5) {
            for my $credit (.9,1) {
                my ($lo,$hi,$possible)=(0,1,1);
                for my $i (0..35) {
                    my $p=$i==0?1:($lo+$hi)/2;
                    my $passes=1;
                    for my $t (@reference) {
                        $passes=0 if project($t,$p,$draft,$extra,$credit,$beta,7.4215,$reject,1)->{http_ms}>=$limit->($t,$target);
                    }
                    if ($i==0 && !$passes) {$possible=0;last;}
                    if ($passes) {$hi=$p;} else {$lo=$p;}
                }
                push @minimum,{target=>$target,drafter_ms=>$draft,extra_batch_ms=>$extra,
                    credited_stage_fraction=>$credit,minimum_conditional_guess_acceptance=>$possible?$hi:undef};
            }
        }
    }
}
my @eligibility;
for my $e (.25,.5,1) {
    my @v=map {project($_,.95,.25,1,1,$beta,7.4215,$reject,$e)} @reference;
    push @eligibility,{eligibility=>$e,guess_acceptance=>.95,drafter_ms=>.25,extra_batch_ms=>1,
        median_http_tok_per_sec=>median(map {$_->{http_tok_per_sec}} @v),min_gain=>minimum(map {$_->{min_gain}} @v)};
}
print $json->pretty->encode({schema_version=>1,scope=>'measured operator, hypothetical full verifier; no new HTTP result',
    operator_sha256=>sha256_hex(read_bytes($operator)),layers=>\@layers,gate_ratio_used=>$beta,
    slowest_sample_extrapolated_gate_all_layers_ms=>$all_layer_gate,
    original_stage_gate_anchor_ms=>7.4215,serial_continuation_rows=>511,known_first_candidate=>JSON::PP::true,
    rejection_extra_ms=>$reject,reference_request_inputs=>\@reference,
    sgemm_gate_replacement_only=>{scope=>'source-order stage attribution and counterfactual subtraction, not measured assembled verifier',
        rows=>\@sgemm_patch,remaining_conversions_and_sgemms=>180,
        remaining_logical_q8_bytes=>1223294976,remaining_f32_write_bytes=>4605345792,
        median_patch_batch_ms=>median(map {$_->{hypothetical_patch_batch_ms}} @sgemm_patch)},
    direct_single_row_others=>{scope=>'must explicitly route other projections through qualified direct one-row consumers; costs remain in 4*S minus gate',
        credit=>[.9,1],extra_batch_ms=>[1,5],real_drafter_acceptance_measured=>JSON::PP::false,
        provisional_kv_bytes=>147456,old_forward_workspace_bytes=>831520,
        other_stage_anchors_ms=>{down=>4.7614,vocabulary=>1.5638,qkv=>1.3918,output=>1.0164,attention=>.7802},
        attention_scope=>'causal per row with absolute positions; serial anchor and rejection allowance are not a late-context guarantee',
        curves=>\@curves,minimum_acceptance=>\@minimum,drafter_budgets=>\@budgets,batch_limits=>\@batch_limits,
        eligibility_sensitivity=>\@eligibility}});
