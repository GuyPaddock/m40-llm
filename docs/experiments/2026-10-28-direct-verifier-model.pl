#!/usr/bin/env perl
# Finite continuation costs from measured whole targets, not FFN extrapolation.
use strict;
use warnings;
use JSON::PP;
use Digest::SHA qw(sha256_hex);
my $json = JSON::PP->new->canonical;
sub require_true { die "$_[1]\n" unless $_[0]; }
sub read_bytes { open my $f,'<',$_[0] or die $!; local $/; return <$f>; }
sub read_rows { return [map {$json->decode($_)} grep {length} split /\n/,read_bytes($_[0])]; }
sub median { my @v=sort {$a<=>$b} @_; my $n=@v;
    return $n%2 ? $v[int($n/2)] : ($v[$n/2-1]+$v[$n/2])/2; }
sub mean { my $sum=0; $sum+=$_ for @_; return $sum/@_; }
sub maximum { my $v=shift; for (@_) {$v=$_ if $_>$v;} return $v; }
sub minimum { my $v=shift; for (@_) {$v=$_ if $_<$v;} return $v; }
sub summarize { return {min=>minimum(@_),median=>median(@_),max=>maximum(@_)}; }

# Four target inputs contain one known seed and three guessed tokens.
# A rejected guess commits its verified prefix plus one ordinary correction.
sub calculate_decode {
    my ($trial,$rejection,$p,$eligibility,$extra_batch_ms)=@_;
    my (@cost,@draft_calls,@attempts,@rejects,@serial);
    ($cost[0],$draft_calls[0],$attempts[0],$rejects[0],$serial[0])=(0,0,0,0,0);
    my $control=$trial->{control};
    my $direct=$trial->{candidate};
    my $bookkeeping=$direct->{decode_other_ms}/130;
    for my $remaining (1..511) {
        my $position=511-$remaining;
        my $serial_ms=$position>=508 ? $direct->{serial_step_ms}[$position-508]
            : $control->{serial_step_ms}[$position];
        if ($remaining<4) {
            $cost[$remaining]=$bookkeeping+$serial_ms+$cost[$remaining-1];
            $draft_calls[$remaining]=$draft_calls[$remaining-1];
            $attempts[$remaining]=$attempts[$remaining-1];
            $rejects[$remaining]=$rejects[$remaining-1];
            $serial[$remaining]=1+$serial[$remaining-1];
            next;
        }
        # Interpolate context-dependent complete-batch observations, clamping
        # the final three possible off-grid starts to the last measured batch.
        my $index=int($position/4);
        my $fraction=($position%4)/4;
        my $v=$direct->{batch_ms}[$index];
        $v+=(($direct->{batch_ms}[$index+1])-$v)*$fraction if $index<126;
        $v+=$extra_batch_ms;
        my $c=$bookkeeping+(1-$eligibility)*($serial_ms+$cost[$remaining-1]);
        my $d=1+(1-$eligibility)*$draft_calls[$remaining-1];
        my $a=(1-$eligibility)*$attempts[$remaining-1];
        my $r=(1-$eligibility)*$rejects[$remaining-1];
        my $s=(1-$eligibility)*(1+$serial[$remaining-1]);
        for my $accepted (1..4) {
            my $prob=$eligibility*($accepted==4 ? $p**3 : $p**($accepted-1)*(1-$p));
            my $progress=$accepted==4 ? 4 : $accepted+1;
            my $extra=0;
            if ($accepted<4) {
                my $observed=$rejection->[$accepted];
                my $target_delta=maximum(0,$observed->{target_ms}-$direct->{batch_ms}[0]);
                my $correction=maximum($observed->{correction_ms},
                    $control->{serial_step_ms}[$position+$accepted]);
                $extra=$target_delta+$observed->{rollback_ms}+$correction
                    +$observed->{other_ms};
            }
            $c+=$prob*($v+$extra+$cost[$remaining-$progress]);
            $d+=$prob*$draft_calls[$remaining-$progress];
            $a+=$prob*(1+$attempts[$remaining-$progress]);
            $r+=$prob*(($accepted<4 ? 1 : 0)+$rejects[$remaining-$progress]);
            $s+=$prob*$serial[$remaining-$progress];
        }
        ($cost[$remaining],$draft_calls[$remaining],$attempts[$remaining],
            $rejects[$remaining],$serial[$remaining])=($c,$d,$a,$r,$s);
    }
    return {decode_without_drafter_ms=>$cost[511],draft_decisions=>$draft_calls[511],
        attempts=>$attempts[511],rejections=>$rejects[511],ordinary_serial_rows=>$serial[511]};
}

if (@ARGV && $ARGV[0] eq '--self-test') {
    # Perfect acceptance must reproduce full measured request accounting;
    # rejected guesses cannot be credited as four useful target rows.
    my $trial={control=>{serial_step_ms=>[(20)x511]},
        candidate=>{batch_ms=>[(60)x127],serial_step_ms=>[(20)x3],decode_other_ms=>.13}};
    my $rejections=[map {{target_ms=>60,rollback_ms=>.02,correction_ms=>20,other_ms=>.01}} 0..3];
    my $v=calculate_decode($trial,$rejections,1,1,0);
    require_true(abs($v->{decode_without_drafter_ms}-7680.13)<1e-8
        && $v->{attempts}==127 && $v->{ordinary_serial_rows}==3,'perfect finite tail');
    require_true($v->{draft_decisions}==127,'one all-in draft cost per decision');
    my $r=calculate_decode($trial,$rejections,0,1,0);
    require_true($r->{attempts}==254 && $r->{rejections}==254
        && $r->{ordinary_serial_rows}==3 && $r->{decode_without_drafter_ms}>20000,
        'known seed plus one correction, not four accepted guesses');
    my $s=calculate_decode($trial,$rejections,.95,0,0);
    require_true($s->{attempts}==0 && $s->{ordinary_serial_rows}==511
        && abs($s->{decode_without_drafter_ms}-10220.511)<1e-8,'ineligible serial work retained');
    my $extra=calculate_decode($trial,$rejections,1,1,5);
    require_true(abs($extra->{decode_without_drafter_ms}-$v->{decode_without_drafter_ms}-635)<1e-8,
        'unmeasured extra cost charged per attempt');
    print $json->pretty->encode({is_passed=>JSON::PP::true,behavioral_cases=>5}); exit;
}

my $root='docs/experiments/';
my $path=$root.'2026-10-28-direct-verifier-measurement.jsonl';
my $rows=read_rows($path);
require_true($rows->[-1]{is_complete} && $rows->[-1]{device_bytes_after}==0,'completed measurements');
my ($gate)=grep {$_->{kind} eq 'direct_verifier_gate'} @$rows;
require_true($gate->{passes_resident_gate},'full verifier gate precedes drafting model');
my @trials;
for my $i (0..4) {
    my @pair=grep {$_->{kind} eq 'direct_verifier_trial' && $_->{trial}==$i} @$rows;
    require_true(@pair==2,'five contemporaneous pairs');
    my %t=(pair=>$i);
    for my $arm (@pair) {$t{$arm->{is_direct} ? 'candidate' : 'control'}=$arm->{result};}
    push @trials,\%t;
}
my @rejection;
for my $accepted (0..3) {
    my @cycles=map {$_->{result}} grep {$_->{kind} eq 'direct_verifier_rejection'
        && $_->{result}{accepted_prefix}==$accepted} @$rows;
    require_true(@cycles==5,'rejection samples');
    my %r=(accepted=>$accepted);
    for my $field (qw(target_ms rollback_ms correction_ms elapsed_ms)) {
        $r{$field}=mean(map {$_->{$field}} @cycles);
        $r{$field.'_range'}=summarize(map {$_->{$field}} @cycles);
    }
    $r{other_ms}=maximum(0,$r{elapsed_ms}-$r{target_ms}-$r{rollback_ms}-$r{correction_ms});
    push @rejection,\%r;
}
my $old=$json->decode(read_bytes($root.'2026-10-28-four-row-model.json'));
my @reference=@{$old->{reference_request_inputs}};
my $http_fixed=maximum(map {$_->{fixed_ms}} @reference);
my $setup_delta=maximum(0,map {$_->{candidate}{session_setup_ms}+$_->{candidate}{session_teardown_ms}
    -$_->{control}{session_setup_ms}-$_->{control}{session_teardown_ms}} @trials);

my %cache;
sub model_rows {
    my ($p,$eligibility,$extra)=@_;
    my $key=join(',', $p,$eligibility,$extra);
    return $cache{$key} if exists $cache{$key};
    my @v=map {calculate_decode($_,\@rejection,$p,$eligibility,$extra)} @trials;
    return $cache{$key}=\@v;
}
sub limit_decode {
    my ($index,$target)=@_;
    my $t=$trials[$index];
    if ($target eq 'resident-break-even' || $target eq 'resident-three-percent') {
        my $factor=$target eq 'resident-break-even' ? 1 : 1.03;
        return $t->{control}{total_ms}/$factor
            -($t->{candidate}{total_ms}-$t->{candidate}{decode_ms});
    }
    # Historical HTTP fixed overhead is an explicit sensitivity assumption,
    # not a fresh HTTP measurement or a paired Ollama qualification.
    return 512000/$target-$http_fixed-$setup_delta;
}
sub max_drafter {
    my ($p,$target,$extra,$eligibility)=@_;
    my $v=model_rows($p,$eligibility,$extra);
    return minimum(map {(limit_decode($_,$target)-$v->[$_]{decode_without_drafter_ms})
        /$v->[$_]{draft_decisions}} 0..4);
}

my (@perfect,@curves,@budgets,@acceptance,@eligibility);
for my $i (0..4) {
    my $v=model_rows(1,1,0)->[$i];
    require_true(abs($v->{decode_without_drafter_ms}-$trials[$i]{candidate}{decode_ms})<1e-6,
        'perfect model reproduces actual full decode');
    push @perfect,{pair=>$i,serial_mean_row_ms=>$trials[$i]{control}{decode_ms}/511,
        mean_complete_batch_ms=>mean(@{$trials[$i]{candidate}{batch_ms}}),
        serial_total_ms=>$trials[$i]{control}{total_ms},candidate_total_ms=>$trials[$i]{candidate}{total_ms},
        %$v};
}
my @targets=('resident-break-even','resident-three-percent',53.06,56.153149);
for my $extra (0,1,5) {
    for my $p (0,.25,.5,.75,.9,.95,.99,1) {
        for my $draft (0,.25,.5,1,2,5) {
            my $v=model_rows($p,1,$extra);
            my @total=map {$trials[$_]{candidate}{total_ms}-$trials[$_]{candidate}{decode_ms}
                +$v->[$_]{decode_without_drafter_ms}+$draft*$v->[$_]{draft_decisions}} 0..4;
            push @curves,{conditional_guess_acceptance=>$p,drafter_ms=>$draft,extra_per_attempt_ms=>$extra,
                resident_tps=>summarize(map {512000/$_} @total),
                min_resident_gain=>minimum(map {$trials[$_]{control}{total_ms}/$total[$_]-1} 0..4),
                mean_useful_rows_per_eligible_attempt=>2+$p+$p*$p,
                all_three_guesses_accept_probability=>$p**3};
        }
        for my $target (@targets) {
            push @budgets,{target=>$target,conditional_guess_acceptance=>$p,extra_per_attempt_ms=>$extra,
                max_all_in_drafter_ms=>max_drafter($p,$target,$extra,1)};
        }
    }
    for my $target (@targets) {
        for my $draft (0,.25,.5,1,2,5) {
            my ($lo,$hi)=(0,1);
            my $is_possible=max_drafter(1,$target,$extra,1)>=$draft;
            if ($is_possible) {
                for (1..30) {
                    my $p=($lo+$hi)/2;
                    if (max_drafter($p,$target,$extra,1)>=$draft) {$hi=$p;} else {$lo=$p;}
                }
            }
            push @acceptance,{target=>$target,drafter_ms=>$draft,extra_per_attempt_ms=>$extra,
                minimum_conditional_guess_acceptance=>$is_possible ? $hi : undef};
        }
    }
}
for my $e (.25,.5,1) {
    for my $draft (.25,1) {
        my $v=model_rows(.95,$e,0);
        my @total=map {$trials[$_]{candidate}{total_ms}-$trials[$_]{candidate}{decode_ms}
            +$v->[$_]{decode_without_drafter_ms}+$draft*$v->[$_]{draft_decisions}} 0..4;
        push @eligibility,{eligibility=>$e,conditional_guess_acceptance=>.95,drafter_ms=>$draft,
            resident_tps=>summarize(map {512000/$_} @total),
            min_resident_gain=>minimum(map {$trials[$_]{control}{total_ms}/$total[$_]-1} 0..4)};
    }
}
print $json->pretty->encode({schema_version=>1,
    scope=>'measured full-verifier costs; hypothetical real-drafter acceptance and historical HTTP overhead',
    measurement_sha256=>sha256_hex(read_bytes($path)),perfect_draft_requests=>\@perfect,
    rejection_means=>\@rejection,serial_continuation_rows=>511,
    known_correct_seed=>JSON::PP::true,guessed_tokens=>3,
    unmeasured_late_rejection_and_real_drafter=>JSON::PP::true,
    accounting=>'interpolated measured complete batches; original per-position serial correction or measured early correction, whichever larger; observed rejection target delta/rollback/bookkeeping; direct setup/prefill/teardown; finite tails; all-in drafter cost on every decision including ineligible ones',
    historical_http_assumption=>{fixed_overhead_ms=>$http_fixed,extra_session_setup_ms=>$setup_delta,
        original_reference_requests=>\@reference,is_new_http_result=>JSON::PP::false},
    curves=>\@curves,drafter_budgets=>\@budgets,minimum_acceptance=>\@acceptance,
    eligibility_sensitivity=>\@eligibility});
