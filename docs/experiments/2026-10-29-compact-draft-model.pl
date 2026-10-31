#!/usr/bin/env perl
# Reuse the preserved finite verifier recurrence; never edit its evidence.
use strict;
use warnings;
use JSON::PP;
use Digest::SHA qw(sha256_hex);
my $json=JSON::PP->new->canonical->pretty;
sub bytes {open my $f,'<',$_[0] or die $!;local $/;return <$f>;}
sub rows {return [map {decode_json($_)} grep {length} split /\n/,bytes($_[0])];}
sub med {my @v=sort {$a<=>$b} @_;return @v%2 ? $v[int(@v/2)] : ($v[@v/2-1]+$v[@v/2])/2;}
sub dist {my @v=sort {$a<=>$b} @_;return {min=>$v[0],median=>med(@v),max=>$v[-1]};}

my $root='docs/experiments/';
my $source_path=$root.'2026-10-28-direct-verifier-model.pl';
my $source=bytes($source_path);
# Import only the existing helpers and calculate_decode function, stopping
# before that standalone program's self-test/measurement entrypoint.
my ($library)=$source=~/\A(.*?)^if \(\@ARGV && \$ARGV\[0\] eq '--self-test'\)/ms;
die 'preserved recurrence unavailable' unless defined $library;
eval "package FrozenTarget; $library; 1;" or die $@;
my $old=decode_json(bytes($root.'2026-10-28-direct-verifier-model.json'));
my $target=rows($root.'2026-10-28-direct-verifier-measurement.jsonl');
my $cost_path=$root.'2026-10-29-compact-draft-cost.jsonl';
my $cost=rows($cost_path);
my ($decision)=grep {$_->{kind} eq 'compact_cost_decision'} @$cost;
my ($setup)=grep {$_->{kind} eq 'compact_cost_setup'} @$cost;
die 'incomplete cost or target evidence' unless $decision->{is_complete}
    && $decision->{tracked_device_bytes_after}==0 && $target->[-1]{is_complete};
my @trials;
for my $i (0..4) {
    my %pair;
    for my $row (grep {$_->{kind} eq 'direct_verifier_trial' && $_->{trial}==$i} @$target) {
        $pair{$row->{is_direct} ? 'candidate' : 'control'}=$row->{result};
    }
    die 'missing target pair' unless keys(%pair)==2;
    my $perfect=FrozenTarget::calculate_decode(\%pair,$old->{rejection_means},1,1,0);
    die 'perfect recurrence no longer reproduces measured decode'
        unless abs($perfect->{decode_without_drafter_ms}-$pair{candidate}{decode_ms})<1e-6;
    push @trials,\%pair;
}
my %cache;
sub projected {
    my ($draft_ms,$p,$coverage,$extra)=@_;
    my $key="$p,$coverage,$extra";
    $cache{$key}//=[map {FrozenTarget::calculate_decode($_,$old->{rejection_means},$p,$coverage,$extra)} @trials];
    my @samples;
    for my $i (0..4) {
        my $c=$cache{$key}[$i];my $t=$trials[$i];
        my $decode=$c->{decode_without_drafter_ms}+$draft_ms*$c->{draft_decisions};
        my $resident=$t->{candidate}{total_ms}-$t->{candidate}{decode_ms}+$decode;
        my $http=$old->{historical_http_assumption}{fixed_overhead_ms}
            +$old->{historical_http_assumption}{extra_session_setup_ms}+$decode;
        push @samples,{pair=>$i,%$c,proposal_charged_ms=>$draft_ms*$c->{draft_decisions},
            resident_ms=>$resident,http_ms=>$http,projected_resident_tps=>512000/$resident,
            projected_http_tps=>512000/$http,resident_gain=>$t->{control}{total_ms}/$resident-1};
    }
    return {samples=>\@samples,resident_tps=>dist(map {$_->{projected_resident_tps}} @samples),
        http_tps=>dist(map {$_->{projected_http_tps}} @samples),resident_gain=>dist(map {$_->{resident_gain}} @samples)};
}
my (@scenarios,@thresholds);
for my $kind (qw(median_ms p95_ms)) {
    my $draft=$decision->{$kind};
    for my $coverage (.5,.75,1) {
        for my $p (.9,.95,.99,1) {
            push @scenarios,{cost_statistic=>$kind,proposal_ms=>$draft,conditional_acceptance=>$p,
                full_batch_probability=>$p**3,eligibility=>$coverage,%{projected($draft,$p,$coverage,0)}};
        }
        for my $target (53.06,56.153149) {
            my ($lo,$hi)=(0,1);
            my $possible=projected($draft,1,$coverage,0)->{http_tps}{min}>=$target;
            if ($possible) {for (1..30) {my $p=($lo+$hi)/2;
                if (projected($draft,$p,$coverage,0)->{http_tps}{min}>=$target){$hi=$p;}else{$lo=$p;}
            }}
            push @thresholds,{cost_statistic=>$kind,proposal_ms=>$draft,eligibility=>$coverage,target_http_tps=>$target,
                minimum_equal_conditional_acceptance=>$possible?$hi:undef};
        }
    }
}
my @overhead;
for my $extra (1,5) {
    push @overhead,{unmeasured_extra_per_attempt_ms=>$extra,conditional_acceptance=>.95,eligibility=>1,
        %{projected($decision->{p95_ms},.95,1,$extra)}};
}
my $with_init=projected($decision->{p95_ms},.95,1,0);
my $initialization=$setup->{proposal_setup}{total_allocation_upload_ms};
print $json->encode({schema_version=>1,is_http_measured=>JSON::PP::false,is_acceptance_measured=>JSON::PP::false,
    recurrence_source=>$source_path,recurrence_sha256=>sha256_hex($source),cost_sha256=>sha256_hex(bytes($cost_path)),
    historical_http_assumption=>$old->{historical_http_assumption},
    assumptions=>'equal conditional acceptance at all three guesses; eligibility scenarios are hypothetical, correction/rollback/finite tails retained; late rejection and live target cache interference remain unmeasured',
    recurrence_perfect_reproduction_cases=>5,scenarios=>\@scenarios,minimum_acceptance=>\@thresholds,
    extra_attempt_sensitivity=>\@overhead,
    first_request_allocation_upload_sensitivity=>{extra_ms=>$initialization,conditional_acceptance=>.95,eligibility=>1,
        projected_http_tps=>dist(map {512000/($_->{http_ms}+$initialization)} @{$with_init->{samples}}),
        excludes=>'serialized trained-weight loading remains unmeasured; synthetic host coefficient generation is not a trained load'},
    next=>'User checkpoint before dataset manifest approval, teacher-data generation or training'});
