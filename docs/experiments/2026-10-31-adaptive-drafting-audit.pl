#!/usr/bin/env perl
# Reconstruct causal decisions and finite costs from retained source records.
use strict;
use warnings;
use JSON::PP;
use List::Util qw(sum min max);
use Digest::SHA qw(sha256_hex);
my $json=JSON::PP->new->canonical;
sub read_rows { open my $f,'<',$_[0] or die $!; return [map {$json->decode($_)} <$f>]; }
sub check { die "$_[1]\n" unless $_[0]; }
sub same { return $json->encode($_[0]) eq $json->encode($_[1]); }
sub close_number {
    my ($actual,$expected,$label)=@_;
    check(defined($actual) && abs($actual-$expected)<1e-8*max(1,abs($expected)),$label);
}
sub median { my @x=sort {$a<=>$b} @_; return $x[int(@x/2)]; }
sub digest { open my $f,'<:raw',$_[0] or die $!;local $/; return sha256_hex(<$f>); }
my $root='docs/experiments/';
my $held_path=$root.'2026-10-28-draft-heldout.jsonl';
my $target_path=$root.'2026-10-28-direct-verifier-measurement.jsonl';
my $quality=read_rows($root.'2026-10-27-maxwell-compound-runtime-quality.jsonl');
my $held=read_rows($held_path);
my $target=read_rows($target_path);
my (%traces,@pairs,@reject);
for my $row (@$quality) {
    my ($id,$generation,$prompt,$budget);
    if ($row->{kind} eq 'q8_runtime_browser_generation' && $row->{label} eq 'candidate') {
        ($id,$generation,$prompt,$budget)=('browser',$row->{generation},$row->{prompt_token_ids},512);
    } elsif ($row->{kind} eq 'model_quality_task') {
        ($id,$generation,$prompt,$budget)=($row->{id},$row->{candidate},$row->{prompt_token_ids},$row->{max_tokens});
    } else {next;}
    $traces{$id}={prompt_token_ids=>$prompt,generated_token_ids=>$generation->{generated_token_ids},
        max_tokens=>$budget,stop_ids=>$quality->[0]{candidate}{stop_ids},stop_reason=>$generation->{metrics}{stop_reason}};
}
for my $row (grep {$_->{kind} eq 'draft_target_trace'} @$held) {
    check(same($row->{trace}{generated_token_ids},$row->{repeat_generated_token_ids}) &&
        $row->{trace}{stop_reason} eq $row->{repeat_stop_reason},'unstable retained trace');
    $traces{$row->{trace}{id}}=$row->{trace};
}
check(keys(%traces)==21 && $held->[-1]{is_complete} && $target->[-1]{is_complete},'source completeness');
for my $index (0..4) {
    my @pair;
    for my $is_direct (0,1) {
        my @rows=grep {$_->{kind} eq 'direct_verifier_trial' && $_->{trial}==$index &&
            ($_->{is_direct}?1:0)==$is_direct} @$target;
        check(@rows==1,'target pair uniqueness');push @pair,$rows[0]{result};
    }
    push @pairs,\@pair;
}
for my $accepted (0..3) {
    my @rows=map {$_->{result}} grep {$_->{kind} eq 'direct_verifier_rejection' &&
        $_->{result}{accepted_prefix}==$accepted} @$target;
    check(@rows==5,'rejection samples');
    my %average;for my $field (qw(target_ms rollback_ms correction_ms elapsed_ms)) {
        $average{$field}=sum(map {$_->{$field}} @rows)/5;
    }push @reject,\%average;
}
sub serial_cost {my ($control,$position)=@_;return $control->{serial_step_ms}[min(510,max(0,$position-65))];}
sub batch_cost {
    my ($direct,$position)=@_;my $offset=min(504,max(0,$position-65));
    my $index=int($offset/4);my $first=$direct->{batch_ms}[$index];
    return $index==126 ? $first : $first+($direct->{batch_ms}[$index+1]-$first)*($offset%4)/4;
}
sub attempt_cost {
    my ($pair,$position,$guesses,$has_correction,$scale,$extra)=@_;
    my ($control,$direct)=@$pair;my $verifier=batch_cost($direct,$position)+$extra;my $correction=0;
    if ($guesses<3) {
        my $accepted=$guesses+1;my $reject=$reject[$accepted];
        $verifier+=max(0,$reject->{target_ms}-$direct->{batch_ms}[0])+$reject->{rollback_ms}+
            max(0,$reject->{elapsed_ms}-$reject->{target_ms}-$reject->{rollback_ms}-$reject->{correction_ms});
        $correction=max($reject->{correction_ms},serial_cost($control,$position+$accepted)) if $has_correction;
    }return $scale*$verifier+$correction;
}
sub costs_at {
    my ($position,$scale)=@_;my (@serial,@candidate);
    for my $guesses (0..3) {
        my $progress=min(4,$guesses+2);
        push @serial,median(map {my $control=$_->[0];sum(map {serial_cost($control,$position+$_)} 0..$progress-1)} @pairs);
        push @candidate,median(map {attempt_cost($_,$position,$guesses,$guesses<3,$scale,0)+
            $scale*$_->[1]{decode_other_ms}/130} @pairs);
    }return {serial_ms=>\@serial,candidate_ms=>\@candidate};
}
sub propose {
    my ($past,$prompt_length,$seed,$stops)=@_;my @history=(@$past,$seed);my $proposal;
    OUTER:for my $width (reverse 4..min(16,scalar @history)) {
        my $last=$prompt_length-$width-3;next if $last<0;
        for my $start (reverse 0..$last) {
            next unless same([@history[$start..$start+$width-1]],[@history[@history-$width..$#history]]);
            $proposal={guesses=>[@history[$start+$width..$start+$width+2]],match_start=>$start,suffix_len=>$width};
            last OUTER;
        }
    }
    return undef if $proposal && grep {$stops->{$_}} @{$proposal->{guesses}};
    return $proposal;
}
sub audit {
    my ($rows)=@_;my $header=$rows->[0];my $end=$rows->[-1];
    check(@$rows==131 && $header->{kind} eq 'adaptive_manifest' && $end->{kind} eq 'adaptive_complete' &&
        $end->{is_complete} && $end->{cases}==126 && $end->{trials}==630,'complete adaptive grid');
    check($header->{target_sha256} eq digest($target_path) && $header->{heldout_source_sha256} eq digest($held_path),
        'retained source identity');
    check(@{$header->{traces}}==21 && $header->{is_retrospective} && !$header->{is_measured_gpu_speedup},'scope');
    my %header_seen;
    for my $trace (@{$header->{traces}}) {
        check(!$header_seen{$trace->{id}}++,'duplicate manifest trace');
        my $source=$traces{$trace->{id}} or die 'unknown source trace';
        check(same([@{$trace}{qw(prompt_token_ids generated_token_ids max_tokens stop_ids stop_reason)}],
            [@{$source}{qw(prompt_token_ids generated_token_ids max_tokens stop_ids stop_reason)}]),'changed trace');
    }
    my (%tables,%seen);my ($round_count,$attempts,$skipped)=(0,0,0);
    for my $table (grep {$_->{kind} eq 'adaptive_cost_table'} @$rows) {
        my $scale=$table->{scale};check(($scale==1 || $scale==1.25 || $scale==2) && !$tables{$scale},'unique hardware table');
        check(@{$table->{costs}}==505,'cost table bounds');
        for my $position (65..569) {
            my $expected=costs_at($position,$scale);my $actual=$table->{costs}[$position-65];
            for my $field (qw(serial_ms candidate_ms)) {for my $guess (0..3) {
                close_number($actual->{$field}[$guess],$expected->{$field}[$guess],'hardware table outcome');
            }}
        }$tables{$scale}=$table->{costs};
    }
    check(keys(%tables)==3,'all hardware sensitivities');
    for my $case (grep {$_->{kind} eq 'adaptive_case'} @$rows) {
        my ($id,$policy,$scale)=@{$case}{qw(id policy scale)};
        check(!$seen{"$id/$policy/$scale"}++ && ($policy eq 'adaptive' || $policy eq 'always_attempt'),'unique policy case');
        my $trace=$traces{$id} or die 'unknown case';my $table=$tables{$scale} or die 'unknown scale';
        check(@{$case->{trials}}==5,'five trials');my $first=$case->{trials}[0];
        my @past=@{$trace->{prompt_token_ids}};my $prompt=@past;my $row=0;my @hist=(0)x4;my $last;
        my %stops=map {$_=>1} @{$trace->{stop_ids}};
        check(@{$first->{decisions}}==@{$first->{replay}{rounds}},'decision/round coverage');
        for my $index (0..$#{$first->{decisions}}) {
            my $record=$first->{decisions}[$index];my $round=$first->{replay}{rounds}[$index];
            my $position=$prompt+$row;my $remaining=$trace->{max_tokens}-1-$row;my $seed=$trace->{generated_token_ids}[$row];
            my $proposal=$remaining>=4 ? propose(\@past,$prompt,$seed,\%stops) : undef;
            check($record->{input_row}==$row && $record->{absolute_position}==$position && $record->{remaining_budget}==$remaining &&
                $round->{input_row}==$row && $round->{absolute_position}==$position && $round->{known_seed}==$seed &&
                same($record->{proposal},$proposal),'causal position/proposal');
            my ($should,$reason,$full,$expected)=(0,'budget',undef,undef);
            if ($remaining>=4) {
                if (!$proposal) {$reason='source_miss';}
                elsif ($policy eq 'always_attempt') {($should,$reason)=(1,'always_attempt');}
                elsif ($position<65 || $position>569) {$reason='unmeasured_position';}
                else {
                    my $cost=$table->[$position-65];$full=$cost->{serial_ms}[3]/($cost->{candidate_ms}[3]+.1)-1;
                    if ($full<.03) {$reason='hardware_ceiling';}
                    elsif (!sum(@hist)) {($should,$reason)=(1,'first_probe');}
                    else {
                        my $count=sum(@hist);my $serial=sum(map {$cost->{serial_ms}[$_]*$hist[$_]} 0..3)/$count;
                        my $candidate=sum(map {$cost->{candidate_ms}[$_]*$hist[$_]} 0..3)/$count;
                        $expected=$serial/($candidate+.1)-1;
                        if ($expected>=.03) {($should,$reason)=(1,'expected_gain');}
                        elsif (defined($last) && $position-$last>=32) {($should,$reason)=(1,'recovery_probe');}
                        else {$reason='expected_loss';}
                    }
                }
            }
            my $decision=$record->{decision};
            check(($decision->{should_attempt}?1:0)==$should && $decision->{reason} eq $reason &&
                same($decision->{histogram},\@hist) && same($decision->{last_attempt_position},$last),'causal controller decision');
            for my $item (['full_acceptance_gain',$full],['expected_gain',$expected]) {
                defined($item->[1]) ? close_number($decision->{$item->[0]},$item->[1],'decision cost gain') :
                    check(!defined($decision->{$item->[0]}),'unexpected cost gain');
            }
            my $accepted=0;
            if ($should) {for my $guess (0..2) {
                last if $row+$guess+1>=@{$trace->{generated_token_ids}} ||
                    $proposal->{guesses}[$guess]!=$trace->{generated_token_ids}[$row+$guess+1];$accepted++;
            }$attempts++;}elsif ($proposal) {$skipped++;}
            my $progress=$should ? min(@{$trace->{generated_token_ids}}-1-$row,$accepted==3?4:$accepted+2) : 1;
            check(same($round->{guesses},$should?$proposal:undef) && $round->{accepted_guesses}==$accepted &&
                $round->{progress}==$progress && $record->{next_input_row}==$row+$progress &&
                same($record->{observed_accepted_guesses},$should?$accepted:undef),'attempt/commit accounting');
            if ($should && $policy eq 'adaptive') {$hist[$accepted]++;$last=$position;}
            push @past,@{$trace->{generated_token_ids}}[$row..$row+$progress-1];$row+=$progress;$round_count++;
        }
        check($row==@{$trace->{generated_token_ids}}-1,'finite request coverage');
        my $cpu=0;
        for my $trial (@{$case->{trials}}) {
            check(same($trial->{decisions},$first->{decisions}),'varying decisions');
            my $replay=$trial->{replay};my $ns=$replay->{construction_ns}+$trial->{teardown_ns};
            $ns+=$_->{proposal_ns}+$_->{maintenance_ns} for @{$replay->{rounds}};
            check($ns==$replay->{cpu_elapsed_ns},'CPU timing accounting');$cpu=max($cpu,$ns/1e6);
            check(same([map {[@{$_}{qw(input_row guesses accepted_guesses progress)}]} @{$replay->{rounds}}],
                [map {[@{$_}{qw(input_row guesses accepted_guesses progress)}]} @{$first->{replay}{rounds}}]),'varying visit path');
        }
        my $projections=$case->{projections};check(@{$projections->{scenarios}}==3,'overhead scenarios');
        for my $scenario (@{$projections->{scenarios}}) {
            my $extra=$scenario->{extra_per_attempt_ms};check($extra==0 || $extra==1 || $extra==5,'unknown overhead');
            check(@{$scenario->{model_samples}}==5,'target samples');
            my @resident_gains;
            for my $sample (0..4) {
                my ($control,$direct)=@{$pairs[$sample]};my $serial=sum(map {serial_cost($control,$prompt+$_)} 0..$row-1);
                my $target=0;
                for my $round (@{$first->{replay}{rounds}}) {
                    $target+=$round->{guesses} ? attempt_cost($pairs[$sample],$round->{absolute_position},$round->{accepted_guesses},
                        $round->{progress}>$round->{accepted_guesses}+1,$scale,$extra) : serial_cost($control,$round->{absolute_position});
                }
                $target+=$scale*$direct->{decode_other_ms}/130*@{$first->{replay}{rounds}};
                my $actual=$scenario->{model_samples}[$sample];
                close_number($actual->{serial_decode_model_ms},$serial,'serial finite cost');
                close_number($actual->{candidate_decode_model_ms},$target,'candidate finite cost');
                close_number($actual->{all_in_cpu_ms},$cpu,'charged maximum CPU trial');
                close_number($actual->{candidate_zero_draft_gain},$serial/$target-1,'zero-draft ceiling');
                my $setup=max(0,$direct->{session_setup_ms}+$direct->{session_teardown_ms}-
                    $control->{session_setup_ms}-$control->{session_teardown_ms});
                close_number($actual->{extra_session_ms},$setup,'session overhead');
                my $fixed=$id eq 'browser' ? $control->{total_ms}-$control->{decode_ms} : $trace->{serial_fixed_ms};
                if (defined($fixed)) {
                    my $gain=($fixed+$serial)/($fixed+$target+$cpu+$setup)-1;
                    close_number($actual->{resident_gain},$gain,'complete resident gain');push @resident_gains,$gain;
                }else {check(!defined($actual->{resident_gain}),'missing prefill cannot qualify resident gain');}
            }
            if (@resident_gains) {
                close_number($scenario->{resident_gain}{min},min(@resident_gains),'minimum gain');
                close_number($scenario->{resident_gain}{median},median(@resident_gains),'median gain');
                close_number($scenario->{resident_gain}{max},max(@resident_gains),'maximum gain');
            }else {check(!defined($scenario->{resident_gain}),'unknown resident summary');}
        }
    }
    check(keys(%seen)==126,'all trace/policy/scale combinations');
    return {is_passed=>JSON::PP::true,cases=>126,trials=>630,causal_rounds=>$round_count,
        attempted_batches=>$attempts,skipped_proposals=>$skipped,
        scope=>'independent source, policy, target cost and CPU accounting; no GPU performance claim'};
}
my $path=shift @ARGV or die "usage: $0 RESULTS.jsonl [--self-test]\n";
my $rows=read_rows($path);my $result=audit($rows);
if (@ARGV && $ARGV[0] eq '--self-test') {
    # Missing completion, corrupted costs, future labels and changed proposals
    # must be rejected after the unmodified report passes the same full audit.
    my @mutations=(
        sub {pop @{$_[0]};},
        sub {$_[0][1]{costs}[0]{candidate_ms}[3]=0;},
        sub {$_[0][2]{trials}[0]{replay}{cpu_elapsed_ns}=0;},
        sub {$_[0][2]{trials}[0]{decisions}[0]{next_input_row}=999;},
        sub {$_[0][2]{trials}[0]{decisions}[0]{proposal}={guesses=>[0,0,0]};},
        sub {$_[0][3]{trials}[0]{decisions}[0]{decision}{histogram}=[99,0,0,0];},
        sub {$_[0][2]{projections}{scenarios}[0]{model_samples}[0]{candidate_decode_model_ms}=0;},
    );
    for my $mutate (@mutations) {my $changed=$json->decode($json->encode($rows));$mutate->($changed);
        check(!eval {audit($changed);1},'corrupted report accepted');}
    $result->{rejected_corruptions}=scalar @mutations;
}
print $json->pretty->encode($result);
