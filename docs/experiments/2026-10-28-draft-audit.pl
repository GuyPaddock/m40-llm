#!/usr/bin/env perl
# Independently reconstruct proposals from causal past, not executable text.
use strict;
use warnings;
use JSON::PP;
my $json=JSON::PP->new->canonical;
sub read_rows {open my $f,'<',$_[0] or die $!;return [map {$json->decode($_)} <$f>];}
sub check {die "$_[1]\n" unless $_[0];}
sub same {return $json->encode($_[0]) eq $json->encode($_[1]);}
sub proposal {
    my ($kind,$past,$prompt_len,$seed,$stops)=@_;
    my @h=(@$past,$seed);
    my $answer;
    if ($kind eq 'history-lookup' || $kind eq 'prompt-copy') {
        my ($min,$max)=$kind eq 'history-lookup' ? (3,8) : (4,16);
        $max=@h-($kind eq 'history-lookup' ? 1 : 0) if $max>@h-($kind eq 'history-lookup' ? 1 : 0);
        OUTER: for my $len (reverse $min..$max) {
            my $end=$kind eq 'history-lookup' ? @h-$len-1 : $prompt_len-$len-3;
            next if $end<0;
            for my $start (reverse 0..$end) {
                next unless same([@h[$start..$start+$len-1]],[@h[@h-$len..$#h]]);
                if ($start+$len+3<=@h) {
                    $answer={guesses=>[@h[$start+$len..$start+$len+2]],match_start=>$start,suffix_len=>$len};
                }
                last OUTER;
            }
        }
    } elsif ($kind eq 'online-backoff') {
        my %counts;
        for my $i (0..$#$past) {
            for my $n (1..4) {next if $n>$i;
                $counts{join(',',@$past[$i-$n..$i-1])}{$past->[$i]}++;}
        }
        my @context=@h;my @guesses;
        for (1..3) {
            my $next;
            for my $n (reverse 1..4) {
                next if $n>@context;
                my $c=$counts{join(',',@context[@context-$n..$#context])};
                next unless $c;
                ($next)=sort {$c->{$b}<=>$c->{$a} || $a<=>$b} keys %$c;
                last;
            }
            last unless defined $next;
            push @guesses,0+$next;push @context,0+$next;
        }
        $answer={guesses=>\@guesses,match_start=>undef,suffix_len=>undef} if @guesses==3;
    } else {die 'unexpected source';}
    $answer=undef if $answer && grep {$stops->{$_}} @{$answer->{guesses}};
    return $answer;
}
my $root='docs/experiments/';
my $capture=read_rows($root.'2026-10-28-draft-heldout.jsonl');
check($capture->[-1]{is_complete} && $capture->[-1]{device_bytes_after}==0,'capture incomplete');
my $quality=read_rows($root.'2026-10-27-maxwell-compound-runtime-quality.jsonl');
my %traces;
for my $r (@$quality) {
    my ($id,$g,$prompt,$budget);
    if ($r->{kind} eq 'q8_runtime_browser_generation' && $r->{label} eq 'candidate') {
        ($id,$g,$prompt,$budget)=('browser',$r->{generation},$r->{prompt_token_ids},512);
    } elsif ($r->{kind} eq 'model_quality_task') {
        ($id,$g,$prompt,$budget)=($r->{id},$r->{candidate},$r->{prompt_token_ids},$r->{max_tokens});
    } else {next;}
    $traces{$id}={prompt_token_ids=>$prompt,generated_token_ids=>$g->{generated_token_ids},
        max_tokens=>$budget,stop_ids=>$quality->[0]{candidate}{stop_ids}};
}
for my $r (grep {$_->{kind} eq 'draft_target_trace'} @$capture) {
    check(same($r->{trace}{generated_token_ids},$r->{repeat_generated_token_ids})
        && $r->{trace}{stop_reason} eq $r->{repeat_stop_reason},'unstable heldout IDs/stop');
    $traces{$r->{trace}{id}}=$r->{trace};
}
my $study=read_rows($root.'2026-10-28-draft-study.jsonl');
my @cases=grep {$_->{kind} eq 'causal_draft_case'} @$study;
check(@cases==63 && $study->[-1]{is_complete},'missing source/case results');
my ($rounds,$proposals)=(0,0);my %seen;
for my $case (@cases) {
    check(!$seen{"$case->{id}/$case->{source}"}++,'duplicate case');
    my $trace=$traces{$case->{id}} or die 'unknown case';
    my $first=$case->{trials}[0];
    my @past=@{$trace->{prompt_token_ids}};my $p=@past;my $row=0;
    my %stops=map {$_=>1} @{$trace->{stop_ids}};
    my (@reached,@matched,@histogram);@reached=(0)x3;@matched=(0)x3;@histogram=(0)x4;
    for my $r (@{$first->{rounds}}) {
        check($r->{input_row}==$row && $r->{absolute_position}==$p+$row
            && $r->{known_seed}==$trace->{generated_token_ids}[$row],'causal position/seed');
        my $guess=$trace->{max_tokens}-1-$row>=4
            ? proposal($case->{source},\@past,$p,$r->{known_seed},\%stops) : undef;
        check(same($guess,$r->{guesses}),'proposal differs from causal reconstruction');
        my $accepted=0;
        if ($guess) {
            for my $j (0..2) {
                last if $row+$j+1>=@{$trace->{generated_token_ids}}
                    || $guess->{guesses}[$j]!=$trace->{generated_token_ids}[$row+$j+1];
                $accepted++;
            }
            for my $j (0..2) {$reached[$j]++ if $accepted>=$j;$matched[$j]++ if $accepted>$j;}
            $histogram[$accepted]++;$proposals++;
        }
        my $remaining=@{$trace->{generated_token_ids}}-1-$row;
        my $progress=!$guess ? 1 : $accepted==3 ? 4 : $accepted+2;
        $progress=$remaining if $progress>$remaining;
        check($r->{accepted_guesses}==$accepted && $r->{progress}==$progress,'rejection progress');
        push @past,@{$trace->{generated_token_ids}}[$row..$row+$progress-1];
        $row+=$progress;$rounds++;
    }
    check($row==@{$trace->{generated_token_ids}}-1,'missing final input rows');
    check(same($case->{visited_stats}{conditional_reached},\@reached)
        && same($case->{visited_stats}{conditional_matched},\@matched)
        && same($case->{visited_stats}{rejected_guess_or_full_histogram},\@histogram),'joint acceptance counts');
    check(@{$case->{trials}}==5,'five CPU trials');
    for my $trial (@{$case->{trials}}) {
        my $cpu=$trial->{construction_ns};
        $cpu+=$_->{proposal_ns}+$_->{maintenance_ns} for @{$trial->{rounds}};
        check($cpu==$trial->{cpu_elapsed_ns},'all-in proposal costs');
        check(same([map {[@{$_}{qw(input_row guesses accepted_guesses progress)}]} @{$trial->{rounds}}],
            [map {[@{$_}{qw(input_row guesses accepted_guesses progress)}]} @{$first->{rounds}}]),'varying causal replay');
    }
}
print $json->pretty->encode({is_passed=>JSON::PP::true,cases=>scalar @cases,
    independently_reconstructed_rounds=>$rounds,independently_reconstructed_proposals=>$proposals,
    heldout_stable_cases=>8,cpu_trials_per_case=>5,scope=>'independent causal proposal and accounting audit, not measured GPU/HTTP throughput'});
